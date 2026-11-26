"""Read-only asset registration, digest verification, and waiting/blocked triage.

Never downloads weights, datasets, tokenizers, statistics, or simulators.
"""

from __future__ import annotations

import hashlib
import os
import stat
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, TypeAlias

from core.contracts.common import AssetSpec, ScopeKind

AssetStatus: TypeAlias = Literal[
    "ready",
    "waiting",
    "blocked",
    "not_required",
]
OverallAssetStatus: TypeAlias = Literal["ready", "waiting", "blocked"]


@dataclass(frozen=True, slots=True)
class AssetRecord:
    """Verified read-only status record for one declared asset."""

    id: str
    kind: str
    required_for: tuple[str, ...]
    source: str | None
    revision: str | None
    expected_sha256: str | None
    access_policy: str
    local_path: str | None
    observed_sha256: str | None
    size_bytes: int | None
    status: AssetStatus
    reason_code: str | None = None
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "required_for": list(self.required_for),
            "source": self.source,
            "revision": self.revision,
            "expected_sha256": self.expected_sha256,
            "access_policy": self.access_policy,
            "local_path": self.local_path,
            "observed_sha256": self.observed_sha256,
            "size_bytes": self.size_bytes,
            "status": self.status,
            "reason_code": self.reason_code,
            "detail": self.detail,
        }


@dataclass(frozen=True, slots=True)
class AssetTriageReport:
    """Aggregated read-only asset triage written to `analysis/resource_manifest.yaml`."""

    required_scope: str | None
    overall_status: OverallAssetStatus
    records: tuple[AssetRecord, ...]
    waiting_reasons: tuple[str, ...] = ()
    blocked_reasons: tuple[str, ...] = ()
    unknowns: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "required_scope": self.required_scope,
            "overall_status": self.overall_status,
            "records": [rec.to_dict() for rec in self.records],
            "waiting_reasons": list(self.waiting_reasons),
            "blocked_reasons": list(self.blocked_reasons),
            "unknowns": list(self.unknowns),
        }


class _AssetSymlinkError(ValueError):
    """Internal marker when any top-level, nested, or dangling symlink is encountered."""

    def __init__(self, rel_path: str) -> None:
        self.rel_path = rel_path
        super().__init__(f"symlink not allowed in asset path: {rel_path}")


def _lstat_reject_symlink(path: Path, display: str) -> os.stat_result:
    st = os.lstat(path)
    if stat.S_ISLNK(st.st_mode):
        raise _AssetSymlinkError(display)
    return st


def _hash_path_read_only(path: Path) -> tuple[str, int]:
    """Compute deterministic SHA-256 for a regular file or directory via `lstat`/`scandir`.

    Rejects top-level, nested, and dangling symlinks at every level.
    """
    st = _lstat_reject_symlink(path, str(path))
    if stat.S_ISREG(st.st_mode):
        digest = hashlib.sha256()
        size = 0
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(chunk)
                size += len(chunk)
        return digest.hexdigest(), size

    if stat.S_ISDIR(st.st_mode):
        digest = hashlib.sha256()
        total_size = 0
        files: list[tuple[str, Path]] = []

        def _scan(current: Path, prefix: str) -> None:
            with os.scandir(current) as iterator:
                entries = sorted(iterator, key=lambda item: item.name)
            for entry in entries:
                rel = f"{prefix}/{entry.name}" if prefix else entry.name
                entry_path = Path(entry.path)
                entry_st = _lstat_reject_symlink(entry_path, rel)
                if stat.S_ISDIR(entry_st.st_mode):
                    _scan(entry_path, rel)
                elif stat.S_ISREG(entry_st.st_mode):
                    files.append((rel, entry_path))
                else:
                    raise ValueError(f"unsupported non-regular entry in asset dir: {rel}")

        _scan(path, "")
        for rel, file_path in sorted(files, key=lambda item: item[0]):
            file_sha, file_size = _hash_path_read_only(file_path)
            total_size += file_size
            digest.update(f"{rel}:{file_size}:{file_sha}\n".encode("utf-8"))
        return digest.hexdigest(), total_size

    raise ValueError(f"unsupported asset path type: {path}")


def triage_assets(
    assets: Iterable[AssetSpec],
    *,
    required_scope: ScopeKind | None = None,
) -> AssetTriageReport:
    """Classify declared assets into `ready`, `waiting` (`WAITING_ASSET`), or `blocked`.

    Rules (strictly read-only, zero automatic downloads):
    - Local path exists: compute observed SHA-256.
      - If `expected_sha256` is declared and differs -> `blocked` (`ASSET_DIGEST_MISMATCH`).
      - If `expected_sha256` is missing for a required asset -> `blocked` (`ASSET_EXPECTED_DIGEST_MISSING`).
      - Otherwise -> `ready`.
    - Local path missing (or `None`):
      - If asset is not required for `required_scope` -> `not_required` (recorded as unknown if missing identity).
      - If identity (`source` and `expected_sha256`) is declared -> `waiting` (`WAITING_ASSET`).
      - Otherwise (unknown required asset) -> `blocked` (`UNKNOWN_REQUIRED_ASSET`).
    """
    records: list[AssetRecord] = []
    waiting_reasons: list[str] = []
    blocked_reasons: list[str] = []
    unknowns: list[str] = []

    def _add_unique(target: list[str], value: str) -> None:
        if value not in target:
            target.append(value)

    for spec in assets:
        is_required = required_scope is None or required_scope in spec.required_for
        raw_local_path = Path(spec.local_path) if spec.local_path else None

        # Check lstat BEFORE resolve()/exists() so top-level and dangling symlinks are caught.
        raw_lstat: os.stat_result | None = None
        if raw_local_path is not None:
            try:
                raw_lstat = os.lstat(raw_local_path)
            except FileNotFoundError:
                raw_lstat = None

        if raw_local_path is not None and raw_lstat is not None:
            if stat.S_ISLNK(raw_lstat.st_mode):
                _add_unique(blocked_reasons, "ASSET_SYMLINK_REJECTED")
                records.append(
                    AssetRecord(
                        id=spec.id,
                        kind=spec.kind,
                        required_for=tuple(spec.required_for),
                        source=spec.source,
                        revision=spec.revision,
                        expected_sha256=spec.expected_sha256,
                        access_policy=spec.access_policy,
                        local_path=str(raw_local_path),
                        observed_sha256=None,
                        size_bytes=None,
                        status="blocked",
                        reason_code="ASSET_SYMLINK_REJECTED",
                        detail="top-level or dangling symlink asset path rejected",
                    )
                )
                continue

            local_path_obj = raw_local_path.resolve()
            try:
                observed_sha, size_bytes = _hash_path_read_only(local_path_obj)
            except _AssetSymlinkError as sym_err:
                _add_unique(blocked_reasons, "ASSET_SYMLINK_REJECTED")
                records.append(
                    AssetRecord(
                        id=spec.id,
                        kind=spec.kind,
                        required_for=tuple(spec.required_for),
                        source=spec.source,
                        revision=spec.revision,
                        expected_sha256=spec.expected_sha256,
                        access_policy=spec.access_policy,
                        local_path=str(local_path_obj),
                        observed_sha256=None,
                        size_bytes=None,
                        status="blocked",
                        reason_code="ASSET_SYMLINK_REJECTED",
                        detail=f"nested symlink rejected inside asset directory: {sym_err.rel_path}",
                    )
                )
                continue
            if spec.expected_sha256 is not None and observed_sha != spec.expected_sha256:
                _add_unique(blocked_reasons, "ASSET_DIGEST_MISMATCH")
                records.append(
                    AssetRecord(
                        id=spec.id,
                        kind=spec.kind,
                        required_for=tuple(spec.required_for),
                        source=spec.source,
                        revision=spec.revision,
                        expected_sha256=spec.expected_sha256,
                        access_policy=spec.access_policy,
                        local_path=str(local_path_obj),
                        observed_sha256=observed_sha,
                        size_bytes=size_bytes,
                        status="blocked",
                        reason_code="ASSET_DIGEST_MISMATCH",
                        detail=(
                            f"asset {spec.id!r} observed sha256 {observed_sha} "
                            f"!= expected {spec.expected_sha256}"
                        ),
                    )
                )
            elif spec.expected_sha256 is None and is_required:
                _add_unique(blocked_reasons, "ASSET_EXPECTED_DIGEST_MISSING")
                records.append(
                    AssetRecord(
                        id=spec.id,
                        kind=spec.kind,
                        required_for=tuple(spec.required_for),
                        source=spec.source,
                        revision=spec.revision,
                        expected_sha256=None,
                        access_policy=spec.access_policy,
                        local_path=str(local_path_obj),
                        observed_sha256=observed_sha,
                        size_bytes=size_bytes,
                        status="blocked",
                        reason_code="ASSET_EXPECTED_DIGEST_MISSING",
                        detail=f"required asset {spec.id!r} lacks expected_sha256",
                    )
                )
            else:
                records.append(
                    AssetRecord(
                        id=spec.id,
                        kind=spec.kind,
                        required_for=tuple(spec.required_for),
                        source=spec.source,
                        revision=spec.revision,
                        expected_sha256=spec.expected_sha256,
                        access_policy=spec.access_policy,
                        local_path=str(local_path_obj),
                        observed_sha256=observed_sha,
                        size_bytes=size_bytes,
                        status="ready" if is_required else "not_required",
                        reason_code=None,
                        detail="verified read-only local asset",
                    )
                )
            continue

        # Local file/directory is absent. Never auto-download.
        if not is_required:
            if spec.source is None or spec.expected_sha256 is None:
                _add_unique(unknowns, f"asset:{spec.id}:undeclared_identity")
            records.append(
                AssetRecord(
                    id=spec.id,
                    kind=spec.kind,
                    required_for=tuple(spec.required_for),
                    source=spec.source,
                    revision=spec.revision,
                    expected_sha256=spec.expected_sha256,
                    access_policy=spec.access_policy,
                    local_path=spec.local_path,
                    observed_sha256=None,
                    size_bytes=None,
                    status="not_required",
                    reason_code=None,
                    detail="not required for current intake scope",
                )
            )
            continue

        has_declared_identity = bool(spec.source) and bool(spec.expected_sha256)
        if has_declared_identity:
            _add_unique(waiting_reasons, "WAITING_ASSET")
            records.append(
                AssetRecord(
                    id=spec.id,
                    kind=spec.kind,
                    required_for=tuple(spec.required_for),
                    source=spec.source,
                    revision=spec.revision,
                    expected_sha256=spec.expected_sha256,
                    access_policy=spec.access_policy,
                    local_path=spec.local_path,
                    observed_sha256=None,
                    size_bytes=None,
                    status="waiting",
                    reason_code="WAITING_ASSET",
                    detail=f"declared asset {spec.id!r} missing locally; auto-download disabled",
                )
            )
        else:
            _add_unique(blocked_reasons, "UNKNOWN_REQUIRED_ASSET")
            records.append(
                AssetRecord(
                    id=spec.id,
                    kind=spec.kind,
                    required_for=tuple(spec.required_for),
                    source=spec.source,
                    revision=spec.revision,
                    expected_sha256=spec.expected_sha256,
                    access_policy=spec.access_policy,
                    local_path=spec.local_path,
                    observed_sha256=None,
                    size_bytes=None,
                    status="blocked",
                    reason_code="UNKNOWN_REQUIRED_ASSET",
                    detail=f"required asset {spec.id!r} has neither local file nor declared identity",
                )
            )

    if blocked_reasons:
        overall: OverallAssetStatus = "blocked"
    elif waiting_reasons:
        overall = "waiting"
    else:
        overall = "ready"

    return AssetTriageReport(
        required_scope=required_scope,
        overall_status=overall,
        records=tuple(records),
        waiting_reasons=tuple(waiting_reasons),
        blocked_reasons=tuple(blocked_reasons),
        unknowns=tuple(unknowns),
    )

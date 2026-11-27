"""Stage 1 (`intake`) preparation lifecycle: `init_preparation`, `analyze_intake`, `freeze_intake`."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypeAlias

import yaml

from core.artifacts import PackageRef, publish
from core.contracts import (
    ContractParseError,
    ContractValidationError,
    HandoffResult,
    load_intake_task_pack,
)
from core.contracts.intake_v2 import IntakeTaskPackV2
from .handoff import V2_ANALYSIS_FILES, import_handoff

from .assets import triage_assets
from .discovery import DiscoveryError, discover_entrypoints
from .probe import ProbeResult, run_startup_probe, validate_probe_argv
from .source import (
    SourceError,
    create_recursive_source_bundles,
    snapshot_source,
    verify_source_unchanged,
)

ProbeMode: TypeAlias = Literal["none", "startup"]
FreezeStatus: TypeAlias = Literal["completed", "waiting", "blocked", "error"]

STATE_FILENAME = "prep_state.json"


@dataclass(frozen=True, slots=True)
class FreezeResult:
    """Result of attempting to validate and atomically freeze an intake package."""

    status: FreezeStatus
    summary: str
    package_ref: PackageRef | None = None
    release_record_path: Path | None = None
    verdict: dict[str, Any] | None = None
    reason_codes: tuple[str, ...] = ()
    observed_scope: tuple[str, ...] = ()
    unknowns: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "summary": self.summary,
            "package_dir": str(self.package_ref.package_dir)
            if self.package_ref is not None
            else None,
            "manifest_sha256": self.package_ref.manifest_sha256
            if self.package_ref is not None
            else None,
            "release_record_path": str(self.release_record_path)
            if self.release_record_path is not None
            else None,
            "verdict": self.verdict,
            "reason_codes": list(self.reason_codes),
            "observed_scope": list(self.observed_scope),
            "unknowns": list(self.unknowns),
        }


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    ).encode("utf-8")
    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=str(path.parent),
    )
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(tmp_path, path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    return _sha256_bytes(encoded)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump(payload, allow_unicode=True, sort_keys=False)
    path.write_text(text, encoding="utf-8")


def init_preparation(
    request_path: str | Path,
    prep_dir: str | Path,
    *,
    preparation_id: str | None = None,
    request_id: str = "intake-request-01",
) -> dict[str, Any]:
    """Initialize a preparation directory by validating and copying `task.yaml`."""
    req = Path(request_path).resolve()
    task = load_intake_task_pack(req)

    prep = Path(prep_dir).resolve()
    prep.mkdir(parents=True, exist_ok=True)
    inputs_dir = prep / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    raw_bytes = req.read_bytes()
    request_sha = _sha256_bytes(raw_bytes)
    target_task = inputs_dir / "task.yaml"
    target_task.write_bytes(raw_bytes)

    prep_id = preparation_id or f"prep-{request_sha[:12]}"
    state = {
        "schema_version": 1,
        "preparation_id": prep_id,
        "request_id": request_id,
        "request_sha256": request_sha,
        "source_repo": task.source.repo,
        "source_commit": task.source.commit,
        "status": "initialized",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
    }
    _write_json_atomic(prep / STATE_FILENAME, state)
    return state


def analyze_intake(
    prep_dir: str | Path,
    *,
    probe_mode: ProbeMode = "none",
    repo_root: str | Path | None = None,
    timeout_sec: float = 60.0,
    allowed_ignored: tuple[str, ...] = (),
    handoff_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Run read-only source identity lock, AST discovery, asset triage, and optional probe."""
    prep = Path(prep_dir).resolve()
    state_path = prep / STATE_FILENAME
    if not state_path.is_file():
        raise FileNotFoundError(f"preparation not initialized: {state_path}")
    state = json.loads(state_path.read_text(encoding="utf-8"))

    task_path = prep / "inputs" / "task.yaml"
    task = load_intake_task_pack(task_path)
    resolved_repo = Path(repo_root or task.source.repo).resolve()

    blocked_reasons: list[str] = list(task.blocked_reasons)
    waiting_reasons: list[str] = []
    unknowns: list[str] = list(task.unknowns)
    observed_scope: list[str] = []

    def _add_unique(bucket: list[str], value: str) -> None:
        if value not in bucket:
            bucket.append(value)

    # 1. Pre-analysis source snapshot
    try:
        before_snap = snapshot_source(
            resolved_repo,
            task.source.commit,
            require_clean=True,
            allowed_ignored=allowed_ignored,
        )
    except SourceError as error:
        _add_unique(blocked_reasons, error.reason_code)
        before_snap = None

    # 2. Pinned-tree AST entrypoint & config discovery
    discovery_doc: dict[str, Any]
    if before_snap is not None:
        try:
            disc = discover_entrypoints(
                resolved_repo,
                task.source.commit,
                entry_hint=task.entrypoint.path,
            )
            discovery_doc = disc.to_dict()
            if disc.status == "resolved":
                _add_unique(observed_scope, "static")
            else:
                for reason in disc.blocked_reasons:
                    _add_unique(blocked_reasons, reason)
        except DiscoveryError as error:
            _add_unique(blocked_reasons, error.reason_code)
            discovery_doc = {
                "commit": task.source.commit,
                "tree_oid": before_snap.tree_oid,
                "status": "blocked",
                "selected": None,
                "candidates": [],
                "configs": [],
                "blocked_reasons": [error.reason_code],
                "error": error.to_dict(),
            }
    else:
        discovery_doc = {
            "commit": task.source.commit,
            "tree_oid": None,
            "status": "blocked",
            "selected": None,
            "candidates": [],
            "configs": [],
            "blocked_reasons": list(blocked_reasons),
        }

    # 3. Read-only asset triage (never downloads)
    required_asset_scope = "f1_startup_probe" if task.scope.startup_probe else None
    asset_report = triage_assets(task.assets, required_scope=required_asset_scope)
    for reason in asset_report.waiting_reasons:
        _add_unique(waiting_reasons, reason)
    for reason in asset_report.blocked_reasons:
        _add_unique(blocked_reasons, reason)
    for unk in asset_report.unknowns:
        _add_unique(unknowns, unk)

    # 4. Controlled startup probe (only if requested AND source clean AND assets ready)
    probe_result: ProbeResult
    if probe_mode == "startup":
        argv_ok, argv_reason, argv_detail = validate_probe_argv(
            task.entrypoint.argv,
            entrypoint_path=task.entrypoint.path,
            cwd=task.entrypoint.cwd,
        )
        if before_snap is None:
            probe_result = ProbeResult(
                status="blocked",
                argv=tuple(task.entrypoint.argv),
                cwd=task.entrypoint.cwd,
                sanitized_env={},
                exit_code=None,
                observed_scope=(),
                source_unchanged=False,
                before_identity_sha256=None,
                after_identity_sha256=None,
                entrypoint_path=task.entrypoint.path,
                reason_codes=("SOURCE_PRECHECK_FAILED",),
                detail="startup probe skipped because source precheck failed",
            )
        elif not argv_ok:
            _add_unique(blocked_reasons, argv_reason or "PROBE_ARGV_ENTRYPOINT_MISMATCH")
            probe_result = ProbeResult(
                status="blocked",
                argv=tuple(task.entrypoint.argv),
                cwd=task.entrypoint.cwd,
                sanitized_env={},
                exit_code=None,
                observed_scope=(),
                source_unchanged=True,
                before_identity_sha256=before_snap.identity_sha256,
                after_identity_sha256=before_snap.identity_sha256,
                entrypoint_path=task.entrypoint.path,
                reason_codes=(argv_reason or "PROBE_ARGV_ENTRYPOINT_MISMATCH",),
                detail=argv_detail,
            )
        elif asset_report.overall_status != "ready":
            probe_result = ProbeResult(
                status="not_run",
                argv=tuple(task.entrypoint.argv),
                cwd=task.entrypoint.cwd,
                sanitized_env={},
                exit_code=None,
                observed_scope=(),
                source_unchanged=True,
                before_identity_sha256=before_snap.identity_sha256,
                after_identity_sha256=before_snap.identity_sha256,
                entrypoint_path=task.entrypoint.path,
                reason_codes=tuple(
                    asset_report.blocked_reasons or asset_report.waiting_reasons
                ),
                detail="startup probe not run because required assets are not ready",
            )
        else:
            probe_out_dir = prep / "work" / "probe"
            probe_result = run_startup_probe(
                repo_root=resolved_repo,
                expected_commit=task.source.commit,
                argv=task.entrypoint.argv,
                cwd=task.entrypoint.cwd,
                env=task.entrypoint.env,
                output_dir=probe_out_dir,
                entrypoint_path=task.entrypoint.path,
                timeout_sec=timeout_sec,
            )
            if probe_result.status == "passed":
                _add_unique(observed_scope, "startup_probe")
            else:
                for reason in probe_result.reason_codes:
                    _add_unique(blocked_reasons, reason)
    else:
        probe_result = ProbeResult(
            status="not_run",
            argv=tuple(task.entrypoint.argv),
            cwd=task.entrypoint.cwd,
            sanitized_env={},
            exit_code=None,
            observed_scope=(),
            source_unchanged=True,
            before_identity_sha256=before_snap.identity_sha256 if before_snap else None,
            after_identity_sha256=before_snap.identity_sha256 if before_snap else None,
            entrypoint_path=task.entrypoint.path,
            reason_codes=(),
            detail="startup probe was not requested in analyze step",
        )

    # 5. Post-analysis source snapshot & immutability assertion
    after_snap = None
    if before_snap is not None:
        try:
            after_snap = snapshot_source(
                resolved_repo,
                task.source.commit,
                require_clean=False,
                allowed_ignored=allowed_ignored,
            )
            verify_source_unchanged(before_snap, after_snap)
        except SourceError as error:
            _add_unique(blocked_reasons, error.reason_code)

    # Persist analysis workspace artifacts under prep/work/analysis
    work_dir = prep / "work"
    work_dir.mkdir(parents=True, exist_ok=True)
    # Evidence belongs to this analysis run, not to the preparation directory's
    # history. Do this even when the replacement handoff fails validation.
    evidence_dir = work_dir / "evidence"
    if evidence_dir.exists():
        shutil.rmtree(evidence_dir)
    if isinstance(task, IntakeTaskPackV2):
        try:
            if handoff_dir is None:
                raise ValueError("v2 intake requires --handoff-dir; no Skill harness is executed")
            if before_snap is None:
                raise ValueError("v2 handoff requires a verified pinned source tree")
            import_handoff(task, Path(handoff_dir).resolve(), work_dir, repo=resolved_repo)
        except Exception as error:
            _add_unique(blocked_reasons, "V2_HANDOFF_INVALID")
            unknowns.append(f"handoff_only import failed: {error}")

    # 6. Create offline recursive git bundles for root + submodules inside work/source_bundles
    bundles_manifest: dict[str, Any] | None = None
    if before_snap is not None and after_snap is not None and not blocked_reasons:
        bundles_dir = work_dir / "source_bundles"
        if bundles_dir.exists():
            shutil.rmtree(bundles_dir)
        try:
            bundles_manifest = create_recursive_source_bundles(
                resolved_repo,
                before_snap,
                bundles_dir,
                bundle_rel_prefix="evidence/source_bundles",
            )
        except SourceError as error:
            _add_unique(blocked_reasons, error.reason_code)

    fingerprint_doc = {
        "schema_version": 1,
        "before": before_snap.to_dict() if before_snap is not None else None,
        "after": after_snap.to_dict() if after_snap is not None else None,
        "unchanged": (
            before_snap is not None
            and after_snap is not None
            and before_snap.identity_sha256 == after_snap.identity_sha256
        ),
        "bundles": bundles_manifest,
    }
    _write_json_atomic(work_dir / "source_fingerprint.json", fingerprint_doc)

    reference_manifest = {
        "schema_version": 1,
        "repo": task.source.repo,
        "repo_root": str(resolved_repo),
        "model_variant": task.source.model_variant,
        "head_commit": before_snap.head_commit if before_snap else None,
        "tree_oid": before_snap.tree_oid if before_snap else None,
        "object_format": before_snap.object_format if before_snap else None,
        "submodules": [s.to_dict() for s in before_snap.submodules]
        if before_snap
        else [],
        "target_framework": task.target.framework,
        "target_commit": task.target.commit,
    }
    _write_yaml(work_dir / "reference_manifest.yaml", reference_manifest)
    _write_yaml(work_dir / "reference_entrypoints.yaml", discovery_doc)

    launch_contract = {
        "schema_version": 1,
        "environment": task.environment.model_dump(mode="json"),
        "entrypoint": task.entrypoint.model_dump(mode="json"),
        "requested_scope": task.scope.model_dump(mode="json"),
        "probe": probe_result.to_dict(),
    }
    _write_yaml(work_dir / "launch_contract.yaml", launch_contract)
    _write_yaml(work_dir / "resource_manifest.yaml", asset_report.to_dict())

    report_lines = [
        "# Stage 1 (`intake`) Analysis Report",
        "",
        f"- Preparation ID: `{state['preparation_id']}`",
        f"- Model Variant: `{task.source.model_variant}`",
        f"- Source Commit: `{task.source.commit}`",
        f"- Asset Status: `{asset_report.overall_status}`",
        f"- Probe Status: `{probe_result.status}`",
        f"- Observed Scope: `{observed_scope}`",
        f"- Waiting Reasons: `{waiting_reasons}`",
        f"- Blocked Reasons: `{blocked_reasons}`",
    ]
    (work_dir / "analysis_report.md").write_text(
        "\n".join(report_lines) + "\n",
        encoding="utf-8",
    )

    analysis_summary = {
        **state,
        # Preparation-only control state: freeze must recheck this same checkout,
        # not resolve a remote URL or a different task.source.repo as a local path.
        "resolved_repo": str(resolved_repo),
        "status": "analyzed",
        "probe_mode": probe_mode,
        "probe_status": probe_result.status,
        "asset_status": asset_report.overall_status,
        "observed_scope": observed_scope,
        "waiting_reasons": waiting_reasons,
        "blocked_reasons": blocked_reasons,
        "unknowns": unknowns,
        "updated_at": _utc_now(),
    }
    _write_json_atomic(state_path, analysis_summary)
    return analysis_summary


def freeze_intake(
    prep_dir: str | Path,
    *,
    release_name: str | None = None,
    attempt_id: str = "intake-attempt-01",
) -> FreezeResult:
    """Stage, independently validate, and atomically publish an immutable `intake` package.

    Only publishes when `IntakeChecker` passes cleanly. Never treats an unsealed
    `.staging` directory as a successful handoff.
    """
    prep = Path(prep_dir).resolve()
    state_path = prep / STATE_FILENAME
    if not state_path.is_file():
        return FreezeResult(
            status="error",
            summary=f"preparation state not found: {state_path}",
            reason_codes=("PREP_NOT_INITIALIZED",),
        )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") not in {"analyzed", "frozen"}:
        return FreezeResult(
            status="blocked",
            summary="cannot freeze before `analyze` completes",
            reason_codes=("PREP_NOT_ANALYZED",),
        )

    task_path = prep / "inputs" / "task.yaml"
    try:
        task = load_intake_task_pack(task_path)
    except (ContractParseError, ContractValidationError) as error:
        return FreezeResult(
            status="error",
            summary=f"task contract invalid: {error}",
            reason_codes=(error.reason_code,),
        )

    waiting_reasons = tuple(state.get("waiting_reasons", ()))
    blocked_reasons = tuple(state.get("blocked_reasons", ()))
    observed_scope = tuple(state.get("observed_scope", ()))
    unknowns = tuple(state.get("unknowns", ()))

    # Fast-path waiting vs blocked classification before staging.
    if blocked_reasons:
        return FreezeResult(
            status="blocked",
            summary="intake blocked prior to sealing: " + ", ".join(blocked_reasons),
            reason_codes=blocked_reasons,
            observed_scope=observed_scope,
            unknowns=unknowns,
        )
    if waiting_reasons:
        return FreezeResult(
            status="waiting",
            summary="intake waiting on declared assets: " + ", ".join(waiting_reasons),
            reason_codes=waiting_reasons,
            observed_scope=observed_scope,
            unknowns=unknowns,
        )

    resolved_repo = Path(state.get("resolved_repo") or task.source.repo).resolve()
    if not resolved_repo.is_dir():
        return FreezeResult(
            status="blocked",
            summary=f"analyzed source checkout is unavailable: {resolved_repo}; rerun analyze with --repo-root",
            reason_codes=("SOURCE_NOT_DIRECTORY",),
            observed_scope=observed_scope,
            unknowns=unknowns,
        )

    work_dir = prep / "work"
    releases_dir = prep / "packages"
    releases_dir.mkdir(parents=True, exist_ok=True)
    rel_name = release_name or f"intake-{state['preparation_id']}"
    staging_dir = releases_dir / f"{rel_name}.staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    try:
        (staging_dir / "inputs").mkdir(parents=True)
        (staging_dir / "analysis").mkdir(parents=True)
        (staging_dir / "handoff").mkdir(parents=True)
        (staging_dir / "evidence" / "validate").mkdir(parents=True)

        shutil.copy2(task_path, staging_dir / "inputs" / "task.yaml")
        shutil.copy2(
            work_dir / "source_fingerprint.json",
            staging_dir / "inputs" / "source_fingerprint.json",
        )
        for name in (
            "reference_manifest.yaml",
            "reference_entrypoints.yaml",
            "launch_contract.yaml",
            "resource_manifest.yaml",
            "analysis_report.md",
        ):
            shutil.copy2(work_dir / name, staging_dir / "analysis" / name)
        if isinstance(task, IntakeTaskPackV2):
            for name in V2_ANALYSIS_FILES:
                shutil.copy2(work_dir / name, staging_dir / "analysis" / name)
            shutil.copytree(work_dir / "evidence", staging_dir / "evidence", dirs_exist_ok=True)

        # Copy offline source bundles (root + recursive submodules) into the sealed package.
        work_bundles_dir = work_dir / "source_bundles"
        if work_bundles_dir.is_dir():
            shutil.copytree(
                work_bundles_dir,
                staging_dir / "evidence" / "source_bundles",
            )

        # Copy structured probe marker into sealed package if startup_probe passed.
        probe_marker_src = work_dir / "probe" / "probe_marker.json"
        if probe_marker_src.is_file():
            (staging_dir / "evidence" / "probe").mkdir(parents=True, exist_ok=True)
            shutil.copy2(
                probe_marker_src,
                staging_dir / "evidence" / "probe" / "probe_marker.json",
            )

        from validate.intake_check import IntakeChecker

        checker = IntakeChecker()
        request_sha = str(state["request_sha256"])
        # Execute real pre-publish validation via `IntakeChecker` (produces structured
        # check details and deterministic `checks_sha256` bound to actual staged artifacts).
        pre_verdict_rel = "evidence/validate/pre_publish_verdict.json"
        pre_verdict_path = staging_dir / pre_verdict_rel
        pre_verdict_doc, pre_gate = checker.build_pre_publish_verdict(
            staging_dir=staging_dir,
            expected_commit=task.source.commit,
            request_sha256=request_sha,
            reference_dir=resolved_repo,
        )
        if pre_gate.status != "pass" or pre_verdict_doc is None:
            shutil.rmtree(staging_dir, ignore_errors=True)
            return FreezeResult(
                status="blocked" if pre_gate.status == "blocked" else "error",
                summary=pre_gate.summary,
                verdict=pre_gate.to_dict(),
                reason_codes=pre_gate.reason_codes,
                observed_scope=observed_scope,
                unknowns=unknowns,
            )
        pre_verdict_sha = _write_json_atomic(pre_verdict_path, pre_verdict_doc)

        handoff_scope = list(observed_scope) or ["static"]
        handoff = HandoffResult(
            schema_version=1,
            preparation_id=str(state["preparation_id"]),
            request_id=str(state["request_id"]),
            stage="f1",
            attempt_id=attempt_id,
            request_sha256=request_sha,
            execution_status="completed",
            observed_scope=handoff_scope,  # type: ignore[arg-type]
            validator_verdict={
                "path": pre_verdict_rel,
                "sha256": pre_verdict_sha,
            },
            unknowns=list(unknowns),
            blocked_reasons=[],
        )
        _write_json_atomic(
            staging_dir / "handoff" / "result.json",
            handoff.model_dump(mode="json"),
        )

        # Full pre-publish validation on staging directory (all required files + handoff).
        full_pre_verdict = checker.validate_directory(
            package_dir=staging_dir,
            expected_commit=task.source.commit,
            manifest_sha256=request_sha,
            reference_dir=resolved_repo,
            require_sealed=False,
        )
        if full_pre_verdict.status != "pass":
            shutil.rmtree(staging_dir, ignore_errors=True)
            return FreezeResult(
                status="blocked" if full_pre_verdict.status == "blocked" else "error",
                summary=full_pre_verdict.summary,
                verdict=full_pre_verdict.to_dict(),
                reason_codes=full_pre_verdict.reason_codes,
                observed_scope=observed_scope,
                unknowns=unknowns,
            )

        # Atomically seal and publish the package via core/artifacts.py:publish().
        package_ref = publish(staging_dir)

        # Independent post-publish verification against the real sealed manifest SHA-256.
        sealed_verdict = checker.validate_directory(
            package_dir=package_ref.package_dir,
            expected_commit=task.source.commit,
            manifest_sha256=package_ref.manifest_sha256,
            reference_dir=resolved_repo,
            require_sealed=True,
        )
        if sealed_verdict.status != "pass":
            return FreezeResult(
                status="error",
                summary=f"post-publish verification failed: {sealed_verdict.summary}",
                verdict=sealed_verdict.to_dict(),
                reason_codes=sealed_verdict.reason_codes,
                observed_scope=observed_scope,
                unknowns=unknowns,
            )

        # Persist out-of-package trusted release record (the external root of trust).
        release_record_path = prep / "release.json"
        release_record = {
            "schema_version": 1,
            "stage": "intake",
            "preparation_id": state["preparation_id"],
            "request_id": state["request_id"],
            "request_sha256": state["request_sha256"],
            "source_commit": task.source.commit,
            "package_dir": str(package_ref.package_dir),
            "manifest_sha256": package_ref.manifest_sha256,
            "observed_scope": list(observed_scope),
            "validator_verdict": sealed_verdict.to_dict(),
            "published_at": _utc_now(),
        }
        _write_json_atomic(release_record_path, release_record)

        state["status"] = "frozen"
        state["package_dir"] = str(package_ref.package_dir)
        state["manifest_sha256"] = package_ref.manifest_sha256
        state["updated_at"] = _utc_now()
        _write_json_atomic(state_path, state)

        return FreezeResult(
            status="completed",
            summary="intake package validated and atomically sealed",
            package_ref=package_ref,
            release_record_path=release_record_path,
            verdict=sealed_verdict.to_dict(),
            reason_codes=(),
            observed_scope=observed_scope,
            unknowns=unknowns,
        )
    except Exception as error:
        shutil.rmtree(staging_dir, ignore_errors=True)
        return FreezeResult(
            status="error",
            summary=f"freeze failed: {type(error).__name__}: {error}",
            reason_codes=("FREEZE_EXCEPTION",),
            observed_scope=observed_scope,
            unknowns=unknowns,
        )

"""Independent checker and CLI for Stage 1 (`intake`) sealed packages.

Verifies:
- Package is sealed (rejects `.staging` / unmanifested directories) and passes
  `core.artifacts.verify` against the external trusted `manifest_sha256`;
- Required intake layout exists in the manifest;
- Source commit, tree OID, submodules, clean worktree, and pre/post immutability;
- Selected entrypoint belongs to the pinned Git tree and matches worktree bytes;
- Asset triage status (missing declared asset -> waiting/blocked, digest mismatch -> blocked);
- Probe evidence satisfies declared task scope (`--help` / `not_run` cannot pass `startup_probe`);
- `handoff/result.json` validates against strict `HandoffResult` and binds request SHA.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import tempfile
from pathlib import Path
from typing import Any

from core import artifacts as _artifacts_module
from core.artifacts import STAGING_SUFFIX, ArtifactError, PackageRef, verify as verify_package
from core.contracts import (
    ContractParseError,
    ContractValidationError,
    common as _contracts_common,
    errors as _contracts_errors,
    f1 as _contracts_f1,
    handoff as _contracts_handoff,
    load_handoff_result,
    load_intake_task_pack,
)
from core.contracts.common import load_yaml_document
from core.contracts import intake_v2 as _contracts_intake_v2
from core.contracts import anatomy_spec as _contracts_anatomy_spec
from core.contracts.intake_v2 import IntakeTaskPackV2
from core.orchestrator.judger import GateBinding, GateEvidence, GateOutcome, GateResult
from core.stages.f1_intake import assets as _assets_module
from core.stages.f1_intake import discovery as _discovery_module
from core.stages.f1_intake import probe as _probe_module
from core.stages.f1_intake import source as _source_module
from core.stages.f1_intake import handoff as _handoff_module
from core.stages.f1_intake.handoff import V2_PACKAGE_FILES, recheck_handoff
from core.stages.f2_anatomy import contract as _anatomy_contract_module
from core.stages.f1_intake.assets import triage_assets
from core.stages.f1_intake.discovery import DiscoveryError, discover_entrypoints
from core.stages.f1_intake.probe import compute_argv_sha256, validate_probe_argv
from core.stages.f1_intake.source import (
    SourceError,
    SourceSnapshot,
    restore_from_bundle_manifest,
    snapshot_source,
    verify_source_unchanged,
)

from .base import CheckContext, validator_source_sha256

REQUIRED_INTAKE_PACKAGE_FILES: tuple[str, ...] = (
    "inputs/task.yaml",
    "inputs/source_fingerprint.json",
    "analysis/reference_manifest.yaml",
    "analysis/reference_entrypoints.yaml",
    "analysis/launch_contract.yaml",
    "analysis/resource_manifest.yaml",
    "analysis/analysis_report.md",
    "handoff/result.json",
)
PRE_PUBLISH_REQUIRED_FILES: tuple[str, ...] = tuple(
    rel for rel in REQUIRED_INTAKE_PACKAGE_FILES if rel != "handoff/result.json"
)


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _intake_dependency_sources() -> tuple[Path, ...]:
    sources: list[Path] = []
    for mod in (
        _artifacts_module,
        _contracts_common,
        _contracts_errors,
        _contracts_f1,
        _contracts_intake_v2,
        _contracts_anatomy_spec,
        _contracts_handoff,
        _source_module,
        _discovery_module,
        _assets_module,
        _probe_module,
        _handoff_module,
        _anatomy_contract_module,
    ):
        src = inspect.getsourcefile(mod)
        if src is not None:
            sources.append(Path(src).resolve())
    return tuple(sources)


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class IntakeChecker:
    """Independent validator for Stage 1 (`intake`) packages."""

    gate_id = "intake"
    check_name = "intake"
    validator_id = "validate.intake_check"
    validator_version = "2"
    historical_validator_id = "stages.verify.intake_check"

    def validator_sha256(self) -> str:
        return validator_source_sha256(
            type(self),
            extra_sources=_intake_dependency_sources(),
        )

    def build_pre_publish_verdict(
        self,
        *,
        staging_dir: Path,
        expected_commit: str,
        request_sha256: str,
        reference_dir: Path | None = None,
    ) -> tuple[dict[str, Any] | None, GateResult]:
        """Execute real pre-publish validation on `staging_dir` and return bound verdict payload."""
        gate = self.validate_directory(
            package_dir=staging_dir,
            expected_commit=expected_commit,
            manifest_sha256=request_sha256,
            reference_dir=reference_dir,
            require_sealed=False,
            require_handoff=False,
        )
        if gate.status != "pass":
            return None, gate
        checks = gate.metrics.get("checks")
        if not isinstance(checks, dict):
            return None, self._fail(
                commit=expected_commit,
                manifest_sha256=request_sha256,
                status="error",
                reason_code="PRE_PUBLISH_CHECKS_MISSING",
                first_mismatch="pre_publish:checks",
                summary="pre-publish validation did not produce structured checks map",
                evidence_path="evidence/validate/pre_publish_verdict.json",
                expected="dict of checks",
                actual=type(checks).__name__,
            )
        payload = {
            "schema_version": 1,
            "phase": "pre_publish",
            "status": "pass",
            "request_sha256": request_sha256,
            "source_commit": expected_commit,
            "validator_id": self.validator_id,
            "validator_version": self.validator_version,
            "validator_sha256": self.validator_sha256(),
            "checks": checks,
            "checks_sha256": _canonical_json_sha256(checks),
        }
        return payload, gate

    def _fail(
        self,
        *,
        commit: str,
        manifest_sha256: str,
        status: GateOutcome,
        reason_code: str,
        first_mismatch: str,
        summary: str,
        evidence_path: str,
        expected: Any,
        actual: Any,
    ) -> GateResult:
        return GateResult(
            gate_id=self.gate_id,
            check="intake",
            status=status,
            binding=GateBinding(
                commit=commit,
                manifest_sha256=manifest_sha256,
                validator_id=self.validator_id,
                validator_sha256=self.validator_sha256(),
            ),
            reason_codes=(reason_code,),
            first_mismatch=first_mismatch,
            summary=summary,
            evidence=(
                GateEvidence(
                    key=reason_code,
                    detail=summary,
                    path=evidence_path,
                ),
            ),
            metrics={
                "reason_code": reason_code,
                "first_mismatch": first_mismatch,
                "expected": expected,
                "actual": actual,
                "evidence_path": evidence_path,
            },
        )

    def validate_directory(
        self,
        *,
        package_dir: Path,
        expected_commit: str,
        manifest_sha256: str,
        reference_dir: Path | None = None,
        require_sealed: bool = True,
        require_handoff: bool = True,
    ) -> GateResult:
        """Validate either a sealed package (`require_sealed=True`) or pre-publish staging dir."""
        pkg_dir = Path(package_dir)

        if require_sealed:
            if pkg_dir.name.endswith(STAGING_SUFFIX):
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="UNSEALED_PACKAGE_REJECTED",
                    first_mismatch="package_dir:staging_suffix",
                    summary="unsealed .staging directory cannot be accepted as a released intake package",
                    evidence_path=pkg_dir.name,
                    expected="sealed release directory",
                    actual=pkg_dir.name,
                )
            try:
                verified = verify_package(
                    PackageRef(package_dir=pkg_dir, manifest_sha256=manifest_sha256)
                )
            except ArtifactError as error:
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="error",
                    reason_code=error.reason_code,
                    first_mismatch=f"artifact:{error.reason_code}",
                    summary=error.message,
                    evidence_path=error.path or "artifact_manifest.json",
                    expected=manifest_sha256,
                    actual=error.reason_code,
                )
            present_paths = verified.manifest_paths
        else:
            present_paths = {
                p.relative_to(pkg_dir).as_posix()
                for p in pkg_dir.rglob("*")
                if p.is_file()
            }

        # 1. Required files present
        required_layout = (
            REQUIRED_INTAKE_PACKAGE_FILES
            if (require_sealed or require_handoff)
            else PRE_PUBLISH_REQUIRED_FILES
        )
        missing_files = [rel for rel in required_layout if rel not in present_paths]
        if missing_files:
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code="MISSING_PACKAGE_FILE",
                first_mismatch=f"layout:{missing_files[0]}",
                summary=f"required intake file missing: {missing_files[0]}",
                evidence_path=missing_files[0],
                expected=list(REQUIRED_INTAKE_PACKAGE_FILES),
                actual=sorted(present_paths),
            )

        # 2. Load and validate inputs/task.yaml
        task_path = pkg_dir / "inputs/task.yaml"
        try:
            task = load_intake_task_pack(task_path)
        except (ContractParseError, ContractValidationError) as error:
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code=error.reason_code,
                first_mismatch="inputs/task.yaml:contract_invalid",
                summary=f"task contract invalid: {error}",
                evidence_path="inputs/task.yaml",
                expected="valid IntakeTaskPack",
                actual=error.to_dict(),
            )
        request_sha = _sha256_bytes(task_path.read_bytes())
        v2_checks: dict[str, str] = {}
        if isinstance(task, IntakeTaskPackV2):
            missing = [x for x in V2_PACKAGE_FILES if x not in present_paths]
            if missing:
                return self._fail(
                    commit=expected_commit, manifest_sha256=manifest_sha256, status="blocked",
                    reason_code="V2_HANDOFF_MISSING", first_mismatch=missing[0],
                    summary="v2 package lacks required handoff artifact", evidence_path=missing[0],
                    expected=list(V2_PACKAGE_FILES), actual=sorted(present_paths),
                )

        # 3. Source fingerprint & pre/post immutability
        fp_path = pkg_dir / "inputs/source_fingerprint.json"
        try:
            fp_doc = json.loads(fp_path.read_text(encoding="utf-8"))
            before_snap = SourceSnapshot.from_dict(fp_doc["before"])
            after_snap = SourceSnapshot.from_dict(fp_doc["after"])
        except Exception as error:
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="error",
                reason_code="INVALID_SOURCE_FINGERPRINT",
                first_mismatch="inputs/source_fingerprint.json:parse",
                summary=f"invalid source_fingerprint.json: {error}",
                evidence_path="inputs/source_fingerprint.json",
                expected="before/after SourceSnapshot",
                actual=str(error),
            )

        if before_snap.head_commit != expected_commit:
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code="SOURCE_COMMIT_MISMATCH",
                first_mismatch="source.head_commit",
                summary=(
                    f"source commit {before_snap.head_commit} != expected {expected_commit}"
                ),
                evidence_path="inputs/source_fingerprint.json",
                expected=expected_commit,
                actual=before_snap.head_commit,
            )

        if task.source.commit != expected_commit:
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code="TASK_COMMIT_MISMATCH",
                first_mismatch="inputs/task.yaml:source.commit",
                summary=(
                    f"task.source.commit {task.source.commit} != expected {expected_commit}"
                ),
                evidence_path="inputs/task.yaml",
                expected=expected_commit,
                actual=task.source.commit,
            )

        if not before_snap.is_clean or not after_snap.is_clean:
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code="SOURCE_WORKTREE_DIRTY",
                first_mismatch="source.worktree_status",
                summary="recorded upstream source snapshot is not clean",
                evidence_path="inputs/source_fingerprint.json",
                expected="clean worktree and matching submodules",
                actual=before_snap.worktree_status.to_dict(),
            )

        try:
            verify_source_unchanged(before_snap, after_snap)
        except SourceError as error:
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code=error.reason_code,
                first_mismatch="source.immutability",
                summary=error.message,
                evidence_path="inputs/source_fingerprint.json",
                expected=before_snap.identity_sha256,
                actual=after_snap.identity_sha256,
            )

        # 3b. Verify packaged offline recursive git bundles (root + submodules) can restore cleanly
        # and yield the exact same host-independent `identity_sha256` on a fresh path.
        bundles_manifest = fp_doc.get("bundles")
        if not isinstance(bundles_manifest, dict) or not isinstance(
            bundles_manifest.get("root"), dict
        ):
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code="SOURCE_BUNDLE_MISSING",
                first_mismatch="inputs/source_fingerprint.json:bundles",
                summary="source_fingerprint.json lacks offline source bundle manifest",
                evidence_path="inputs/source_fingerprint.json",
                expected="root + submodule bundle metadata",
                actual=bundles_manifest,
            )

        root_bundle_rel = bundles_manifest["root"].get("rel_path")
        if not isinstance(root_bundle_rel, str) or root_bundle_rel not in present_paths:
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code="SOURCE_BUNDLE_MISSING",
                first_mismatch="inputs/source_fingerprint.json:bundles.root.rel_path",
                summary="root source bundle is not present inside the package",
                evidence_path=str(root_bundle_rel or "inputs/source_fingerprint.json"),
                expected="manifested root.bundle path",
                actual=root_bundle_rel,
            )

        sub_bundle_entries = bundles_manifest.get("submodules", [])
        expected_sub_paths = [sub.path for sub in before_snap.submodules]
        actual_sub_paths = [
            entry.get("submodule_path")
            for entry in sub_bundle_entries
            if isinstance(entry, dict)
        ]
        if sorted(expected_sub_paths) != sorted(p for p in actual_sub_paths if isinstance(p, str)):
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code="SUBMODULE_BUNDLE_MISSING",
                first_mismatch="inputs/source_fingerprint.json:bundles.submodules",
                summary="submodule bundles do not cover all recursive submodules",
                evidence_path="inputs/source_fingerprint.json",
                expected=sorted(expected_sub_paths),
                actual=actual_sub_paths,
            )
        for sub_entry in sub_bundle_entries:
            sub_rel = sub_entry.get("rel_path")
            if not isinstance(sub_rel, str) or sub_rel not in present_paths:
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="SUBMODULE_BUNDLE_MISSING",
                    first_mismatch="inputs/source_fingerprint.json:bundles.submodules",
                    summary=f"submodule bundle {sub_rel!r} is missing from package",
                    evidence_path=str(sub_rel or "inputs/source_fingerprint.json"),
                    expected="manifested submodule bundle path",
                    actual=sub_rel,
                )

        with tempfile.TemporaryDirectory(prefix="intake-check-restore-") as restore_tmp:
            restored_dir = Path(restore_tmp) / "restored_repo"
            try:
                restored_snap = restore_from_bundle_manifest(
                    pkg_dir,
                    bundles_manifest,
                    restored_dir,
                )
            except SourceError as error:
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="error",
                    reason_code=error.reason_code,
                    first_mismatch="source.bundle_restore",
                    summary=f"offline bundle restore failed: {error.message}",
                    evidence_path=str(root_bundle_rel),
                    expected=expected_commit,
                    actual=error.actual,
                )

            # Offline-restored repo must match recorded tree/commit/submodules/worktree identity.
            if (
                restored_snap.head_commit != before_snap.head_commit
                or restored_snap.tree_oid != before_snap.tree_oid
                or (
                    not before_snap.allowed_ignored
                    and restored_snap.identity_sha256 != before_snap.identity_sha256
                )
            ):
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="error",
                    reason_code="BUNDLE_IDENTITY_MISMATCH",
                    first_mismatch="inputs/source_fingerprint.json:identity_sha256",
                    summary="offline-restored bundle identity does not match recorded source_fingerprint",
                    evidence_path="inputs/source_fingerprint.json",
                    expected=before_snap.identity_sha256,
                    actual=restored_snap.identity_sha256,
                )

            # Perform independent pinned-tree entrypoint & config verification on the restored tree
            # even when the original host repo path is no longer present (true offline verification).
            recheck_repo = (
                reference_dir
                if reference_dir is not None and reference_dir.is_dir()
                else restored_dir
            )
            if isinstance(task, IntakeTaskPackV2):
                try:
                    v2_checks = recheck_handoff(task, pkg_dir, restored_dir)
                except Exception as error:
                    return self._fail(
                        commit=expected_commit, manifest_sha256=manifest_sha256, status="blocked",
                        reason_code="V2_HANDOFF_INVALID", first_mismatch="analysis/model_analysis.json",
                        summary=str(error), evidence_path="analysis/model_analysis.json",
                        expected="independently reproducible v2 handoff", actual=str(error),
                    )

            if reference_dir is not None and reference_dir.is_dir():
                try:
                    live_snap = snapshot_source(
                        reference_dir,
                        expected_commit,
                        require_clean=True,
                        allowed_ignored=before_snap.allowed_ignored,
                    )
                    verify_source_unchanged(before_snap, live_snap)
                except SourceError as error:
                    return self._fail(
                        commit=expected_commit,
                        manifest_sha256=manifest_sha256,
                        status="blocked",
                        reason_code=error.reason_code,
                        first_mismatch="source.live_recheck",
                        summary=error.message,
                        evidence_path=error.evidence_path or "inputs/source_fingerprint.json",
                        expected=error.expected,
                        actual=error.actual,
                    )

            # 4. Reference manifest & entrypoints verification
            try:
                ref_manifest = load_yaml_document(pkg_dir / "analysis/reference_manifest.yaml")
                entrypoints_doc = load_yaml_document(
                    pkg_dir / "analysis/reference_entrypoints.yaml"
                )
                launch_doc = load_yaml_document(pkg_dir / "analysis/launch_contract.yaml")
                resource_doc = load_yaml_document(pkg_dir / "analysis/resource_manifest.yaml")
            except ContractParseError as error:
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="error",
                    reason_code=error.reason_code,
                    first_mismatch=f"yaml:{error.path or 'analysis'}",
                    summary=error.message,
                    evidence_path=error.path or "analysis/reference_manifest.yaml",
                    expected="valid YAML mapping",
                    actual=error.reason_code,
                )

            if (
                ref_manifest.get("head_commit") != expected_commit
                or ref_manifest.get("tree_oid") != before_snap.tree_oid
                or ref_manifest.get("target_commit") != task.target.commit
                or ref_manifest.get("model_variant") != task.source.model_variant
            ):
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="REFERENCE_MANIFEST_MISMATCH",
                    first_mismatch="analysis/reference_manifest.yaml:tree_oid",
                    summary="reference_manifest does not match source snapshot and task contract",
                    evidence_path="analysis/reference_manifest.yaml",
                    expected={
                        "head_commit": expected_commit,
                        "tree_oid": before_snap.tree_oid,
                        "target_commit": task.target.commit,
                        "model_variant": task.source.model_variant,
                    },
                    actual={
                        "head_commit": ref_manifest.get("head_commit"),
                        "tree_oid": ref_manifest.get("tree_oid"),
                        "target_commit": ref_manifest.get("target_commit"),
                        "model_variant": ref_manifest.get("model_variant"),
                    },
                )

            selected_entry = entrypoints_doc.get("selected")
            if entrypoints_doc.get("status") != "resolved" or not isinstance(
                selected_entry, dict
            ):
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="ENTRYPOINT_NOT_RESOLVED",
                    first_mismatch="analysis/reference_entrypoints.yaml:selected",
                    summary="entrypoint discovery is not resolved",
                    evidence_path="analysis/reference_entrypoints.yaml",
                    expected="resolved",
                    actual=entrypoints_doc.get("status"),
                )

            if selected_entry.get("path") != task.entrypoint.path:
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="ENTRYPOINT_PATH_MISMATCH",
                    first_mismatch="analysis/reference_entrypoints.yaml:selected.path",
                    summary="discovered entrypoint path differs from task.entrypoint.path",
                    evidence_path="analysis/reference_entrypoints.yaml",
                    expected=task.entrypoint.path,
                    actual=selected_entry.get("path"),
                )

            try:
                rechecked_disc = discover_entrypoints(
                    recheck_repo,
                    expected_commit,
                    entry_hint=task.entrypoint.path,
                )
            except DiscoveryError as error:
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code=error.reason_code,
                    first_mismatch="discovery.live_recheck",
                    summary=error.message,
                    evidence_path=error.evidence_path
                    or "analysis/reference_entrypoints.yaml",
                    expected=error.expected,
                    actual=error.actual,
                )
            if (
                rechecked_disc.selected is None
                or rechecked_disc.selected.blob_oid != selected_entry.get("blob_oid")
                or rechecked_disc.selected.blob_sha256 != selected_entry.get("blob_sha256")
            ):
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="ENTRYPOINT_BLOB_MISMATCH",
                    first_mismatch="analysis/reference_entrypoints.yaml:selected.blob_oid",
                    summary="selected entrypoint blob differs from pinned commit tree",
                    evidence_path="analysis/reference_entrypoints.yaml",
                    expected=(
                        rechecked_disc.selected.blob_oid
                        if rechecked_disc.selected
                        else None
                    ),
                    actual=selected_entry.get("blob_oid"),
                )

            # Cross-document check: discovered config set must match pinned tree configs.
            expected_configs = [cfg.to_dict() for cfg in rechecked_disc.configs]
            actual_configs = entrypoints_doc.get("configs", [])
            if actual_configs != expected_configs:
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="DISCOVERY_CONFIGS_MISMATCH",
                    first_mismatch="analysis/reference_entrypoints.yaml:configs",
                    summary="recorded config file list differs from pinned commit tree",
                    evidence_path="analysis/reference_entrypoints.yaml",
                    expected=expected_configs,
                    actual=actual_configs,
                )

        # 5. Asset status & per-record cross-document reconciliation
        required_asset_scope = (
            "f1_startup_probe" if task.scope.startup_probe else None
        )
        rechecked_assets = triage_assets(
            task.assets,
            required_scope=required_asset_scope,
        )
        if rechecked_assets.overall_status != "ready":
            reason = (
                rechecked_assets.blocked_reasons[0]
                if rechecked_assets.blocked_reasons
                else (
                    rechecked_assets.waiting_reasons[0]
                    if rechecked_assets.waiting_reasons
                    else "ASSET_NOT_READY"
                )
            )
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code=reason,
                first_mismatch="analysis/resource_manifest.yaml:overall_status",
                summary=f"required assets are not ready (status={rechecked_assets.overall_status})",
                evidence_path="analysis/resource_manifest.yaml",
                expected="ready",
                actual=rechecked_assets.overall_status,
            )

        if resource_doc.get("overall_status") != "ready":
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code="ASSET_STATUS_NOT_READY",
                first_mismatch="analysis/resource_manifest.yaml:overall_status",
                summary="recorded resource_manifest overall_status is not ready",
                evidence_path="analysis/resource_manifest.yaml",
                expected="ready",
                actual=resource_doc.get("overall_status"),
            )

        recorded_asset_records = {
            rec["id"]: rec
            for rec in resource_doc.get("records", [])
            if isinstance(rec, dict) and "id" in rec
        }
        expected_asset_records = {rec.id: rec for rec in rechecked_assets.records}
        if set(recorded_asset_records.keys()) != set(expected_asset_records.keys()):
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code="RESOURCE_MANIFEST_MISMATCH",
                first_mismatch="analysis/resource_manifest.yaml:records",
                summary="resource_manifest asset IDs differ from task.yaml",
                evidence_path="analysis/resource_manifest.yaml",
                expected=sorted(expected_asset_records.keys()),
                actual=sorted(recorded_asset_records.keys()),
            )
        for asset_id, exp_rec in sorted(expected_asset_records.items()):
            rec_doc = recorded_asset_records[asset_id]
            if (
                rec_doc.get("observed_sha256") != exp_rec.observed_sha256
                or rec_doc.get("expected_sha256") != exp_rec.expected_sha256
                or rec_doc.get("status") != exp_rec.status
            ):
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="RESOURCE_DIGEST_RECONCILIATION_MISMATCH",
                    first_mismatch=f"analysis/resource_manifest.yaml:records[{asset_id}]",
                    summary=f"resource_manifest record for asset {asset_id!r} differs from live verification",
                    evidence_path="analysis/resource_manifest.yaml",
                    expected={
                        "expected_sha256": exp_rec.expected_sha256,
                        "observed_sha256": exp_rec.observed_sha256,
                        "status": exp_rec.status,
                    },
                    actual={
                        "expected_sha256": rec_doc.get("expected_sha256"),
                        "observed_sha256": rec_doc.get("observed_sha256"),
                        "status": rec_doc.get("status"),
                    },
                )

        # 6. Cross-document reconciliation: launch_contract <-> task.yaml + probe evidence
        if (
            launch_doc.get("entrypoint") != task.entrypoint.model_dump(mode="json")
            or launch_doc.get("environment") != task.environment.model_dump(mode="json")
            or launch_doc.get("requested_scope") != task.scope.model_dump(mode="json")
        ):
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code="LAUNCH_CONTRACT_TASK_MISMATCH",
                first_mismatch="analysis/launch_contract.yaml",
                summary="launch_contract entrypoint/environment/scope disagrees with inputs/task.yaml",
                evidence_path="analysis/launch_contract.yaml",
                expected={
                    "entrypoint": task.entrypoint.model_dump(mode="json"),
                    "environment": task.environment.model_dump(mode="json"),
                    "requested_scope": task.scope.model_dump(mode="json"),
                },
                actual={
                    "entrypoint": launch_doc.get("entrypoint"),
                    "environment": launch_doc.get("environment"),
                    "requested_scope": launch_doc.get("requested_scope"),
                },
            )

        probe_doc = launch_doc.get("probe")
        if not isinstance(probe_doc, dict):
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code="MISSING_PROBE_RECORD",
                first_mismatch="analysis/launch_contract.yaml:probe",
                summary="launch_contract lacks probe evidence record",
                evidence_path="analysis/launch_contract.yaml",
                expected="probe mapping",
                actual=type(probe_doc).__name__,
            )

        probe_status = probe_doc.get("status")
        probe_observed = list(probe_doc.get("observed_scope", ()))
        if probe_doc.get("source_unchanged") is not True:
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code="SOURCE_MUTATED_DURING_ANALYSIS",
                first_mismatch="analysis/launch_contract.yaml:probe.source_unchanged",
                summary="startup probe reported source repository mutation",
                evidence_path="analysis/launch_contract.yaml",
                expected=True,
                actual=probe_doc.get("source_unchanged"),
            )

        if task.scope.startup_probe:
            argv_ok, argv_reason, argv_detail = validate_probe_argv(
                list(probe_doc.get("argv", ())),
                entrypoint_path=task.entrypoint.path,
                cwd=str(probe_doc.get("cwd", task.entrypoint.cwd)),
            )
            if not argv_ok:
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code=argv_reason or "PROBE_ARGV_ENTRYPOINT_MISMATCH",
                    first_mismatch="analysis/launch_contract.yaml:probe.argv",
                    summary=argv_detail,
                    evidence_path="analysis/launch_contract.yaml",
                    expected=task.entrypoint.path,
                    actual=probe_doc.get("argv"),
                )
            if probe_status != "passed" or "startup_probe" not in probe_observed:
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="SCOPE_NOT_SATISFIED",
                    first_mismatch="analysis/launch_contract.yaml:probe.status",
                    summary=(
                        "task declares startup_probe=true, but probe did not pass "
                        f"(status={probe_status!r}, observed={probe_observed})"
                    ),
                    evidence_path="analysis/launch_contract.yaml",
                    expected={"status": "passed", "observed_scope": ["startup_probe"]},
                    actual={"status": probe_status, "observed_scope": probe_observed},
                )
            # Verify structured probe marker is sealed inside the package and matches recorded digest.
            marker_rel = "evidence/probe/probe_marker.json"
            marker_file = pkg_dir / marker_rel
            recorded_marker_sha = probe_doc.get("marker_sha256")
            if (
                marker_rel not in present_paths
                or not marker_file.is_file()
                or not isinstance(recorded_marker_sha, str)
                or _sha256_bytes(marker_file.read_bytes()) != recorded_marker_sha
            ):
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="PROBE_MARKER_MISSING",
                    first_mismatch="evidence/probe/probe_marker.json",
                    summary="startup_probe requires a manifested, digest-verified probe_marker.json",
                    evidence_path=marker_rel,
                    expected=recorded_marker_sha,
                    actual=(
                        _sha256_bytes(marker_file.read_bytes())
                        if marker_file.is_file()
                        else None
                    ),
                )
            try:
                marker_json = json.loads(marker_file.read_text(encoding="utf-8"))
            except Exception:
                marker_json = None
            expected_argv_sha = compute_argv_sha256(list(probe_doc.get("argv", ())))
            recorded_nonce = probe_doc.get("probe_nonce")
            if (
                not isinstance(marker_json, dict)
                or marker_json.get("event") != "startup_probe_ok"
                or marker_json.get("source_commit") != expected_commit
                or marker_json.get("entrypoint_path") != task.entrypoint.path
                or marker_json.get("argv_sha256") != expected_argv_sha
                or probe_doc.get("argv_sha256") != expected_argv_sha
                or not isinstance(recorded_nonce, str)
                or len(recorded_nonce) < 16
                or marker_json.get("probe_nonce") != recorded_nonce
            ):
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="PROBE_MARKER_BINDING_MISMATCH",
                    first_mismatch="evidence/probe/probe_marker.json:binding",
                    summary="probe_marker.json does not match commit, entrypoint, argv_sha256, and probe_nonce",
                    evidence_path=marker_rel,
                    expected={
                        "event": "startup_probe_ok",
                        "source_commit": expected_commit,
                        "entrypoint_path": task.entrypoint.path,
                        "argv_sha256": expected_argv_sha,
                        "probe_nonce": recorded_nonce,
                    },
                    actual=marker_json,
                )
        else:
            if probe_status not in {"not_run", "cli_parsed", "passed"}:
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="INVALID_PROBE_STATUS",
                    first_mismatch="analysis/launch_contract.yaml:probe.status",
                    summary=f"unexpected probe status: {probe_status!r}",
                    evidence_path="analysis/launch_contract.yaml",
                    expected=["not_run", "cli_parsed", "passed"],
                    actual=probe_status,
                )

        # 7. Strict semantic validation of handoff/result.json & pre-publish verdict binding
        expected_checks = {
            "source_identity_sha256": before_snap.identity_sha256,
            "source_tree_oid": before_snap.tree_oid,
            "root_bundle_sha256": bundles_manifest["root"]["sha256"],
            "submodule_bundle_sha256s": sorted(
                [str(item["submodule_path"]), str(item["sha256"])]
                for item in sub_bundle_entries
            ),
            "entrypoint_path": selected_entry["path"],
            "entrypoint_blob_sha256": selected_entry.get("blob_sha256"),
            "assets_overall_status": rechecked_assets.overall_status,
            "asset_digests": {
                rec.id: rec.observed_sha256 for rec in rechecked_assets.records
            },
            "probe_status": probe_status,
            "probe_marker_sha256": (
                probe_doc.get("marker_sha256") if task.scope.startup_probe else None
            ),
            "probe_nonce": (
                probe_doc.get("probe_nonce") if task.scope.startup_probe else None
            ),
        }
        if isinstance(task, IntakeTaskPackV2):
            expected_checks["v2_handoff_artifact_sha256s"] = v2_checks
        expected_checks_sha = _canonical_json_sha256(expected_checks)

        allowed_observed_scope = {"static"}
        if probe_status == "passed" and "startup_probe" in probe_observed:
            allowed_observed_scope.add("startup_probe")

        if require_sealed or require_handoff:
            try:
                handoff = load_handoff_result(pkg_dir / "handoff/result.json")
            except (ContractParseError, ContractValidationError) as error:
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code=error.reason_code,
                    first_mismatch="handoff/result.json:contract_invalid",
                    summary=f"handoff/result.json invalid: {error}",
                    evidence_path="handoff/result.json",
                    expected="valid HandoffResult",
                    actual=error.to_dict(),
                )
            if handoff.execution_status != "completed" or handoff.blocked_reasons:
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="HANDOFF_STATUS_NOT_COMPLETED",
                    first_mismatch="handoff/result.json:execution_status",
                    summary="sealed intake handoff must have execution_status='completed' and empty blocked_reasons",
                    evidence_path="handoff/result.json",
                    expected={"execution_status": "completed", "blocked_reasons": []},
                    actual={
                        "execution_status": handoff.execution_status,
                        "blocked_reasons": handoff.blocked_reasons,
                    },
                )
            if handoff.stage != "f1":
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="HANDOFF_STAGE_MISMATCH",
                    first_mismatch="handoff/result.json:stage",
                    summary=f"expected handoff stage 'f1', got {handoff.stage!r}",
                    evidence_path="handoff/result.json",
                    expected="f1",
                    actual=handoff.stage,
                )
            if handoff.request_sha256 != request_sha:
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="REQUEST_SHA_MISMATCH",
                    first_mismatch="handoff/result.json:request_sha256",
                    summary="handoff request_sha256 does not match inputs/task.yaml",
                    evidence_path="handoff/result.json",
                    expected=request_sha,
                    actual=handoff.request_sha256,
                )
            # Prevent scope inflation: handoff.observed_scope must be a subset of actually evidenced scopes.
            inflated_scopes = sorted(set(handoff.observed_scope) - allowed_observed_scope)
            if inflated_scopes:
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="HANDOFF_SCOPE_EXCEEDED",
                    first_mismatch="handoff/result.json:observed_scope",
                    summary=(
                        "handoff claims unverified observed_scope entries: "
                        + ", ".join(inflated_scopes)
                    ),
                    evidence_path="handoff/result.json",
                    expected=sorted(allowed_observed_scope),
                    actual=list(handoff.observed_scope),
                )
            if task.scope.startup_probe and "startup_probe" not in handoff.observed_scope:
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="SCOPE_NOT_SATISFIED",
                    first_mismatch="handoff/result.json:observed_scope",
                    summary="handoff observed_scope omits required startup_probe",
                    evidence_path="handoff/result.json",
                    expected="startup_probe in observed_scope",
                    actual=list(handoff.observed_scope),
                )

            # Verify handoff.validator_verdict exists, is manifested, matches SHA, is valid JSON,
            # binds this exact validator_id/validator_sha256/request_sha256/commit, and contains no dummy SHA.
            if handoff.validator_verdict is None:
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="HANDOFF_VERDICT_MISSING",
                    first_mismatch="handoff/result.json:validator_verdict",
                    summary="completed handoff lacks validator_verdict reference",
                    evidence_path="handoff/result.json",
                    expected="evidence/validate/pre_publish_verdict.json",
                    actual=None,
                )
            verdict_rel = handoff.validator_verdict.path
            verdict_file = pkg_dir / verdict_rel
            if verdict_rel not in present_paths or not verdict_file.is_file():
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="error",
                    reason_code="HANDOFF_VERDICT_MISSING",
                    first_mismatch="handoff/result.json:validator_verdict.path",
                    summary=f"handoff validator_verdict file missing from manifest: {verdict_rel}",
                    evidence_path=verdict_rel,
                    expected="manifested verdict JSON",
                    actual=None,
                )
            verdict_bytes = verdict_file.read_bytes()
            actual_verdict_sha = _sha256_bytes(verdict_bytes)
            if actual_verdict_sha != handoff.validator_verdict.sha256:
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="error",
                    reason_code="HANDOFF_VERDICT_DIGEST_MISMATCH",
                    first_mismatch="handoff/result.json:validator_verdict.sha256",
                    summary="handoff validator_verdict SHA-256 mismatch",
                    evidence_path=verdict_rel,
                    expected=handoff.validator_verdict.sha256,
                    actual=actual_verdict_sha,
                )
            try:
                verdict_doc = json.loads(verdict_bytes.decode("utf-8"))
            except Exception as error:
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="HANDOFF_VERDICT_INVALID",
                    first_mismatch="handoff/result.json:validator_verdict",
                    summary=f"handoff validator_verdict is not valid JSON: {error}",
                    evidence_path=verdict_rel,
                    expected="pre_publish JSON verdict",
                    actual=str(error),
                )
            if isinstance(verdict_doc, dict) and verdict_doc.get("validator_id") == self.historical_validator_id:
                return self._fail(
                    commit=expected_commit, manifest_sha256=manifest_sha256, status="blocked",
                    reason_code="LEGACY_VALIDATOR_RUNTIME_REQUIRED", first_mismatch="validator_id",
                    summary="historical stages.verify.* v1 sealed packages require their pinned v1 runtime",
                    evidence_path=verdict_rel, expected=self.validator_id, actual=verdict_doc.get("validator_id"),
                )
            if (
                not isinstance(verdict_doc, dict)
                or verdict_doc.get("phase") != "pre_publish"
                or verdict_doc.get("status") != "pass"
                or verdict_doc.get("request_sha256") != request_sha
                or verdict_doc.get("source_commit") != expected_commit
                or verdict_doc.get("validator_id") != self.validator_id
                or verdict_doc.get("validator_version") != self.validator_version
                or verdict_doc.get("validator_sha256") != self.validator_sha256()
                or verdict_doc.get("checks") != expected_checks
                or verdict_doc.get("checks_sha256") != expected_checks_sha
            ):
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="HANDOFF_VERDICT_BINDING_MISMATCH",
                    first_mismatch=f"{verdict_rel}:binding",
                    summary=(
                        "handoff validator_verdict does not match independent pre_publish "
                        "checks and binding digest"
                    ),
                    evidence_path=verdict_rel,
                    expected={
                        "phase": "pre_publish",
                        "status": "pass",
                        "request_sha256": request_sha,
                        "source_commit": expected_commit,
                        "validator_id": self.validator_id,
                        "validator_sha256": self.validator_sha256(),
                        "checks_sha256": expected_checks_sha,
                    },
                    actual=verdict_doc,
                )

        scope_summary = {
            "static": "passed",
            "startup_probe": "passed" if probe_status == "passed" else "not_run",
            "train_step": "not_run",
            "checkpoint_resume": "not_run",
            "task_eval": "not_run",
        }
        return GateResult(
            gate_id=self.gate_id,
            check="intake",
            status="pass",
            binding=GateBinding(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                validator_id=self.validator_id,
                validator_sha256=self.validator_sha256(),
            ),
            summary="sealed intake package verified across manifest, source, entrypoint, assets, and probe scope",
            evidence=(
                GateEvidence(
                    key="source_identity",
                    detail=f"commit={expected_commit} tree={before_snap.tree_oid}",
                    path="inputs/source_fingerprint.json",
                    digest=before_snap.identity_sha256,
                ),
                GateEvidence(
                    key="entrypoint",
                    detail=f"{selected_entry['path']}:{selected_entry['symbol']}",
                    path="analysis/reference_entrypoints.yaml",
                    digest=selected_entry.get("blob_sha256"),
                ),
            ),
            metrics={
                "validator_version": self.validator_version,
                "request_sha256": request_sha,
                "scope_status": scope_summary,
                "checks": expected_checks,
                "checks_sha256": expected_checks_sha,
            },
        )

    def check(self, context: CheckContext) -> GateResult:
        return self.validate_directory(
            package_dir=context.package_dir,
            expected_commit=context.commit,
            manifest_sha256=context.manifest_sha256,
            reference_dir=context.reference_dir,
            require_sealed=True,
        )


def main(argv: list[str] | None = None) -> int:
    """Independent CLI entry for verifying a sealed Stage 1 (`intake`) package."""
    parser = argparse.ArgumentParser(
        description="Independent validator for sealed Stage 1 (intake) packages."
    )
    parser.add_argument("--package-dir", required=True, help="Path to sealed intake package.")
    parser.add_argument(
        "--manifest-sha256",
        required=True,
        help="External trusted manifest SHA-256 from release record.",
    )
    parser.add_argument(
        "--expected-commit",
        required=True,
        help="Expected upstream commit SHA-1/SHA-256.",
    )
    parser.add_argument(
        "--reference-dir",
        default=None,
        help="Optional local upstream repo path for live immutability/blob recheck.",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Optional output path to persist JSON verdict.",
    )
    args = parser.parse_args(argv)

    checker = IntakeChecker()
    result = checker.check(
        CheckContext(
            commit=args.expected_commit,
            manifest_sha256=args.manifest_sha256,
            package_dir=Path(args.package_dir),
            reference_dir=Path(args.reference_dir) if args.reference_dir else None,
        )
    )
    payload = json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

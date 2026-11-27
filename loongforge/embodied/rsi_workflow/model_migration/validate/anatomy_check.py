"""Independent checker and CLI for Stage 2 (`anatomy`) sealed behavior contract packages.

Verifies:
- Sealed package integrity (`core.artifacts.verify`, rejecting `.staging` unsealed dirs);
- Required Stage 2 anatomy layout exists in the manifest;
- Strict `AnatomyBehaviorContract` (`behavior_contract.yaml`) & `HandoffResult` (`handoff/result.json`);
- Cross-stage binding to the sealed Stage 1 (`intake`) manifest SHA-256 and upstream commit;
- Distinct OS PIDs for `attempt-1` and `attempt-2`, identical initial weights, real
  parameter updates (rejecting `step_counter_only`), full per-rank receipts, and clean
  independent re-comparison via `compare_attempts`;
- Mandatory mock/real execution labels (`execution_environment`, `evidence_scope`);
- Composite `validator_sha256` covering `anatomy_check`, `base`, `judger`, `artifacts`,
  `compare`, `contract`, `capture`, `adapters/fake`, and `core/contracts` models.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from core import artifacts as _artifacts_module
from core.artifacts import STAGING_SUFFIX, ArtifactError, PackageRef, verify as verify_package
from core.contracts import (
    ContractParseError,
    ContractValidationError,
    common as _contracts_common,
    errors as _contracts_errors,
    f2 as _contracts_f2,
    handoff as _contracts_handoff,
    load_anatomy_behavior_contract,
    load_handoff_result,
    load_intake_task_pack,
)
from core.contracts.common import load_yaml_document
from core.orchestrator.judger import GateBinding, GateEvidence, GateOutcome, GateResult
from core.stages.f2_anatomy import capture as _capture_module
from core.stages.f2_anatomy import compare as _compare_module
from core.stages.f2_anatomy import contract as _contract_module
from core.stages.f2_anatomy import spec as _spec_module
from core.stages.f2_anatomy.spec import load_anatomy_spec, verify_frozen_spec
from core.stages.f2_anatomy.adapters import fake as _fake_adapter_module
from core.stages.f2_anatomy.adapters import openpi_jax as _openpi_jax_module
from core.stages.f2_anatomy.compare import REQUIRED_PHASE_IDS, compare_attempts
from core.stages.f2_anatomy.contract import (
    observation_boundaries,
    negative_injections,
)
from core.stages.f1_intake.assets import triage_assets
from validate import intake_check as _intake_check_module
from validate.intake_check import IntakeChecker

from .base import CheckContext, validator_source_sha256

REQUIRED_ANATOMY_PACKAGE_FILES: tuple[str, ...] = (
    "inputs/anatomy_case.yaml",
    "analysis/step_anatomy.yaml",
    "analysis/behavior_contract.yaml",
    "analysis/observation_plan.yaml",
    "analysis/anatomy_spec.yaml",
    "analysis/spec_lock.json",
    "evidence/spec_freeze/anatomy_spec.yaml",
    "evidence/spec_freeze/artifact_manifest.json",
    "evidence/attempt-1/capture.json",
    "evidence/attempt-2/capture.json",
    "evidence/comparison.json",
    "handoff/result.json",
)
PRE_PUBLISH_ANATOMY_FILES: tuple[str, ...] = tuple(
    rel for rel in REQUIRED_ANATOMY_PACKAGE_FILES if rel != "handoff/result.json"
)
REQUIRED_ASSERTION_IDS: tuple[str, ...] = (
    "sample-and-camera-order",
    "rng-consumer-order",
    "model-forward-count",
    "loss-and-grad-endpoints",
    "real-parameter-update",
    "action-semantics",
)


def _anatomy_dependency_sources() -> tuple[Path, ...]:
    sources: list[Path] = []
    for mod in (
        _artifacts_module,
        _contracts_common,
        _contracts_errors,
        _contracts_f2,
        _contracts_handoff,
        _compare_module,
        _contract_module,
        _spec_module,
        _capture_module,
        _fake_adapter_module,
        _openpi_jax_module,
        _intake_check_module,
    ):
        src = inspect.getsourcefile(mod)
        if src is not None:
            sources.append(Path(src).resolve())
    return tuple(sources) + _intake_check_module._intake_dependency_sources()


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _validate_fake_launch_receipt(
    receipt: Any,
    *,
    case_id: str,
    seed: int,
    world_size: int,
    sample_ids: list[str],
    spec_file: Path,
    package_root: Path,
    expected_output: str,
) -> tuple[bool, str]:
    """Strictly verify a fake subprocess launch_receipt's command, exit code, and timestamp."""
    if not isinstance(receipt, dict):
        return False, "launch_receipt must be a mapping"
    if receipt.get("exit_code") != 0:
        return False, f"launch_receipt exit_code must be 0, got {receipt.get('exit_code')!r}"
    executed_at = receipt.get("executed_at")
    if not isinstance(executed_at, str) or not executed_at.strip():
        return False, "launch_receipt.executed_at must be a valid ISO-8601 timestamp"
    try:
        datetime.fromisoformat(executed_at)
    except ValueError:
        return False, f"launch_receipt.executed_at is not valid ISO-8601: {executed_at!r}"

    cmd = receipt.get("command")
    if not isinstance(cmd, list) or len(cmd) < 4 or not all(isinstance(x, str) for x in cmd):
        return False, "launch_receipt.command must be a non-empty list of strings"
    # This is canonical evidence, not an executable command. Fresh replay uses
    # the validator's sys.executable, never a recorded host path or task PATH.
    if cmd[0] != "python3":
        return False, "launch_receipt.command[0] must be the portable name 'python3'"
    if cmd[1:3] != ["-m", "core.stages.f2_anatomy.adapters.fake"]:
        return False, f"launch_receipt.command must invoke '-m core.stages.f2_anatomy.adapters.fake', got {cmd[1:3]!r}"

    flags: dict[str, str] = {}
    idx = 3
    while idx < len(cmd):
        token = cmd[idx]
        if token in {"--output", "--seed", "--world-size", "--case-id", "--sample-ids", "--inject", "--spec-file"}:
            if idx + 1 >= len(cmd):
                return False, f"launch_receipt.command missing value for {token}"
            if token in flags:
                return False, f"duplicate launch flag {token}"
            flags[token] = cmd[idx + 1]
            idx += 2
        else:
            return False, f"launch_receipt.command contains unexpected argument {token!r}"

    if flags.get("--spec-file") != spec_file.relative_to(package_root).as_posix():
        return False, "launch_receipt must reference the pre-capture sealed spec"
    if "--inject" in flags:
        return False, "launch_receipt.command must not carry --inject for passing capture"
    if flags.get("--seed") != str(seed):
        return False, f"launch_receipt --seed {flags.get('--seed')!r} != {seed}"
    if flags.get("--world-size") != str(world_size):
        return False, f"launch_receipt --world-size {flags.get('--world-size')!r} != {world_size}"
    if flags.get("--case-id") != case_id:
        return False, f"launch_receipt --case-id {flags.get('--case-id')!r} != {case_id!r}"
    if flags.get("--sample-ids") != ",".join(sample_ids):
        return False, "launch_receipt --sample-ids disagrees with frozen case sample_ids"
    if flags.get("--output") != expected_output:
        return False, f"launch_receipt --output must equal package-relative {expected_output!r}"
    return True, "ok"


class AnatomyChecker:
    """Independent validator for Stage 2 (`anatomy`) two-attempt capture packages."""

    gate_id = "anatomy"
    check_name = "anatomy"
    validator_id = "validate.anatomy_check"
    validator_version = "2"
    historical_validator_id = "stages.verify.anatomy_check"

    def validator_sha256(self) -> str:
        return validator_source_sha256(
            type(self),
            extra_sources=_anatomy_dependency_sources(),
        )

    def build_pre_publish_verdict(
        self,
        *,
        staging_dir: Path,
        expected_commit: str,
        case_sha256: str,
        intake_package_dir: Path,
        intake_manifest_sha256: str,
        spec_manifest_sha256: str,
    ) -> tuple[dict[str, Any] | None, GateResult]:
        """Run real pre-publish validation on `staging_dir` and return bound verdict payload."""
        gate = self.validate_directory(
            package_dir=staging_dir,
            expected_commit=expected_commit,
            manifest_sha256=case_sha256,
            intake_package_dir=intake_package_dir,
            expected_intake_manifest_sha256=intake_manifest_sha256,
            expected_spec_manifest_sha256=spec_manifest_sha256,
            require_sealed=False,
            require_handoff=False,
        )
        if gate.status != "pass":
            return None, gate
        checks = gate.metrics.get("checks")
        framework_commit = gate.metrics.get("framework_commit")
        if not isinstance(checks, dict) or not isinstance(framework_commit, str):
            return None, self._fail(
                commit=expected_commit,
                manifest_sha256=case_sha256,
                status="error",
                reason_code="PRE_PUBLISH_CHECKS_MISSING",
                first_mismatch="pre_publish:checks",
                summary="pre-publish validation did not produce structured checks map",
                evidence_path="evidence/validate/pre_publish_verdict.json",
                expected="checks map + framework_commit",
                actual=gate.metrics,
            )
        payload = {
            "schema_version": 1,
            "phase": "pre_publish",
            "status": "pass",
            "case_sha256": case_sha256,
            "intake_manifest_sha256": intake_manifest_sha256,
            "source_commit": expected_commit,
            "framework_commit": framework_commit,
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
            check="anatomy",
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
        intake_package_dir: Path | None = None,
        expected_intake_manifest_sha256: str | None = None,
        expected_spec_manifest_sha256: str | None = None,
        require_sealed: bool = True,
        require_handoff: bool = True,
    ) -> GateResult:
        """Validate a sealed anatomy package (or pre-publish staging dir)."""
        pkg_dir = Path(package_dir)

        if require_sealed:
            if pkg_dir.name.endswith(STAGING_SUFFIX):
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="UNSEALED_PACKAGE_REJECTED",
                    first_mismatch="package_dir:staging_suffix",
                    summary="unsealed .staging directory cannot be accepted as a released anatomy package",
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
            REQUIRED_ANATOMY_PACKAGE_FILES
            if (require_sealed or require_handoff)
            else PRE_PUBLISH_ANATOMY_FILES
        )
        missing_files = [rel for rel in required_layout if rel not in present_paths]
        if missing_files:
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code="MISSING_PACKAGE_FILE",
                first_mismatch=f"layout:{missing_files[0]}",
                summary=f"required anatomy file missing: {missing_files[0]}",
                evidence_path=missing_files[0],
                expected=list(REQUIRED_ANATOMY_PACKAGE_FILES),
                actual=sorted(present_paths),
            )

        # 2. Load & validate behavior_contract.yaml, anatomy_case.yaml, step_anatomy.yaml, observation_plan.yaml
        try:
            case_doc = load_yaml_document(pkg_dir / "inputs/anatomy_case.yaml")
            step_doc = load_yaml_document(pkg_dir / "analysis/step_anatomy.yaml")
            obs_doc = load_yaml_document(pkg_dir / "analysis/observation_plan.yaml")
            contract = load_anatomy_behavior_contract(
                pkg_dir / "analysis/behavior_contract.yaml"
            )
        except (ContractParseError, ContractValidationError) as error:
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code=error.reason_code,
                first_mismatch="analysis/behavior_contract.yaml:contract_invalid",
                summary=f"anatomy contract invalid: {error}",
                evidence_path="analysis/behavior_contract.yaml",
                expected="valid AnatomyBehaviorContract",
                actual=error.to_dict(),
            )

        if contract.validator.id == self.historical_validator_id:
            return self._fail(
                commit=expected_commit, manifest_sha256=manifest_sha256, status="blocked",
                reason_code="LEGACY_VALIDATOR_RUNTIME_REQUIRED", first_mismatch="validator.id",
                summary="historical stages.verify.* v1 sealed packages require their pinned v1 runtime",
                evidence_path="analysis/behavior_contract.yaml", expected=self.validator_id, actual=contract.validator.id,
            )
        case_sha = _sha256_bytes((pkg_dir / "inputs/anatomy_case.yaml").read_bytes())

        # 3. Cross-check upstream commit & Stage 1 (`intake`) manifest binding
        if intake_package_dir is None or not expected_intake_manifest_sha256:
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code="INTAKE_REFERENCE_REQUIRED",
                first_mismatch="context.intake_reference",
                summary="AnatomyChecker requires both intake_package_dir and intake_manifest_sha256",
                evidence_path="inputs/anatomy_case.yaml",
                expected="non-null intake_package_dir and intake_manifest_sha256",
                actual={
                    "intake_package_dir": str(intake_package_dir) if intake_package_dir else None,
                    "intake_manifest_sha256": expected_intake_manifest_sha256,
                },
            )

        if contract.case.upstream_commit != expected_commit:
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code="UPSTREAM_COMMIT_MISMATCH",
                first_mismatch="analysis/behavior_contract.yaml:case.upstream_commit",
                summary="behavior_contract upstream_commit does not match expected commit",
                evidence_path="analysis/behavior_contract.yaml",
                expected=expected_commit,
                actual=contract.case.upstream_commit,
            )

        recorded_intake_sha = contract.case.f1_manifest_sha256
        if (
            case_doc.get("intake_manifest_sha256") != recorded_intake_sha
            or step_doc.get("intake_manifest_sha256") != recorded_intake_sha
            or obs_doc.get("intake_manifest_sha256") != recorded_intake_sha
        ):
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code="INTAKE_MANIFEST_MISMATCH",
                first_mismatch="analysis/step_anatomy.yaml:intake_manifest_sha256",
                summary="internal intake_manifest_sha256 mismatch across anatomy documents",
                evidence_path="analysis/step_anatomy.yaml",
                expected=recorded_intake_sha,
                actual={
                    "case": case_doc.get("intake_manifest_sha256"),
                    "step_anatomy": step_doc.get("intake_manifest_sha256"),
                    "observation_plan": obs_doc.get("intake_manifest_sha256"),
                },
            )

        if recorded_intake_sha != expected_intake_manifest_sha256:
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code="INTAKE_MANIFEST_MISMATCH",
                first_mismatch="analysis/behavior_contract.yaml:case.f1_manifest_sha256",
                summary="anatomy package intake manifest SHA differs from expected Stage 1 release",
                evidence_path="analysis/behavior_contract.yaml",
                expected=expected_intake_manifest_sha256,
                actual=recorded_intake_sha,
            )

        intake_dir_path = Path(intake_package_dir)
        intake_gate = IntakeChecker().check(
            CheckContext(
                commit=expected_commit,
                manifest_sha256=expected_intake_manifest_sha256,
                package_dir=intake_dir_path,
            )
        )
        if intake_gate.status != "pass":
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked" if intake_gate.status == "blocked" else "error",
                reason_code="INTAKE_CHECK_FAILED",
                first_mismatch=f"intake:{intake_gate.first_mismatch or intake_gate.reason_codes}",
                summary=f"Stage 1 IntakeChecker failed: {intake_gate.summary}",
                evidence_path=str(intake_dir_path),
                expected="IntakeChecker status == 'pass'",
                actual={
                    "status": intake_gate.status,
                    "reason_codes": list(intake_gate.reason_codes),
                },
            )

        try:
            intake_task = load_intake_task_pack(intake_dir_path / "inputs" / "task.yaml")
        except (ContractParseError, ContractValidationError) as error:
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="error",
                reason_code="INTAKE_TASK_INVALID",
                first_mismatch="intake:inputs/task.yaml",
                summary=f"failed to load Stage 1 task.yaml: {error}",
                evidence_path=str(intake_dir_path / "inputs" / "task.yaml"),
                expected="valid IntakeTaskPack",
                actual=error.to_dict(),
            )
        if not intake_task.scope.train_step:
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code="INTAKE_TRAIN_STEP_SCOPE_DISABLED",
                first_mismatch="intake:scope.train_step",
                summary="Stage 1 task.scope.train_step is false",
                evidence_path=str(intake_dir_path / "inputs" / "task.yaml"),
                expected=True,
                actual=False,
            )
        f2_assets = triage_assets(intake_task.assets, required_scope="f2_train_step")
        if f2_assets.overall_status != "ready":
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code=(
                    f2_assets.blocked_reasons[0]
                    if f2_assets.blocked_reasons
                    else (
                        f2_assets.waiting_reasons[0]
                        if f2_assets.waiting_reasons
                        else "INTAKE_ASSET_NOT_READY"
                    )
                ),
                first_mismatch="intake:f2_train_step_assets",
                summary=f"Stage 1 f2_train_step assets not ready (status={f2_assets.overall_status})",
                evidence_path=str(intake_dir_path / "analysis" / "resource_manifest.yaml"),
                expected="ready",
                actual=f2_assets.overall_status,
            )

        if not expected_spec_manifest_sha256:
            return self._fail(
                commit=expected_commit, manifest_sha256=manifest_sha256, status="blocked",
                reason_code="SPEC_TRUST_REQUIRED", first_mismatch="context.spec_manifest_sha256",
                summary="caller-held trusted spec manifest digest is required; packaged self-declaration is not trusted",
                evidence_path="analysis/spec_lock.json", expected="external spec manifest SHA", actual=None,
            )
        try:
            spec, spec_file = verify_frozen_spec(
                pkg_dir, intake_manifest_sha256=recorded_intake_sha, upstream_commit=expected_commit,
                trusted_spec_manifest_sha256=expected_spec_manifest_sha256,
            )
            if any(doc.get("anatomy_spec_sha256") != spec.sha256() for doc in (case_doc, step_doc, obs_doc)):
                raise ValueError("cross-document spec digest mismatch")
            if (contract.case.anatomy_spec_sha256 != spec.sha256()
                    or contract.case.spec_manifest_sha256 != expected_spec_manifest_sha256
                    or any(doc.get("spec_manifest_sha256") != expected_spec_manifest_sha256 for doc in (case_doc, step_doc, obs_doc))):
                raise ValueError("behavior contract spec digest mismatch")
            if intake_task.schema_version == 2:
                intake_spec_doc = load_yaml_document(intake_dir_path / "analysis/anatomy_spec.yaml")
                intake_spec = load_anatomy_spec(intake_dir_path / "analysis/anatomy_spec.yaml")
                if (spec != intake_spec or load_yaml_document(pkg_dir / "analysis/anatomy_spec.yaml")["source_ref"] != intake_spec_doc["source_ref"]):
                    raise ValueError("AnatomySpec differs from frozen v2 Intake handoff")
        except (ValueError, OSError, KeyError, ArtifactError, ContractParseError) as error:
            return self._fail(
                commit=expected_commit, manifest_sha256=manifest_sha256, status="blocked",
                reason_code="ANATOMY_SPEC_INVALID", first_mismatch="analysis/anatomy_spec.yaml",
                summary=str(error), evidence_path="analysis/anatomy_spec.yaml",
                expected="verified pre-capture frozen spec", actual=str(error),
            )

        expected_framework_commit = intake_task.target.commit
        expected_precision = contract.case.precision.model_dump(mode="json")

        # Batch 3 P0: fail-closed on non-fake backends (cannot masquerade fake CPU evidence
        # as real jax/pytorch evidence by relabeling behavior_contract.yaml).
        if contract.case.precision.backend != "fake":
            backend = contract.case.precision.backend
            actual: dict[str, Any] = {"backend": backend}
            summary = (
                f"backend={backend!r} cannot pass offline verification "
                "without independent real-hardware replay"
            )
            if backend == "jax":
                from core.stages.f1_intake.source import SourceError, restore_from_bundle_manifest
                from core.stages.f2_anatomy.adapters.openpi_jax import OpenPIJaxAdapter, OpenPIJaxConfig

                # source.repo is identity metadata and may be a remote URL.
                # Use the verified Intake's pinned offline tree, not a host path.
                try:
                    fingerprint = json.loads((intake_dir_path / "inputs/source_fingerprint.json").read_text())
                    with tempfile.TemporaryDirectory(prefix="anatomy-jax-preflight-") as temp:
                        source = Path(temp) / "source"
                        restore_from_bundle_manifest(intake_dir_path, fingerprint["bundles"], source)
                        preflight = OpenPIJaxAdapter().preflight(
                            OpenPIJaxConfig(upstream_repo=source, require_gpu=True)
                        )
                    actual.update(
                        preflight_status=preflight.status,
                        preflight_reasons=list(preflight.reason_codes),
                    )
                except (SourceError, OSError, ValueError, KeyError) as error:
                    actual.update(
                        preflight_status="not_run",
                        preflight_reasons=["JAX_PREFLIGHT_SOURCE_UNAVAILABLE"],
                        preflight_error=str(error),
                    )
                summary += (
                    f" (preflight={actual['preflight_status']}, "
                    f"reasons={actual['preflight_reasons']})"
                )
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code="UNSUPPORTED_BACKEND_REPLAY_REQUIRED",
                first_mismatch="analysis/behavior_contract.yaml:case.precision.backend",
                summary=summary,
                evidence_path="analysis/behavior_contract.yaml",
                expected="fake (for offline CPU closed-loop verification)",
                actual=actual,
            )

        # P2-2: behavior_contract must have empty unknowns/blocked_reasons.
        if contract.unknowns or contract.blocked_reasons:
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code="CONTRACT_HAS_UNKNOWNS_OR_BLOCKERS",
                first_mismatch="analysis/behavior_contract.yaml:unknowns",
                summary="passing behavior_contract must have empty unknowns and blocked_reasons",
                evidence_path="analysis/behavior_contract.yaml",
                expected={"unknowns": [], "blocked_reasons": []},
                actual={
                    "unknowns": contract.unknowns,
                    "blocked_reasons": contract.blocked_reasons,
                },
            )

        if (
            case_doc.get("case_id") != contract.case.id
            or step_doc.get("case_id") != contract.case.id
            or obs_doc.get("case_id") != contract.case.id
            or case_doc.get("upstream_commit") != expected_commit
            or step_doc.get("upstream_commit") != expected_commit
            or case_doc.get("framework_commit") != expected_framework_commit
            or contract.case.framework_commit != expected_framework_commit
            or step_doc.get("framework_commit") != expected_framework_commit
            or case_doc.get("seed") != contract.case.seed
            or step_doc.get("seed") != contract.case.seed
            or case_doc.get("sample_ids") != contract.case.batch.sample_ids
            or step_doc.get("sample_ids") != contract.case.batch.sample_ids
            or case_doc.get("precision") != expected_precision
            or step_doc.get("precision") != expected_precision
            or obs_doc.get("precision") != expected_precision
            or contract.case.gpu_count != 0
        ):
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code="CROSS_DOCUMENT_MISMATCH",
                first_mismatch="analysis:cross_document_binding",
                summary=(
                    "anatomy_case, behavior_contract, step_anatomy, observation_plan, "
                    "and Stage 1 task disagree on case_id/commits/seed/sample_ids/precision"
                ),
                evidence_path="inputs/anatomy_case.yaml",
                expected={
                    "case_id": contract.case.id,
                    "upstream_commit": expected_commit,
                    "framework_commit": expected_framework_commit,
                    "seed": contract.case.seed,
                    "sample_ids": contract.case.batch.sample_ids,
                    "precision": expected_precision,
                    "gpu_count": 0,
                },
                actual={
                    "case_doc": {
                        "case_id": case_doc.get("case_id"),
                        "upstream_commit": case_doc.get("upstream_commit"),
                        "framework_commit": case_doc.get("framework_commit"),
                        "seed": case_doc.get("seed"),
                        "sample_ids": case_doc.get("sample_ids"),
                        "precision": case_doc.get("precision"),
                    },
                    "contract_framework_commit": contract.case.framework_commit,
                    "contract_gpu_count": contract.case.gpu_count,
                    "step_doc_framework_commit": step_doc.get("framework_commit"),
                    "step_doc_precision": step_doc.get("precision"),
                    "obs_precision": obs_doc.get("precision"),
                    "obs_case_id": obs_doc.get("case_id"),
                },
            )

        # P1-1 & P2-2: Validate observation_plan frozen required camera roles, RNG consumers,
        # boundaries, and negative_injections.
        expected_boundaries = observation_boundaries(spec)
        expected_negatives = negative_injections(spec)
        if (
            obs_doc.get("required_camera_roles") != list(spec.required_camera_roles)
            or obs_doc.get("required_rng_consumers") != list(spec.required_rng_consumers)
            or obs_doc.get("optional_camera_roles") != list(spec.optional_camera_roles)
            or obs_doc.get("action") != spec.to_dict()["action"]
            or obs_doc.get("views") != spec.to_dict()["views"]
            or obs_doc.get("boundaries") != expected_boundaries
            or obs_doc.get("negative_injections") != expected_negatives
        ):
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code="OBSERVATION_PLAN_MISMATCH",
                first_mismatch="analysis/observation_plan.yaml",
                summary=(
                    "observation_plan required_camera_roles, required_rng_consumers, "
                    "boundaries, or negative_injections do not match frozen policy"
                ),
                evidence_path="analysis/observation_plan.yaml",
                expected={
                    "required_camera_roles": list(spec.required_camera_roles),
                    "required_rng_consumers": list(spec.required_rng_consumers),
                    "boundaries": expected_boundaries,
                    "negative_injections": expected_negatives,
                },
                actual={
                    "required_camera_roles": obs_doc.get("required_camera_roles"),
                    "required_rng_consumers": obs_doc.get("required_rng_consumers"),
                    "boundaries": obs_doc.get("boundaries"),
                    "negative_injections": obs_doc.get("negative_injections"),
                },
            )

        phase_required_map = {p.id: p.required for p in contract.phases}
        missing_req_phases = [
            pid for pid in REQUIRED_PHASE_IDS if phase_required_map.get(pid) is not True
        ]
        if missing_req_phases:
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code="REQUIRED_PHASE_DISABLED",
                first_mismatch=f"analysis/behavior_contract.yaml:phases[{missing_req_phases[0]}]",
                summary=(
                    "behavior_contract disables or omits mandatory required phases: "
                    + ", ".join(missing_req_phases)
                ),
                evidence_path="analysis/behavior_contract.yaml",
                expected={pid: True for pid in REQUIRED_PHASE_IDS},
                actual=phase_required_map,
            )

        assertion_required_map = {a.id: a.required for a in contract.assertions}
        missing_req_assertions = [
            aid for aid in REQUIRED_ASSERTION_IDS if assertion_required_map.get(aid) is not True
        ]
        if missing_req_assertions:
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code="REQUIRED_ASSERTION_DISABLED",
                first_mismatch=f"analysis/behavior_contract.yaml:assertions[{missing_req_assertions[0]}]",
                summary=(
                    "behavior_contract disables or omits mandatory required assertions: "
                    + ", ".join(missing_req_assertions)
                ),
                evidence_path="analysis/behavior_contract.yaml",
                expected={aid: True for aid in REQUIRED_ASSERTION_IDS},
                actual=assertion_required_map,
            )

        # 4. Verify contract validator SHA matches this checker
        expected_val_sha = self.validator_sha256()
        if (
            contract.validator.sha256 != expected_val_sha
            or contract.validator.id != self.validator_id
            or contract.validator.version != self.validator_version
            or obs_doc.get("validator_sha256") != expected_val_sha
            or obs_doc.get("validator_id") != self.validator_id
        ):
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code="VALIDATOR_SHA_MISMATCH",
                first_mismatch="analysis/behavior_contract.yaml:validator.sha256",
                summary="behavior_contract/observation_plan validator binding does not match AnatomyChecker",
                evidence_path="analysis/behavior_contract.yaml",
                expected={
                    "id": self.validator_id,
                    "version": self.validator_version,
                    "sha256": expected_val_sha,
                },
                actual={
                    "contract_id": contract.validator.id,
                    "contract_version": contract.validator.version,
                    "contract": contract.validator.sha256,
                    "observation_plan": obs_doc.get("validator_sha256"),
                },
            )

        # 5. Load attempt-1 & attempt-2 captures, assert distinct PIDs & re-run comparator independently
        try:
            attempt1 = json.loads(
                (pkg_dir / "evidence/attempt-1/capture.json").read_text(encoding="utf-8")
            )
            attempt2 = json.loads(
                (pkg_dir / "evidence/attempt-2/capture.json").read_text(encoding="utf-8")
            )
            recorded_cmp = json.loads(
                (pkg_dir / "evidence/comparison.json").read_text(encoding="utf-8")
            )
        except Exception as error:
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="error",
                reason_code="CAPTURE_JSON_INVALID",
                first_mismatch="evidence/attempt-1/capture.json",
                summary=f"failed to read attempt capture JSON: {error}",
                evidence_path="evidence/attempt-1/capture.json",
                expected="valid JSON capture object",
                actual=str(error),
            )

        if attempt1.get("pid") == attempt2.get("pid"):
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code="SAME_PROCESS_ATTEMPTS",
                first_mismatch="evidence/attempt-2/capture.json:pid",
                summary="attempt-1 and attempt-2 share the same PID; independent subprocesses required",
                evidence_path="evidence/attempt-2/capture.json",
                expected="attempt1.pid != attempt2.pid",
                actual={"attempt1_pid": attempt1.get("pid"), "attempt2_pid": attempt2.get("pid")},
            )

        # Verify launch_receipt + labels + case binding on BOTH attempts
        for label, att, ev_path in (
            ("attempt-1", attempt1, "evidence/attempt-1/capture.json"),
            ("attempt-2", attempt2, "evidence/attempt-2/capture.json"),
        ):
            receipt = att.get("launch_receipt")
            receipt_ok, receipt_detail = _validate_fake_launch_receipt(
                receipt,
                case_id=contract.case.id,
                seed=int(contract.case.seed),
                world_size=int(case_doc["world_size"]),
                sample_ids=list(contract.case.batch.sample_ids),
                spec_file=spec_file, package_root=pkg_dir, expected_output=ev_path,
            )
            if not receipt_ok:
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="LAUNCH_RECEIPT_INVALID",
                    first_mismatch=f"{ev_path}:launch_receipt",
                    summary=f"{label} invalid launch_receipt: {receipt_detail}",
                    evidence_path=ev_path,
                    expected="trusted interpreter -m core.stages.f2_anatomy.adapters.fake matching frozen case",
                    actual=receipt,
                )
            if (
                att.get("spec_manifest_sha256") != expected_spec_manifest_sha256
                or att.get("case_id") != contract.case.id
                or att.get("seed") != contract.case.seed
                or att.get("sample_ids") != contract.case.batch.sample_ids
                or att.get("world_size") != case_doc.get("world_size")
                or att.get("precision") != expected_precision
            ):
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="ATTEMPT_CASE_MISMATCH",
                    first_mismatch=f"{ev_path}:case_binding",
                    summary=f"{label} case_id/seed/sample_ids/world_size/precision disagrees with frozen case",
                    evidence_path=ev_path,
                    expected={
                        "case_id": contract.case.id,
                        "seed": contract.case.seed,
                        "sample_ids": contract.case.batch.sample_ids,
                        "world_size": case_doc.get("world_size"),
                        "precision": expected_precision,
                    },
                    actual={
                        "case_id": att.get("case_id"),
                        "seed": att.get("seed"),
                        "sample_ids": att.get("sample_ids"),
                        "world_size": att.get("world_size"),
                        "precision": att.get("precision"),
                    },
                )
            if (
                att.get("execution_environment") != "cpu_mock"
                or att.get("evidence_scope") != "orchestration_only"
                or att.get("adapter_id") != "fake_cpu_step_adapter"
            ):
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="MOCK_SCOPE_TAG_MISSING",
                    first_mismatch=f"{ev_path}:execution_environment",
                    summary=f"{label} fake backend capture must carry cpu_mock + orchestration_only tags",
                    evidence_path=ev_path,
                    expected={
                        "execution_environment": "cpu_mock",
                        "evidence_scope": "orchestration_only",
                        "adapter_id": "fake_cpu_step_adapter",
                    },
                    actual={
                        "execution_environment": att.get("execution_environment"),
                        "evidence_scope": att.get("evidence_scope"),
                        "adapter_id": att.get("adapter_id"),
                    },
                )

        exec_env = attempt1.get("execution_environment")
        ev_scope = attempt1.get("evidence_scope")

        # Independent re-comparison (never trusts pre-written comparison.json blindly)
        recompared = compare_attempts(
            attempt1,
            attempt2,
            evidence_path="evidence/attempt-2/capture.json",
            spec=spec,
        )
        if recompared.status != "pass":
            reason = (
                recompared.reason_codes[0]
                if recompared.reason_codes
                else "ATTEMPT_COMPARISON_FAILED"
            )
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code=reason,
                first_mismatch=recompared.first_mismatch or "evidence/comparison.json",
                summary=recompared.summary,
                evidence_path=recompared.evidence_path or "evidence/comparison.json",
                expected=recompared.expected,
                actual=recompared.actual,
            )
        if recorded_cmp != recompared.to_dict():
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code="RECORDED_COMPARISON_MISMATCH",
                first_mismatch="evidence/comparison.json",
                summary="recorded comparison.json disagrees with independent compare_attempts recomputation",
                evidence_path="evidence/comparison.json",
                expected=recompared.to_dict(),
                actual=recorded_cmp,
            )

        # Independent two-subprocess replay verification (proves fresh process isolation & determinism)
        from core.stages.f2_anatomy.capture import _run_fake_subprocess

        replay_pids: list[int] = []
        replay_receipts: list[dict[str, Any]] = []
        with tempfile.TemporaryDirectory(prefix="anatomy-verify-replay-") as replay_tmp:
            rtmp = Path(replay_tmp)
            rep1 = _run_fake_subprocess(
                output_path=rtmp / "replay-1.json",
                seed=int(contract.case.seed),
                world_size=int(case_doc["world_size"]),
                case_id=contract.case.id,
                sample_ids=tuple(contract.case.batch.sample_ids),
                spec_file=spec_file, package_root=pkg_dir,
            )
            rep2 = _run_fake_subprocess(
                output_path=rtmp / "replay-2.json",
                seed=int(contract.case.seed),
                world_size=int(case_doc["world_size"]),
                case_id=contract.case.id,
                sample_ids=tuple(contract.case.batch.sample_ids),
                spec_file=spec_file, package_root=pkg_dir,
            )
            replay_pids = [int(rep1["pid"]), int(rep2["pid"])]
            replay_receipts = [
                {
                    "pid": int(rep1["pid"]),
                    "command": list(rep1["launch_receipt"]["command"]),
                    "exit_code": int(rep1["launch_receipt"]["exit_code"]),
                },
                {
                    "pid": int(rep2["pid"]),
                    "command": list(rep2["launch_receipt"]["command"]),
                    "exit_code": int(rep2["launch_receipt"]["exit_code"]),
                },
            ]
            if rep1["pid"] == rep2["pid"]:
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="error",
                    reason_code="REPLAY_SAME_PID",
                    first_mismatch="replay.pids",
                    summary="independent checker replay subprocesses unexpectedly shared a PID",
                    evidence_path="evidence/attempt-1/capture.json",
                    expected="distinct subprocess PIDs",
                    actual=replay_pids,
                )
            for label, packaged_attempt, replayed_attempt, ev_path in (
                ("attempt-1", attempt1, rep1, "evidence/attempt-1/capture.json"),
                ("attempt-2", attempt2, rep2, "evidence/attempt-2/capture.json"),
            ):
                rep_cmp = compare_attempts(
                    replayed_attempt,
                    packaged_attempt,
                    evidence_path=ev_path,
                    spec=spec,
                )
                if rep_cmp.status != "pass":
                    return self._fail(
                        commit=expected_commit,
                        manifest_sha256=manifest_sha256,
                        status="blocked",
                        reason_code="INDEPENDENT_REPLAY_MISMATCH",
                        first_mismatch=rep_cmp.first_mismatch or ev_path,
                        summary=f"packaged {label} diverges from independent fresh subprocess replay: {rep_cmp.summary}",
                        evidence_path=ev_path,
                        expected=rep_cmp.expected,
                        actual=rep_cmp.actual,
                    )

        # Cross-check step_anatomy.yaml against attempt-1 facts (and reject legacy self-reported PIDs)
        if (
            "attempt_pids" in step_doc
            or step_doc.get("weights_before_digest") != attempt1.get("weights_before_digest")
            or step_doc.get("weights_after_digest") != attempt1.get("weights_after_digest")
            or step_doc.get("execution_environment") != exec_env
            or step_doc.get("evidence_scope") != ev_scope
            or step_doc.get("precision") != expected_precision
            or step_doc.get("cameras") != attempt1.get("cameras")
            or step_doc.get("rng_events") != attempt1.get("rng_events")
            or step_doc.get("action_evidence") != attempt1.get("action_evidence")
            or step_doc.get("phases") != attempt1.get("phases")
            or step_doc.get("loss") != attempt1.get("loss")
            or step_doc.get("grad_global_norm") != attempt1.get("grad_global_norm")
            or step_doc.get("changed_parameters") != attempt1.get("changed_parameters")
        ):
            return self._fail(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                status="blocked",
                reason_code="STEP_ANATOMY_EVIDENCE_MISMATCH",
                first_mismatch="analysis/step_anatomy.yaml",
                summary="step_anatomy.yaml facts do not match attempt-1/attempt-2 captures",
                evidence_path="analysis/step_anatomy.yaml",
                expected={
                    "weights_before_digest": attempt1.get("weights_before_digest"),
                    "weights_after_digest": attempt1.get("weights_after_digest"),
                    "execution_environment": exec_env,
                    "evidence_scope": ev_scope,
                    "precision": expected_precision,
                },
                actual={
                    "weights_before_digest": step_doc.get("weights_before_digest"),
                    "weights_after_digest": step_doc.get("weights_after_digest"),
                    "attempt_pids": step_doc.get("attempt_pids"),
                    "execution_environment": step_doc.get("execution_environment"),
                    "evidence_scope": step_doc.get("evidence_scope"),
                    "precision": step_doc.get("precision"),
                },
            )

        expected_checks = {
            "anatomy_spec_sha256": spec.sha256(),
            "spec_manifest_sha256": expected_spec_manifest_sha256,
            "intake_manifest_sha256": recorded_intake_sha,
            "upstream_commit": expected_commit,
            "framework_commit": expected_framework_commit,
            "case_id": contract.case.id,
            "precision": expected_precision,
            "execution_environment": exec_env,
            "evidence_scope": ev_scope,
            "weights_before_digest": attempt1["weights_before_digest"],
            "weights_after_digest": attempt1["weights_after_digest"],
            "batch_digest": attempt1["batch_digest"],
            "comparison_status": recompared.status,
            "required_phases": list(REQUIRED_PHASE_IDS),
            "required_assertions": list(REQUIRED_ASSERTION_IDS),
            "required_camera_roles": list(spec.required_camera_roles),
            "required_rng_consumers": list(spec.required_rng_consumers),
            "observation_boundaries": expected_boundaries,
            "negative_injections": expected_negatives,
        }
        expected_checks_sha = _canonical_json_sha256(expected_checks)

        # 6. Validate handoff/result.json strictly when checking a sealed package
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
                    summary="sealed anatomy handoff must have execution_status='completed' and empty blocked_reasons",
                    evidence_path="handoff/result.json",
                    expected={"execution_status": "completed", "blocked_reasons": []},
                    actual={
                        "execution_status": handoff.execution_status,
                        "blocked_reasons": handoff.blocked_reasons,
                    },
                )
            if handoff.stage != "f2":
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="HANDOFF_STAGE_MISMATCH",
                    first_mismatch="handoff/result.json:stage",
                    summary=f"expected handoff stage 'f2', got {handoff.stage!r}",
                    evidence_path="handoff/result.json",
                    expected="f2",
                    actual=handoff.stage,
                )
            if (handoff.anatomy_spec_sha256 != spec.sha256()
                    or handoff.spec_manifest_sha256 != expected_spec_manifest_sha256):
                return self._fail(
                    commit=expected_commit, manifest_sha256=manifest_sha256, status="blocked",
                    reason_code="ANATOMY_SPEC_MISMATCH", first_mismatch="handoff.anatomy_spec_sha256",
                    summary="handoff does not bind frozen spec", evidence_path="handoff/result.json",
                    expected=spec.sha256(), actual=handoff.anatomy_spec_sha256,
                )
            if handoff.request_sha256 != case_sha:
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="REQUEST_SHA_MISMATCH",
                    first_mismatch="handoff/result.json:request_sha256",
                    summary="handoff request_sha256 does not match inputs/anatomy_case.yaml",
                    evidence_path="handoff/result.json",
                    expected=case_sha,
                    actual=handoff.request_sha256,
                )
            # P2-2: handoff.observed_scope must be strictly equal to `["train_step_replay"]`
            # (reject inflated/unexecuted scopes like `checkpoint_resume` or `task_eval`).
            if list(handoff.observed_scope) != ["train_step_replay"]:
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="HANDOFF_SCOPE_EXCEEDED"
                    if "train_step_replay" in handoff.observed_scope
                    else "SCOPE_NOT_SATISFIED",
                    first_mismatch="handoff/result.json:observed_scope",
                    summary="handoff observed_scope must equal ['train_step_replay'] exactly",
                    evidence_path="handoff/result.json",
                    expected=["train_step_replay"],
                    actual=list(handoff.observed_scope),
                )
            if handoff.validator_verdict is None:
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="HANDOFF_VERDICT_MISSING",
                    first_mismatch="handoff/result.json:validator_verdict",
                    summary="completed handoff lacks validator_verdict reference",
                    evidence_path="handoff/result.json",
                    expected="evidence/validate/verdict.json",
                    actual=None,
                )
            verdict_rel = handoff.validator_verdict.path
            verdict_ref_path = pkg_dir / verdict_rel
            if (
                verdict_rel not in present_paths
                or not verdict_ref_path.is_file()
                or _sha256_bytes(verdict_ref_path.read_bytes())
                != handoff.validator_verdict.sha256
            ):
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="error",
                    reason_code="HANDOFF_VERDICT_DIGEST_MISMATCH",
                    first_mismatch="handoff/result.json:validator_verdict",
                    summary="handoff validator_verdict file missing or digest mismatch",
                    evidence_path=verdict_rel,
                    expected=handoff.validator_verdict.sha256,
                    actual=(
                        _sha256_bytes(verdict_ref_path.read_bytes())
                        if verdict_ref_path.is_file()
                        else None
                    ),
                )
            try:
                verdict_doc = json.loads(verdict_ref_path.read_text(encoding="utf-8"))
            except Exception as error:
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="HANDOFF_VERDICT_INVALID",
                    first_mismatch=f"{verdict_rel}:parse",
                    summary=f"handoff validator_verdict is not valid JSON: {error}",
                    evidence_path=verdict_rel,
                    expected="valid pre_publish verdict JSON",
                    actual=str(error),
                )
            if (
                not isinstance(verdict_doc, dict)
                or verdict_doc.get("phase") != "pre_publish"
                or verdict_doc.get("status") != "pass"
                or verdict_doc.get("case_sha256") != case_sha
                or verdict_doc.get("intake_manifest_sha256") != recorded_intake_sha
                or verdict_doc.get("source_commit") != expected_commit
                or verdict_doc.get("framework_commit") != expected_framework_commit
                or verdict_doc.get("validator_id") != self.validator_id
                or verdict_doc.get("validator_version") != self.validator_version
                or verdict_doc.get("validator_sha256") != expected_val_sha
                or verdict_doc.get("checks") != expected_checks
                or verdict_doc.get("checks_sha256") != expected_checks_sha
            ):
                return self._fail(
                    commit=expected_commit,
                    manifest_sha256=manifest_sha256,
                    status="blocked",
                    reason_code="HANDOFF_VERDICT_BINDING_MISMATCH",
                    first_mismatch=f"{verdict_rel}:binding",
                    summary="handoff validator_verdict does not match independent pre_publish checks and bindings",
                    evidence_path=verdict_rel,
                    expected={
                        "phase": "pre_publish",
                        "status": "pass",
                        "case_sha256": case_sha,
                        "intake_manifest_sha256": recorded_intake_sha,
                        "source_commit": expected_commit,
                        "framework_commit": expected_framework_commit,
                        "validator_id": self.validator_id,
                        "validator_sha256": expected_val_sha,
                        "checks_sha256": expected_checks_sha,
                    },
                    actual=verdict_doc,
                )

        return GateResult(
            gate_id=self.gate_id,
            check="anatomy",
            status="pass",
            binding=GateBinding(
                commit=expected_commit,
                manifest_sha256=manifest_sha256,
                validator_id=self.validator_id,
                validator_sha256=expected_val_sha,
            ),
            summary="sealed anatomy package verified via IntakeChecker, independent two-subprocess replay, and cross-document checks",
            evidence=(
                GateEvidence(
                    key="intake_binding",
                    detail=f"intake_manifest_sha256={recorded_intake_sha}",
                    path="analysis/behavior_contract.yaml",
                    digest=recorded_intake_sha,
                ),
                GateEvidence(
                    key="weights_after",
                    detail=f"replay_pids={replay_pids} changed={attempt1['changed_parameters']}",
                    path="evidence/attempt-2/capture.json",
                    digest=attempt2["weights_after_digest"],
                ),
                GateEvidence(
                    key="independent_replay",
                    detail=(
                        f"replay_pids={replay_pids} "
                        f"exit_codes={[r['exit_code'] for r in replay_receipts]}"
                    ),
                    path="evidence/comparison.json",
                ),
            ),
            metrics={
                "validator_version": self.validator_version,
                "execution_environment": exec_env,
                "evidence_scope": ev_scope,
                "intake_manifest_sha256": recorded_intake_sha,
                "framework_commit": expected_framework_commit,
                "replay_pids": replay_pids,
                "replay_receipts": replay_receipts,
                "checks": expected_checks,
                "checks_sha256": expected_checks_sha,
            },
        )

    def check(self, context: CheckContext) -> GateResult:
        intake_dir_raw = context.parameters.get("intake_package_dir")
        intake_sha = context.parameters.get("intake_manifest_sha256")
        return self.validate_directory(
            package_dir=context.package_dir,
            expected_commit=context.commit,
            manifest_sha256=context.manifest_sha256,
            intake_package_dir=Path(intake_dir_raw) if intake_dir_raw else None,
            expected_intake_manifest_sha256=str(intake_sha) if intake_sha else None,
            expected_spec_manifest_sha256=context.parameters.get("spec_manifest_sha256"),
            require_sealed=True,
        )


def main(argv: list[str] | None = None) -> int:
    """Independent CLI entry for verifying a sealed Stage 2 (`anatomy`) package."""
    parser = argparse.ArgumentParser(
        description="Independent validator for sealed Stage 2 (anatomy) packages."
    )
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--expected-commit", required=True)
    parser.add_argument("--intake-dir", required=True)
    parser.add_argument("--intake-manifest-sha256", required=True)
    parser.add_argument("--spec-manifest-sha256", required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args(argv)

    checker = AnatomyChecker()
    params: dict[str, Any] = {}
    params["spec_manifest_sha256"] = args.spec_manifest_sha256
    if args.intake_dir:
        params["intake_package_dir"] = args.intake_dir
    if args.intake_manifest_sha256:
        params["intake_manifest_sha256"] = args.intake_manifest_sha256

    result = checker.check(
        CheckContext(
            commit=args.expected_commit,
            manifest_sha256=args.manifest_sha256,
            package_dir=Path(args.package_dir),
            parameters=params,
        )
    )
    payload = json.dumps(result.to_dict(), ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(payload + "\n", encoding="utf-8")
    print(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

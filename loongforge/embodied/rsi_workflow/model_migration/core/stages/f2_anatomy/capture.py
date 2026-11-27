"""Two-subprocess single-step capture driver for Stage 2 (`anatomy`).

Always verifies the sealed Stage 1 (`intake`) `PackageRef` first, then executes
`attempt-1` and `attempt-2` in two separate fresh Python child processes so each
attempt starts from an identical clean initial state with distinct OS PIDs.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, TypeAlias

import yaml

from core.artifacts import MANIFEST_NAME, ArtifactError, PackageRef, verify as verify_package
import hashlib
from core.contracts import (
    ContractParseError,
    ContractValidationError,
    load_intake_task_pack,
)
from core.stages.f1_intake.assets import triage_assets

from .adapters.fake import (
    DEFAULT_SAMPLE_IDS,
    FakeNegativeInjection,
)
from .compare import ComparisonReport, compare_attempts
from .contract import (
    AnatomySpec, PI05_SPEC,
    build_behavior_contract,
    build_observation_plan,
    build_step_anatomy,
)
from .spec import freeze_spec, load_anatomy_spec, verify_frozen_spec

CaptureStatus: TypeAlias = Literal["captured", "failed", "blocked", "not_run"]
STATE_FILENAME = "anatomy_state.json"


@dataclass(frozen=True, slots=True)
class CaptureResult:
    """Structured outcome of two-subprocess single-step capture."""

    status: CaptureStatus
    summary: str
    execution_environment: str
    evidence_scope: str
    attempt1_pid: int | None = None
    attempt2_pid: int | None = None
    comparison: ComparisonReport | None = None
    reason_codes: tuple[str, ...] = ()
    prep_dir: Path | None = None
    spec_manifest_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "spec_manifest_sha256": self.spec_manifest_sha256,
            "summary": self.summary,
            "execution_environment": self.execution_environment,
            "evidence_scope": self.evidence_scope,
            "attempt1_pid": self.attempt1_pid,
            "attempt2_pid": self.attempt2_pid,
            "comparison": self.comparison.to_dict() if self.comparison is not None else None,
            "reason_codes": list(self.reason_codes),
            "prep_dir": str(self.prep_dir) if self.prep_dir is not None else None,
        }


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
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


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _run_fake_subprocess(
    *,
    output_path: Path,
    seed: int,
    world_size: int,
    case_id: str,
    sample_ids: tuple[str, ...],
    inject: FakeNegativeInjection | None = None,
    spec_file: Path | None = None,
    package_root: Path | None = None,
) -> dict[str, Any]:
    cmd = [
        sys.executable,
        "-m",
        "core.stages.f2_anatomy.adapters.fake",
        "--output",
        str(output_path),
        "--seed",
        str(seed),
        "--world-size",
        str(world_size),
        "--case-id",
        case_id,
        "--sample-ids",
        ",".join(sample_ids),
    ]
    if inject is not None:
        cmd.extend(["--inject", inject])
    if spec_file is not None:
        cmd.extend(["--spec-file", spec_file.relative_to(package_root).as_posix() if package_root else str(spec_file)])

    env = dict(os.environ)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    existing_pypath = env.get("PYTHONPATH", "")
    repo_str = str(_repo_root())
    env["PYTHONPATH"] = (
        f"{repo_str}{os.pathsep}{existing_pypath}" if existing_pypath else repo_str
    )

    proc = subprocess.run(
        cmd,
        cwd=package_root or _repo_root(),
        env=env,
        text=True,
        capture_output=True,
        check=False,
        timeout=60,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"fake adapter subprocess failed (rc={proc.returncode}): {proc.stderr.strip()}"
        )
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if spec_file is not None:
        payload["spec_manifest_sha256"] = hashlib.sha256((spec_file.parent / MANIFEST_NAME).read_bytes()).hexdigest()
    # Receipt metadata is portable; it is never executed. Keep the actual launch
    # bound to sys.executable and its real output path above.
    receipt_cmd = list(cmd)
    receipt_cmd[0] = "python3"
    receipt_output = output_path.name
    if package_root is not None and output_path.is_relative_to(package_root):
        receipt_output = output_path.relative_to(package_root).as_posix()
    receipt_cmd[receipt_cmd.index("--output") + 1] = receipt_output
    payload["launch_receipt"] = {
        "command": receipt_cmd,
        "exit_code": proc.returncode,
        "executed_at": _utc_now(),
    }
    _write_json_atomic(output_path, payload)
    return payload


def capture_two_attempts(
    *,
    intake_package_dir: str | Path,
    intake_manifest_sha256: str,
    prep_dir: str | Path,
    case_id: str = "pi05-cpu-mock-step-1",
    seed: int = 42,
    world_size: int = 1,
    sample_ids: tuple[str, ...] = DEFAULT_SAMPLE_IDS,
    inject_attempt1: FakeNegativeInjection | None = None,
    inject_attempt2: FakeNegativeInjection | None = None,
    spec: AnatomySpec | None = None,
) -> CaptureResult:
    """Verify sealed `intake` package via `IntakeChecker` and run two independent subprocesses."""
    prep = Path(prep_dir).resolve()
    prep.mkdir(parents=True, exist_ok=True)
    intake_dir = Path(intake_package_dir).resolve()

    # 1. Load task contract & fail-closed verification via real `IntakeChecker`.
    try:
        task = load_intake_task_pack(intake_dir / "inputs" / "task.yaml")
    except (ContractParseError, ContractValidationError) as error:
        return CaptureResult(
            status="blocked",
            summary=f"intake task.yaml invalid or missing: {error}",
            execution_environment="cpu_mock",
            evidence_scope="orchestration_only",
            reason_codes=(error.reason_code,),
            prep_dir=prep,
        )

    from validate.base import CheckContext
    from validate.intake_check import IntakeChecker

    intake_checker = IntakeChecker()
    intake_gate = intake_checker.check(
        CheckContext(
            commit=task.source.commit,
            manifest_sha256=intake_manifest_sha256,
            package_dir=intake_dir,
        )
    )
    if intake_gate.status != "pass":
        return CaptureResult(
            status="blocked",
            summary=f"Stage 1 IntakeChecker failed: {intake_gate.summary}",
            execution_environment="cpu_mock",
            evidence_scope="orchestration_only",
            reason_codes=intake_gate.reason_codes or ("INTAKE_CHECK_FAILED",),
            prep_dir=prep,
        )

    # Stage 2 requires `train_step` scope enabled and all `f2_train_step` assets ready.
    if not task.scope.train_step:
        return CaptureResult(
            status="blocked",
            summary="intake task.scope.train_step is false; Stage 2 anatomy requires train_step=true",
            execution_environment="cpu_mock",
            evidence_scope="orchestration_only",
            reason_codes=("INTAKE_TRAIN_STEP_SCOPE_DISABLED",),
            prep_dir=prep,
        )

    f2_assets = triage_assets(task.assets, required_scope="f2_train_step")
    if f2_assets.overall_status != "ready":
        reasons = f2_assets.blocked_reasons or f2_assets.waiting_reasons or ("INTAKE_ASSET_NOT_READY",)
        return CaptureResult(
            status="blocked",
            summary=f"required f2_train_step assets are not ready (status={f2_assets.overall_status})",
            execution_environment="cpu_mock",
            evidence_scope="orchestration_only",
            reason_codes=reasons,
            prep_dir=prep,
        )

    # Freeze independently of the observed attempts and refuse policy drift on re-capture.
    try:
        source_ref = None
        if task.schema_version == 2:
            from core.contracts.common import load_yaml_document
            intake_spec_doc = load_yaml_document(intake_dir / "analysis/anatomy_spec.yaml")
            intake_spec = load_anatomy_spec(intake_dir / "analysis/anatomy_spec.yaml")
            source_ref = intake_spec_doc["source_ref"]
            if spec is not None and spec != intake_spec:
                raise ValueError("requested AnatomySpec differs from frozen v2 Intake handoff")
            spec = intake_spec
        spec = spec or PI05_SPEC
        spec_file, _, spec_manifest_sha = freeze_spec(
            prep, spec, intake_manifest_sha256=intake_manifest_sha256,
            upstream_commit=task.source.commit, source_ref=source_ref,
        )
    except (ValueError, OSError, ArtifactError) as error:
        return CaptureResult(
            status="blocked", summary=str(error), execution_environment="cpu_mock",
            evidence_scope="orchestration_only", reason_codes=("ANATOMY_SPEC_INVALID",), prep_dir=prep,
        )
    # 2. Execute attempt-1 and attempt-2 in two separate OS subprocesses.
    work_dir = prep / "work"
    a1_path = work_dir / "evidence" / "attempt-1" / "capture.json"
    a2_path = work_dir / "evidence" / "attempt-2" / "capture.json"

    attempt1 = _run_fake_subprocess(
        output_path=a1_path,
        seed=seed,
        world_size=world_size,
        case_id=case_id,
        sample_ids=sample_ids,
        inject=inject_attempt1,
        spec_file=spec_file,
        package_root=work_dir,
    )
    attempt2 = _run_fake_subprocess(
        output_path=a2_path,
        seed=seed,
        world_size=world_size,
        case_id=case_id,
        sample_ids=sample_ids,
        inject=inject_attempt2,
        spec_file=spec_file,
        package_root=work_dir,
    )

    if attempt1["pid"] == attempt2["pid"]:
        return CaptureResult(
            status="blocked",
            summary="attempt-1 and attempt-2 did not execute in distinct processes",
            execution_environment=attempt1["execution_environment"],
            evidence_scope=attempt1["evidence_scope"],
            attempt1_pid=attempt1["pid"],
            attempt2_pid=attempt2["pid"],
            reason_codes=("SAME_PROCESS_ATTEMPTS",),
            prep_dir=prep,
        )

    comparison = compare_attempts(
        attempt1,
        attempt2,
        evidence_path="evidence/attempt-2/capture.json",
        spec=spec,
    )
    _write_json_atomic(work_dir / "evidence" / "comparison.json", comparison.to_dict())

    # Obtain real composite AnatomyChecker validator metadata.
    from validate.anatomy_check import AnatomyChecker

    checker = AnatomyChecker()
    validator_sha = checker.validator_sha256()

    # 3. Emit case input + step_anatomy.yaml + behavior_contract.yaml + observation_plan.yaml.
    case_input = {
        "schema_version": 1,
        "anatomy_spec_sha256": spec.sha256(),
        "spec_manifest_sha256": spec_manifest_sha,
        "case_id": case_id,
        "intake_package_dir": str(intake_dir),
        "intake_manifest_sha256": intake_manifest_sha256,
        "upstream_commit": task.source.commit,
        "framework_commit": task.target.commit,
        "model_variant": task.source.model_variant,
        "seed": seed,
        "world_size": world_size,
        "sample_ids": list(sample_ids),
        "precision": dict(attempt1["precision"]),
        "execution_environment": attempt1["execution_environment"],
        "evidence_scope": attempt1["evidence_scope"],
    }
    _write_yaml(work_dir / "inputs" / "anatomy_case.yaml", case_input)

    step_anatomy = build_step_anatomy(
        case_id=case_id,
        intake_manifest_sha256=intake_manifest_sha256,
        upstream_commit=task.source.commit,
        framework_commit=task.target.commit,
        attempt1=attempt1,
        attempt2=attempt2,
        spec=spec,
    )
    _write_yaml(work_dir / "analysis" / "step_anatomy.yaml", step_anatomy)

    behavior_contract = build_behavior_contract(
        case_id=case_id,
        intake_manifest_sha256=intake_manifest_sha256,
        upstream_commit=task.source.commit,
        framework_commit=task.target.commit,
        attempt1=attempt1,
        validator_id=checker.validator_id,
        validator_version=checker.validator_version,
        validator_sha256=validator_sha,
        blocked_reasons=list(comparison.reason_codes),
        spec=spec,
    )
    _write_yaml(
        work_dir / "analysis" / "behavior_contract.yaml",
        behavior_contract.model_dump(mode="json"),
    )

    obs_plan = build_observation_plan(
        case_id=case_id,
        intake_manifest_sha256=intake_manifest_sha256,
        attempt1=attempt1,
        validator_id=checker.validator_id,
        validator_sha256=validator_sha,
        spec=spec,
    )
    _write_yaml(work_dir / "analysis" / "observation_plan.yaml", obs_plan)

    capture_status: CaptureStatus = "captured" if comparison.status == "pass" else "failed"
    state = {
        "schema_version": 1,
        "status": capture_status,
        "case_id": case_id,
        "intake_package_dir": str(intake_dir),
        "intake_manifest_sha256": intake_manifest_sha256,
        "upstream_commit": task.source.commit,
        "framework_commit": task.target.commit,
        "execution_environment": attempt1["execution_environment"],
        "evidence_scope": attempt1["evidence_scope"],
        "attempt1_pid": attempt1["pid"],
        "attempt2_pid": attempt2["pid"],
        "spec_manifest_sha256": spec_manifest_sha,
        "comparison_status": comparison.status,
        "reason_codes": list(comparison.reason_codes),
        "first_mismatch": comparison.first_mismatch,
        "updated_at": _utc_now(),
    }
    _write_json_atomic(prep / STATE_FILENAME, state)

    return CaptureResult(
        status=capture_status,
        summary=comparison.summary,
        spec_manifest_sha256=spec_manifest_sha,
        execution_environment=attempt1["execution_environment"],
        evidence_scope=attempt1["evidence_scope"],
        attempt1_pid=attempt1["pid"],
        attempt2_pid=attempt2["pid"],
        comparison=comparison,
        reason_codes=comparison.reason_codes,
        prep_dir=prep,
    )


def replay_sealed_anatomy(
    *,
    package_dir: str | Path,
    manifest_sha256: str,
    spec_manifest_sha256: str,
    inject: FakeNegativeInjection | None = None,
) -> ComparisonReport:
    """Verify a sealed anatomy package and replay a fresh subprocess attempt against attempt-1."""
    pkg = Path(package_dir).resolve()
    verified = verify_package(
        PackageRef(package_dir=pkg, manifest_sha256=manifest_sha256)
    )
    baseline = json.loads(verified.read_text("evidence/attempt-1/capture.json"))
    case_doc = yaml.safe_load(verified.read_text("inputs/anatomy_case.yaml"))
    spec, spec_file = verify_frozen_spec(
        pkg, intake_manifest_sha256=case_doc["intake_manifest_sha256"],
        upstream_commit=case_doc["upstream_commit"], trusted_spec_manifest_sha256=spec_manifest_sha256,
    )

    with tempfile.TemporaryDirectory(prefix="anatomy-replay-") as tmp_dir:
        replay_out = Path(tmp_dir) / "replay_capture.json"
        replayed = _run_fake_subprocess(
            output_path=replay_out,
            seed=int(case_doc["seed"]),
            world_size=int(case_doc["world_size"]),
            case_id=str(case_doc["case_id"]),
            sample_ids=tuple(case_doc["sample_ids"]),
            inject=inject,
            spec_file=spec_file, package_root=pkg,
        )
    return compare_attempts(
        baseline,
        replayed,
        evidence_path="replay/capture.json",
        spec=spec,
    )

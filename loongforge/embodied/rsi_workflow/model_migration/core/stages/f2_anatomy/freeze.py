"""Atomic staging, validation, and release sealing for Stage 2 (`anatomy`)."""

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

from core.artifacts import PackageRef, publish
from core.contracts import HandoffResult

from .capture import STATE_FILENAME
from .spec import SPEC_RELEASE_REL

AnatomyFreezeStatus: TypeAlias = Literal["completed", "blocked", "error"]


@dataclass(frozen=True, slots=True)
class AnatomyFreezeResult:
    """Structured result from attempting to seal an anatomy package."""

    status: AnatomyFreezeStatus
    summary: str
    package_ref: PackageRef | None = None
    release_record_path: Path | None = None
    verdict: dict[str, Any] | None = None
    reason_codes: tuple[str, ...] = ()
    execution_environment: str = "cpu_mock"
    evidence_scope: str = "orchestration_only"
    spec_manifest_sha256: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "spec_manifest_sha256": self.spec_manifest_sha256,
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
            "execution_environment": self.execution_environment,
            "evidence_scope": self.evidence_scope,
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


def freeze_anatomy(
    prep_dir: str | Path,
    *,
    release_name: str | None = None,
    attempt_id: str = "anatomy-attempt-01",
) -> AnatomyFreezeResult:
    """Stage, independently validate via `AnatomyChecker`, and atomically publish."""
    prep = Path(prep_dir).resolve()
    state_path = prep / STATE_FILENAME
    if not state_path.is_file():
        return AnatomyFreezeResult(
            status="error",
            summary=f"anatomy capture state not found: {state_path}",
            reason_codes=("ANATOMY_NOT_CAPTURED",),
        )
    state = json.loads(state_path.read_text(encoding="utf-8"))
    exec_env = str(state.get("execution_environment", "cpu_mock"))
    ev_scope = str(state.get("evidence_scope", "orchestration_only"))

    if state.get("status") not in {"captured", "frozen"}:
        reasons = tuple(state.get("reason_codes", ())) or ("CAPTURE_NOT_PASSED",)
        return AnatomyFreezeResult(
            status="blocked",
            summary="cannot freeze anatomy because two-attempt capture did not pass",
            reason_codes=reasons,
            execution_environment=exec_env,
            evidence_scope=ev_scope,
        )

    work_dir = prep / "work"
    case_path = work_dir / "inputs" / "anatomy_case.yaml"
    if not case_path.is_file():
        return AnatomyFreezeResult(
            status="error",
            summary="missing work/inputs/anatomy_case.yaml",
            reason_codes=("ANATOMY_CASE_MISSING",),
            execution_environment=exec_env,
            evidence_scope=ev_scope,
        )

    request_sha = _sha256_bytes(case_path.read_bytes())
    upstream_commit = str(state["upstream_commit"])
    intake_dir = Path(str(state["intake_package_dir"]))
    intake_sha = str(state["intake_manifest_sha256"])
    spec_manifest_sha = state.get("spec_manifest_sha256")

    packages_dir = prep / "packages"
    packages_dir.mkdir(parents=True, exist_ok=True)
    rel_name = release_name or f"anatomy-{state['case_id']}"
    staging_dir = packages_dir / f"{rel_name}.staging"
    if staging_dir.exists():
        shutil.rmtree(staging_dir)

    try:
        (staging_dir / "inputs").mkdir(parents=True)
        (staging_dir / "analysis").mkdir(parents=True)
        (staging_dir / "evidence" / "attempt-1").mkdir(parents=True)
        (staging_dir / "evidence" / "attempt-2").mkdir(parents=True)
        (staging_dir / "evidence" / "validate").mkdir(parents=True)
        (staging_dir / "handoff").mkdir(parents=True)

        shutil.copytree(work_dir / SPEC_RELEASE_REL, staging_dir / SPEC_RELEASE_REL)
        shutil.copy2(case_path, staging_dir / "inputs" / "anatomy_case.yaml")
        for fname in (
            "step_anatomy.yaml",
            "behavior_contract.yaml",
            "observation_plan.yaml",
            "anatomy_spec.yaml",
            "spec_lock.json",
        ):
            shutil.copy2(work_dir / "analysis" / fname, staging_dir / "analysis" / fname)

        shutil.copy2(
            work_dir / "evidence" / "attempt-1" / "capture.json",
            staging_dir / "evidence" / "attempt-1" / "capture.json",
        )
        shutil.copy2(
            work_dir / "evidence" / "attempt-2" / "capture.json",
            staging_dir / "evidence" / "attempt-2" / "capture.json",
        )
        shutil.copy2(
            work_dir / "evidence" / "comparison.json",
            staging_dir / "evidence" / "comparison.json",
        )

        from validate.anatomy_check import AnatomyChecker

        checker = AnatomyChecker()
        pre_verdict_rel = "evidence/validate/pre_publish_verdict.json"
        pre_verdict_doc, pre_gate = checker.build_pre_publish_verdict(
            staging_dir=staging_dir,
            expected_commit=upstream_commit,
            case_sha256=request_sha,
            intake_package_dir=intake_dir,
            intake_manifest_sha256=intake_sha,
            spec_manifest_sha256=spec_manifest_sha,
        )
        if pre_gate.status != "pass" or pre_verdict_doc is None:
            shutil.rmtree(staging_dir, ignore_errors=True)
            return AnatomyFreezeResult(
                status="blocked" if pre_gate.status == "blocked" else "error",
                summary=pre_gate.summary,
                verdict=pre_gate.to_dict(),
                reason_codes=pre_gate.reason_codes,
                execution_environment=exec_env,
                evidence_scope=ev_scope,
            )
        pre_verdict_sha = _write_json_atomic(
            staging_dir / pre_verdict_rel,
            pre_verdict_doc,
        )

        handoff = HandoffResult(
            schema_version=1,
            preparation_id=f"prep-{state['case_id']}",
            request_id=f"req-{request_sha[:12]}",
            stage="f2",
            attempt_id=attempt_id,
            request_sha256=request_sha,
            execution_status="completed",
            observed_scope=["train_step_replay"],
            validator_verdict={
                "path": pre_verdict_rel,
                "sha256": pre_verdict_sha,
            },
            unknowns=[],
            blocked_reasons=[],
            anatomy_spec_sha256=pre_gate.metrics["checks"]["anatomy_spec_sha256"],
            spec_manifest_sha256=spec_manifest_sha,
        )
        _write_json_atomic(
            staging_dir / "handoff" / "result.json",
            handoff.model_dump(mode="json"),
        )

        # Full pre-publish check with all required files + handoff present.
        full_pre_check = checker.validate_directory(
            package_dir=staging_dir,
            expected_commit=upstream_commit,
            manifest_sha256=request_sha,
            intake_package_dir=intake_dir,
            expected_intake_manifest_sha256=intake_sha,
            expected_spec_manifest_sha256=spec_manifest_sha,
            require_sealed=False,
            require_handoff=True,
        )
        if full_pre_check.status != "pass":
            shutil.rmtree(staging_dir, ignore_errors=True)
            return AnatomyFreezeResult(
                status="blocked" if full_pre_check.status == "blocked" else "error",
                summary=full_pre_check.summary,
                verdict=full_pre_check.to_dict(),
                reason_codes=full_pre_check.reason_codes,
                execution_environment=exec_env,
                evidence_scope=ev_scope,
            )

        # Atomically seal and publish the package via core/artifacts.py:publish().
        package_ref = publish(staging_dir)

        # Independent post-publish verification bound to the real sealed manifest SHA-256.
        sealed_verdict = checker.validate_directory(
            package_dir=package_ref.package_dir,
            expected_commit=upstream_commit,
            manifest_sha256=package_ref.manifest_sha256,
            intake_package_dir=intake_dir,
            expected_intake_manifest_sha256=intake_sha,
            expected_spec_manifest_sha256=spec_manifest_sha,
            require_sealed=True,
        )
        if sealed_verdict.status != "pass":
            return AnatomyFreezeResult(
                status="error",
                summary=f"post-publish verification failed: {sealed_verdict.summary}",
                verdict=sealed_verdict.to_dict(),
                reason_codes=sealed_verdict.reason_codes,
                execution_environment=exec_env,
                evidence_scope=ev_scope,
            )

        release_record_path = prep / "release.json"
        release_record = {
            "schema_version": 1,
            "stage": "anatomy",
            "case_id": state["case_id"],
            "upstream_commit": upstream_commit,
            "intake_manifest_sha256": intake_sha,
            "spec_manifest_sha256": spec_manifest_sha,
            "package_dir": str(package_ref.package_dir),
            "manifest_sha256": package_ref.manifest_sha256,
            "execution_environment": exec_env,
            "evidence_scope": ev_scope,
            "validator_verdict": sealed_verdict.to_dict(),
            "published_at": _utc_now(),
        }
        _write_json_atomic(release_record_path, release_record)

        state["status"] = "frozen"
        state["package_dir"] = str(package_ref.package_dir)
        state["manifest_sha256"] = package_ref.manifest_sha256
        state["updated_at"] = _utc_now()
        _write_json_atomic(state_path, state)

        return AnatomyFreezeResult(
            status="completed",
            spec_manifest_sha256=spec_manifest_sha,
            summary="anatomy package validated and atomically sealed",
            package_ref=package_ref,
            release_record_path=release_record_path,
            verdict=sealed_verdict.to_dict(),
            reason_codes=(),
            execution_environment=exec_env,
            evidence_scope=ev_scope,
        )
    except Exception as error:
        shutil.rmtree(staging_dir, ignore_errors=True)
        return AnatomyFreezeResult(
            status="error",
            summary=f"anatomy freeze failed: {type(error).__name__}: {error}",
            reason_codes=("FREEZE_EXCEPTION",),
            execution_environment=exec_env,
            evidence_scope=ev_scope,
        )

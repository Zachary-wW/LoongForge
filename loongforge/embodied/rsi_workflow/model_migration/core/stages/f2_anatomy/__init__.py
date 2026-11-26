"""Stage 2 (`anatomy`): two-attempt single-step capture, observation plan, and behavior contract."""

from __future__ import annotations

from pathlib import Path

from core.stages.base import StageContext, StageResult

from .adapters import (
    FAKE_NEGATIVE_INJECTIONS,
    FakeNegativeInjection,
    FakeStepAdapter,
    FakeStepConfig,
    OpenPIJaxAdapter,
    OpenPIJaxConfig,
    OpenPIPreflightResult,
)
from .capture import CaptureResult, capture_two_attempts, replay_sealed_anatomy
from .compare import ComparisonReport, compare_attempts, diff_weight_snapshots
from .contract import (
    build_behavior_contract,
    build_observation_plan,
    build_step_anatomy,
)
from .freeze import AnatomyFreezeResult, freeze_anatomy


class AnatomyStageRunner:
    """StageRunner implementation executing the offline Stage 2 (`anatomy`) loop."""

    stage_name = "anatomy"

    def run(self, context: StageContext) -> StageResult:
        intake_dir = context.upstream_artifacts.get("intake") or context.parameters.get(
            "intake_package_dir"
        )
        intake_sha = context.parameters.get("intake_manifest_sha256")
        if not intake_dir or not intake_sha:
            raise ValueError(
                "AnatomyStageRunner requires upstream_artifacts['intake'] "
                "(or parameters['intake_package_dir']) and parameters['intake_manifest_sha256']"
            )

        prep_dir = Path(context.output_dir)
        cap = capture_two_attempts(
            intake_package_dir=Path(intake_dir),
            intake_manifest_sha256=str(intake_sha),
            prep_dir=prep_dir,
            case_id=str(context.parameters.get("case_id", f"{context.run_id}-step-1")),
            seed=int(context.parameters.get("seed", 42)),
            world_size=int(context.parameters.get("world_size", 1)),
            spec=context.parameters.get("anatomy_spec"),
        )
        if cap.status != "captured":
            return StageResult(
                stage="anatomy",
                status="blocked",
                summary=cap.summary,
                reason_codes=cap.reason_codes or ("ANATOMY_CAPTURE_FAILED",),
                metadata={
                    "execution_environment": cap.execution_environment,
                    "evidence_scope": cap.evidence_scope,
                },
            )

        freeze_res = freeze_anatomy(
            prep_dir,
            release_name=context.parameters.get("release_name"),
            attempt_id=str(context.parameters.get("attempt_id", "anatomy-attempt-01")),
        )
        return StageResult(
            stage="anatomy",
            status=freeze_res.status,
            summary=freeze_res.summary,
            artifact_dir=freeze_res.package_ref.package_dir
            if freeze_res.package_ref is not None
            else None,
            manifest_sha256=freeze_res.package_ref.manifest_sha256
            if freeze_res.package_ref is not None
            else None,
            reason_codes=freeze_res.reason_codes,
            metadata={
                "spec_manifest_sha256": freeze_res.spec_manifest_sha256,
                "release_record_path": str(freeze_res.release_record_path)
                if freeze_res.release_record_path is not None
                else None,
                "execution_environment": freeze_res.execution_environment,
                "evidence_scope": freeze_res.evidence_scope,
            },
        )


__all__ = [
    "AnatomyFreezeResult",
    "AnatomyStageRunner",
    "CaptureResult",
    "ComparisonReport",
    "FAKE_NEGATIVE_INJECTIONS",
    "FakeNegativeInjection",
    "FakeStepAdapter",
    "FakeStepConfig",
    "OpenPIJaxAdapter",
    "OpenPIJaxConfig",
    "OpenPIPreflightResult",
    "build_behavior_contract",
    "build_observation_plan",
    "build_step_anatomy",
    "capture_two_attempts",
    "compare_attempts",
    "diff_weight_snapshots",
    "freeze_anatomy",
    "replay_sealed_anatomy",
]

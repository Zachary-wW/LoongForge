"""Stage 3 (`scaffold`): LoongForge 6 extension-point mapping and config/code skeleton generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

from core.stages.base import StageContext, StageResult

LoongForgeExtensionPoint: TypeAlias = Literal[
    "model_registry",
    "data_and_transforms",
    "config_map",
    "trainer_builder",
    "eval_adapter",
    "regression_runner",
]

LOONGFORGE_EXTENSION_POINTS: tuple[LoongForgeExtensionPoint, ...] = (
    "model_registry",
    "data_and_transforms",
    "config_map",
    "trainer_builder",
    "eval_adapter",
    "regression_runner",
)


@dataclass(frozen=True, slots=True)
class ExtensionPointBinding:
    """Maps one LoongForge extension point to concrete target paths and symbols."""

    extension_point: LoongForgeExtensionPoint
    registry_path: str
    target_files: tuple[str, ...] = ()
    reuse_components: tuple[str, ...] = ()
    notes: str = ""
    side: str = ""
    slot_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class ScaffoldPlan:
    """Declarative integration plan targeting `baidu-baige/LoongForge`."""

    model_name: str
    target_framework_commit: str
    intake_manifest_sha256: str
    anatomy_manifest_sha256: str
    bindings: tuple[ExtensionPointBinding, ...] = ()
    recommended_skills: tuple[str, ...] = (
        "model-adaptation",
        "precision-alignment",
        "vla-model-eval-adapter",
    )
    metadata: dict[str, str] = field(default_factory=dict)
    anatomy_reference_status: Literal["verified", "unverified_external"] = "unverified_external"

    @classmethod
    def from_intake(
        cls, *, intake_package: Path, intake_manifest_sha256: str,
        anatomy_manifest_sha256: str, source_commit: str,
        anatomy_package_dir: Path | None = None, spec_manifest_sha256: str | None = None,
    ) -> ScaffoldPlan:
        """Consume only independently accepted v2 mappings; no hardcoded slot IDs."""
        from core.artifacts import PackageRef, verify
        from core.contracts import IntakeTaskPackV2, ModelAnalysisReport, load_intake_task_pack
        from core.contracts.common import load_yaml_document
        from validate.base import CheckContext
        from validate.intake_check import IntakeChecker
        from validate.anatomy_check import AnatomyChecker
        from pydantic import TypeAdapter
        from core.contracts.common import Sha256
        TypeAdapter(Sha256).validate_python(anatomy_manifest_sha256)
        gate = IntakeChecker().check(CheckContext(
            commit=source_commit, manifest_sha256=intake_manifest_sha256, package_dir=intake_package,
        ))
        if gate.status != "pass":
            raise ValueError(f"ScaffoldPlan requires accepted intake: {gate.reason_codes}")
        verified = verify(PackageRef(package_dir=intake_package, manifest_sha256=intake_manifest_sha256))
        task = load_intake_task_pack(intake_package / "inputs/task.yaml")
        if not isinstance(task, IntakeTaskPackV2):
            raise ValueError("v1 intake has no soft slot mapping")
        reference_status = "unverified_external"
        if anatomy_package_dir is not None:
            anatomy_gate = AnatomyChecker().check(CheckContext(
                commit=source_commit, manifest_sha256=anatomy_manifest_sha256, package_dir=anatomy_package_dir,
                parameters={"intake_package_dir": str(intake_package), "intake_manifest_sha256": intake_manifest_sha256,
                            "spec_manifest_sha256": spec_manifest_sha256},
            ))
            if anatomy_gate.status != "pass":
                raise ValueError(f"ScaffoldPlan requires accepted anatomy: {anatomy_gate.reason_codes}")
            anatomy_manifest_sha256 = anatomy_gate.binding.manifest_sha256
            reference_status = "verified"
        report = ModelAnalysisReport.model_validate(
            load_yaml_document(intake_package / "analysis/model_analysis.json"),
        )
        bindings = []
        for extension in task.slot_catalog.extensions:
            slots = [x for x in report.slots if x.extension_point == extension.extension_point]
            bindings.append(ExtensionPointBinding(
                extension_point=extension.extension_point, registry_path=extension.registry_path,
                target_files=tuple(dict.fromkeys(path for slot in slots for path in slot.target_files)),
                reuse_components=tuple(extension.reuse_components), side=extension.side,
                slot_ids=tuple(x.slot_id for x in slots),
                notes="; ".join(x.reason for x in slots if x.reason),
            ))
        return cls(
            model_name=task.source.model_variant, target_framework_commit=task.target.commit,
            intake_manifest_sha256=verified.ref.manifest_sha256,
            anatomy_manifest_sha256=anatomy_manifest_sha256, bindings=tuple(bindings),
            anatomy_reference_status=reference_status,
            metadata={"compatibility_profile": "internal", "handoff_boundary": "handoff_only"},
        )


class ScaffoldStageRunner:
    """Placeholder runner for Stage 3 (`scaffold`)."""

    stage_name = "scaffold"

    def plan(
        self,
        *,
        model_name: str,
        target_repo: Path,
        intake_package: Path,
        anatomy_package: Path,
    ) -> ScaffoldPlan:
        raise NotImplementedError(
            "Stage 3 (`scaffold`) planning is not implemented in this milestone"
        )

    def run(self, context: StageContext) -> StageResult:
        raise NotImplementedError(
            "Stage 3 (`scaffold`) is not implemented in this milestone"
        )


__all__ = [
    "ExtensionPointBinding",
    "LOONGFORGE_EXTENSION_POINTS",
    "LoongForgeExtensionPoint",
    "ScaffoldPlan",
    "ScaffoldStageRunner",
]

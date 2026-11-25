"""Stage 3 (`scaffold`): LoongForge 6 extension-point mapping and config/code skeleton generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, TypeAlias

from stages.base import StageContext, StageResult

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

"""Stage 6 (`deliver`): clean-worktree re-verification, report assembly, and PR package."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from core.orchestrator.judger import GateDecision
from core.stages.base import StageContext, StageResult


@dataclass(frozen=True, slots=True)
class DeliveryBundle:
    """Structured deliverable produced only after clean-worktree re-verification."""

    model_name: str
    candidate_commit: str
    target_framework_commit: str
    gate_decision: GateDecision
    validation_report_path: Path
    evidence_index: tuple[str, ...] = ()
    ci_trigger_command: str = "/ok-to-test --suite embodied"
    pr_title: str = ""
    pr_body: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


class DeliverStageRunner:
    """Placeholder runner for Stage 6 (`deliver`)."""

    stage_name = "deliver"

    def assemble(
        self,
        *,
        model_name: str,
        clean_worktree: Path,
        gate_decision: GateDecision,
        output_dir: Path,
    ) -> DeliveryBundle:
        raise NotImplementedError(
            "Stage 6 (`deliver`) assembly is not implemented in this milestone"
        )

    def run(self, context: StageContext) -> StageResult:
        raise NotImplementedError(
            "Stage 6 (`deliver`) is not implemented in this milestone"
        )


__all__ = [
    "DeliverStageRunner",
    "DeliveryBundle",
]

"""Outer-loop RSI supervisor data contracts and placeholder interface.

Records cross-task run trajectories, failure clusters, and equal-budget
holdout regression proposals without mutating the currently executing run.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from core.orchestrator.judger import GateOutcome

ProposalTarget: TypeAlias = Literal[
    "skill_playbook",
    "deterministic_tool",
    "validator_rule",
    "profile_config",
]


@dataclass(frozen=True, slots=True)
class FailureTrace:
    """Normalized failure signature captured from an independent gate verdict."""

    stage: str
    gate_id: str
    reason_codes: tuple[str, ...]
    first_mismatch: str | None = None
    summary: str = ""


@dataclass(frozen=True, slots=True)
class RunRecord:
    """Immutable cross-task run record fed into the outer RSI loop."""

    run_id: str
    model_name: str
    upstream_commit: str
    framework_commit: str
    profile_version: str
    final_status: GateOutcome
    failure_traces: tuple[FailureTrace, ...] = ()
    skill_versions: dict[str, str] = field(default_factory=dict)
    metrics: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ImprovementProposal:
    """Candidate upgrade proposed by RSI for holdout-model evaluation."""

    proposal_id: str
    target: ProposalTarget
    rationale: str
    source_run_ids: tuple[str, ...]
    diff_summary: str
    holdout_models: tuple[str, ...] = ()


@runtime_checkable
class RSISupervisor(Protocol):
    """Outer-loop evolution protocol (never mutates an in-flight adaptation run)."""

    def ingest(self, record: RunRecord) -> None:
        """Record a completed run trajectory for offline clustering."""
        ...

    def propose(self, records: tuple[RunRecord, ...]) -> list[ImprovementProposal]:
        """Propose bounded Skill/checker/profile improvements from failure clusters."""
        ...

    def evaluate_holdout(self, proposal: ImprovementProposal) -> bool:
        """Validate a proposal under equal budget on held-out embodied models."""
        ...


class PlaceholderRSISupervisor:
    """Placeholder RSI supervisor — scheduled after single-model adaptation closes."""

    def ingest(self, record: RunRecord) -> None:
        raise NotImplementedError(
            "RSI outer-loop ingestion is not implemented in this milestone"
        )

    def propose(self, records: tuple[RunRecord, ...]) -> list[ImprovementProposal]:
        raise NotImplementedError(
            "RSI outer-loop proposal generation is not implemented in this milestone"
        )

    def evaluate_holdout(self, proposal: ImprovementProposal) -> bool:
        raise NotImplementedError(
            "RSI holdout regression evaluation is not implemented in this milestone"
        )


__all__ = [
    "FailureTrace",
    "ImprovementProposal",
    "PlaceholderRSISupervisor",
    "ProposalTarget",
    "RSISupervisor",
    "RunRecord",
]

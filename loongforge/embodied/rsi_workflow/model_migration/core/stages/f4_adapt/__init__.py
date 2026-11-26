"""Stage 4 (`adapt`): Typed Handoff execution for Coding Agents + embodied Skills."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, TypeAlias, runtime_checkable

from core.stages.base import StageContext, StageResult

EmbodiedSkillName: TypeAlias = Literal[
    "model-adaptation",
    "precision-alignment",
    "vla-model-eval-adapter",
]


@dataclass(frozen=True, slots=True)
class SkillBinding:
    """Pinned reference to a domain skill playbook (read by Coding Agent, never imported)."""

    name: EmbodiedSkillName
    skill_path: str
    repo_commit: str
    role: str


@dataclass(frozen=True, slots=True)
class AdaptHandoffRequest:
    """Typed handoff dispatched from engine to a Coding Agent worker in an isolated worktree."""

    request_id: str
    model_name: str
    read_only_reference_dir: Path
    candidate_worktree_dir: Path
    intake_manifest_sha256: str
    anatomy_manifest_sha256: str
    allowed_write_globs: tuple[str, ...]
    skills: tuple[SkillBinding, ...]
    repair_feedback: str | None = None
    budget_turns: int = 30


@dataclass(frozen=True, slots=True)
class AdaptReceipt:
    """Receipt emitted after an agent session finishes (subject to independent Stage 5 verify)."""

    request_id: str
    candidate_commit: str
    modified_files: tuple[str, ...] = ()
    summary: str = ""
    metadata: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class CodingAgentAdapter(Protocol):
    """Protocol for invoking an external Coding Agent CLI (e.g. Codex / Claude Code)."""

    backend_name: str

    def execute(self, request: AdaptHandoffRequest) -> AdaptReceipt:
        """Run the coding agent inside `request.candidate_worktree_dir`."""
        ...


class AdaptStageRunner:
    """Placeholder runner for Stage 4 (`adapt`)."""

    stage_name = "adapt"

    def __init__(self, agent: CodingAgentAdapter | None = None) -> None:
        self.agent = agent

    def run(self, context: StageContext) -> StageResult:
        raise NotImplementedError(
            "Stage 4 (`adapt`) is not implemented in this milestone"
        )


__all__ = [
    "AdaptHandoffRequest",
    "AdaptReceipt",
    "AdaptStageRunner",
    "CodingAgentAdapter",
    "EmbodiedSkillName",
    "SkillBinding",
]

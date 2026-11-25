"""Unified stage execution protocol (`StageRunner`, `StageContext`, `StageResult`)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, Protocol, TypeAlias, runtime_checkable

from core.contracts.stage import StageName

StageStatus: TypeAlias = Literal[
    "completed",
    "waiting",
    "repairable",
    "blocked",
    "error",
]


@dataclass(frozen=True, slots=True)
class StageContext:
    """Immutable runtime context supplied by the orchestration engine to a stage."""

    run_id: str
    stage: StageName
    workspace_dir: Path
    output_dir: Path
    profile_path: Path | None = None
    upstream_artifacts: dict[str, Path] = field(default_factory=dict)
    read_only_paths: tuple[Path, ...] = ()
    parameters: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class StageResult:
    """Structured return payload from any stage execution."""

    stage: StageName
    status: StageStatus
    summary: str
    artifact_dir: Path | None = None
    manifest_sha256: str | None = None
    reason_codes: tuple[str, ...] = ()
    metrics: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.status in {"blocked", "error"} and not self.reason_codes:
            raise ValueError(
                f"stage result with status {self.status!r} requires at least one reason_code"
            )


@runtime_checkable
class StageRunner(Protocol):
    """Uniform contract implemented by every pipeline stage runner."""

    stage_name: StageName

    def run(self, context: StageContext) -> StageResult:
        """Execute the stage within `context` and return a structured `StageResult`."""
        ...

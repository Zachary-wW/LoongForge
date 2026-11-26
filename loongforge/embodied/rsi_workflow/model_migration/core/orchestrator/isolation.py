"""Workspace isolation protocol and default providers.

Provides the `IsolationProvider` lifecycle (`acquire / prepare / release / reclaim`)
for separating read-only reference sources from mutable candidate worktrees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


@dataclass(frozen=True, slots=True)
class IsolationLease:
    """Handle representing an acquired execution workspace for a DAG node."""

    lease_id: str
    node_id: str
    workspace_dir: Path
    read_only_paths: tuple[Path, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class IsolationProvider(Protocol):
    """Lifecycle contract for isolating node execution workspaces."""

    def acquire(self, node_id: str, *, run_id: str) -> IsolationLease:
        """Allocate an isolated workspace handle for a node."""
        ...

    def prepare(self, lease: IsolationLease) -> None:
        """Materialize files/worktree state before node execution begins."""
        ...

    def release(self, lease: IsolationLease) -> None:
        """Cleanly release workspace resources after node completion."""
        ...

    def reclaim(self, run_id: str) -> list[str]:
        """Reclaim leaked or interrupted leases from an earlier crashed run."""
        ...


class NoOpIsolation:
    """In-place no-op isolation provider suitable for pure CPU unit tests."""

    def __init__(self, base_dir: str | Path | None = None) -> None:
        self._base_dir = Path(base_dir) if base_dir is not None else Path.cwd()
        self._active: dict[str, IsolationLease] = {}

    @property
    def active_leases(self) -> dict[str, IsolationLease]:
        return dict(self._active)

    def acquire(self, node_id: str, *, run_id: str) -> IsolationLease:
        lease_id = f"{run_id}:{node_id}"
        lease = IsolationLease(
            lease_id=lease_id,
            node_id=node_id,
            workspace_dir=self._base_dir,
            metadata={"provider": "noop"},
        )
        self._active[lease_id] = lease
        return lease

    def prepare(self, lease: IsolationLease) -> None:
        return None

    def release(self, lease: IsolationLease) -> None:
        self._active.pop(lease.lease_id, None)

    def reclaim(self, run_id: str) -> list[str]:
        prefix = f"{run_id}:"
        reclaimed = [
            lease_id for lease_id in list(self._active) if lease_id.startswith(prefix)
        ]
        for lease_id in reclaimed:
            self._active.pop(lease_id, None)
        return reclaimed


class WorktreeIsolation:
    """Placeholder Git Worktree isolation provider for Stage 3-6 candidate runs.

    Full `git worktree add/remove` management will be wired when Stage 3/4
    (`scaffold`/`adapt`) are implemented.
    """

    def __init__(self, repo_root: str | Path, worktree_root: str | Path) -> None:
        self.repo_root = Path(repo_root)
        self.worktree_root = Path(worktree_root)

    def acquire(self, node_id: str, *, run_id: str) -> IsolationLease:
        raise NotImplementedError(
            "WorktreeIsolation.acquire is scheduled for Stage 3/4 implementation"
        )

    def prepare(self, lease: IsolationLease) -> None:
        raise NotImplementedError(
            "WorktreeIsolation.prepare is scheduled for Stage 3/4 implementation"
        )

    def release(self, lease: IsolationLease) -> None:
        raise NotImplementedError(
            "WorktreeIsolation.release is scheduled for Stage 3/4 implementation"
        )

    def reclaim(self, run_id: str) -> list[str]:
        raise NotImplementedError(
            "WorktreeIsolation.reclaim is scheduled for Stage 3/4 implementation"
        )

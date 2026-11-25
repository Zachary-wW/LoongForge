"""Durable DAG scheduler with atomic checkpointing and crash-resume support.

Inspired by Apache-2.0 NVIDIA TensorRT-LLM `agent-flow/orchestration/scheduler.py`.
Uses only Python standard library.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import StrEnum
from pathlib import Path
from typing import Any

from .graph import ExecutionGraph, Node
from .invalidation import downstream_closure, invalidation_set
from .isolation import IsolationLease, IsolationProvider, NoOpIsolation


class NodeStatus(StrEnum):
    """Lifecycle status of an executable node in the scheduler."""

    PENDING = "pending"
    RUNNING = "running"
    DONE = "done"
    FAILED = "failed"
    BLOCKED = "blocked"
    INTERRUPTED = "interrupted"


class SchedulerError(RuntimeError):
    """Structured error raised on scheduler or checkpoint integrity failures."""

    def __init__(self, reason_code: str, message: str) -> None:
        self.reason_code = reason_code
        self.message = message
        super().__init__(message)


@dataclass(frozen=True, slots=True)
class NodeRunResult:
    """Return payload from a node runner callback."""

    status: NodeStatus = NodeStatus.DONE
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None

    def __post_init__(self) -> None:
        if self.status not in {
            NodeStatus.DONE,
            NodeStatus.FAILED,
            NodeStatus.BLOCKED,
            NodeStatus.INTERRUPTED,
        }:
            raise SchedulerError(
                "INVALID_TERMINAL_STATUS",
                f"node runner returned non-terminal status: {self.status}",
            )


@dataclass
class NodeRecord:
    """Persisted state record for a single leaf node."""

    node_id: str
    status: NodeStatus = NodeStatus.PENDING
    attempts: int = 0
    output: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    updated_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "node_id": self.node_id,
            "status": self.status.value,
            "attempts": self.attempts,
            "output": dict(self.output),
            "error": self.error,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NodeRecord:
        return cls(
            node_id=str(data["node_id"]),
            status=NodeStatus(data["status"]),
            attempts=int(data.get("attempts", 0)),
            output=dict(data.get("output", {})),
            error=data.get("error"),
            updated_at=data.get("updated_at"),
        )


@dataclass
class GraphState:
    """Atomic checkpoint representation of a DAG execution."""

    run_id: str
    graph_id: str
    graph_fingerprint: str
    nodes: dict[str, NodeRecord]
    graph_snapshot: dict[str, Any] | None = None
    schema_version: int = 1
    revision: int = 0
    updated_at: str = field(default_factory=lambda: _utc_now())

    @classmethod
    def initial(cls, graph: ExecutionGraph, *, run_id: str) -> GraphState:
        return cls(
            run_id=run_id,
            graph_id=graph.graph_id,
            graph_fingerprint=graph.fingerprint(),
            graph_snapshot=graph.to_dict(),
            nodes={
                leaf_id: NodeRecord(node_id=leaf_id, status=NodeStatus.PENDING)
                for leaf_id in graph.topological_ids()
            },
        )

    def status_of(self, node_id: str) -> NodeStatus:
        return self.nodes[node_id].status

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "graph_id": self.graph_id,
            "graph_fingerprint": self.graph_fingerprint,
            "graph_snapshot": self.graph_snapshot,
            "revision": self.revision,
            "updated_at": self.updated_at,
            "nodes": {
                node_id: record.to_dict() for node_id, record in self.nodes.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GraphState:
        if not isinstance(data, dict):
            raise SchedulerError("INVALID_CHECKPOINT", "checkpoint must be a JSON object")
        try:
            raw_nodes = data["nodes"]
            if not isinstance(raw_nodes, dict):
                raise SchedulerError(
                    "INVALID_CHECKPOINT", "checkpoint 'nodes' must be a mapping"
                )
            raw_snapshot = data.get("graph_snapshot")
            if raw_snapshot is not None and not isinstance(raw_snapshot, dict):
                raise SchedulerError(
                    "INVALID_CHECKPOINT", "checkpoint 'graph_snapshot' must be a mapping"
                )
            return cls(
                schema_version=int(data.get("schema_version", 1)),
                run_id=str(data["run_id"]),
                graph_id=str(data["graph_id"]),
                graph_fingerprint=str(data["graph_fingerprint"]),
                graph_snapshot=raw_snapshot,
                revision=int(data.get("revision", 0)),
                updated_at=str(data.get("updated_at", _utc_now())),
                nodes={
                    node_id: NodeRecord.from_dict(record)
                    for node_id, record in raw_nodes.items()
                },
            )
        except (KeyError, TypeError, ValueError) as error:
            raise SchedulerError(
                "INVALID_CHECKPOINT", f"failed to parse checkpoint: {error}"
            ) from error

    def save_atomic(self, path: str | Path) -> None:
        """Atomically persist checkpoint via temp file + fsync + os.replace."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps(
                self.to_dict(),
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
            + "\n"
        ).encode("utf-8")

        fd, tmp_name = tempfile.mkstemp(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=str(target.parent),
        )
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(tmp_path, target)
            _fsync_directory(target.parent)
        finally:
            if tmp_path.exists():
                tmp_path.unlink()

    @classmethod
    def load(cls, path: str | Path) -> GraphState:
        try:
            raw = Path(path).read_text(encoding="utf-8")
        except OSError as error:
            raise SchedulerError("CHECKPOINT_READ_FAILED", str(error)) from error
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as error:
            raise SchedulerError("CHECKPOINT_JSON_INVALID", str(error)) from error
        return cls.from_dict(parsed)


NodeExecutor = Callable[[Node, IsolationLease, GraphState], NodeRunResult | dict[str, Any] | None]


class NodeScheduler:
    """Deterministic DAG scheduler with atomic per-transition checkpoints.

    - Persists `GraphState` atomically on every state transition.
    - Resumes from an existing checkpoint, skipping nodes already `done`.
    - Any node left in `running` from a prior crash is marked `interrupted`
      on resume and re-executed (after isolation lease reclaim).
    - Downstream dependents of `failed` or `blocked` nodes are marked `blocked`.
    """

    def __init__(
        self,
        graph: ExecutionGraph,
        *,
        checkpoint_path: str | Path,
        run_id: str = "run-1",
        isolation: IsolationProvider | None = None,
        on_transition: Callable[[str, NodeStatus, GraphState], None] | None = None,
        allow_replan: bool = False,
        content_changed: Iterable[str] = (),
    ) -> None:
        if isinstance(content_changed, str):
            raise TypeError("content_changed must be an iterable of node IDs, not str")
        self.graph = graph
        self.checkpoint_path = Path(checkpoint_path)
        self.run_id = run_id
        self.isolation: IsolationProvider = isolation or NoOpIsolation()
        self._on_transition = on_transition
        self._allow_replan = allow_replan
        self._content_changed = tuple(content_changed)
        self.state = self._load_or_initialize_state()

    def _load_or_initialize_state(self) -> GraphState:
        if not self.checkpoint_path.exists():
            state = GraphState.initial(self.graph, run_id=self.run_id)
            self._persist(state)
            return state

        state = GraphState.load(self.checkpoint_path)
        if state.graph_id != self.graph.graph_id:
            raise SchedulerError(
                "CHECKPOINT_GRAPH_MISMATCH",
                f"checkpoint graph_id {state.graph_id!r} != {self.graph.graph_id!r}",
            )

        expected_ids = set(self.graph.topological_ids())
        actual_ids = set(state.nodes.keys())
        current_fingerprint = self.graph.fingerprint()
        fingerprint_mismatch = state.graph_fingerprint != current_fingerprint
        node_set_mismatch = expected_ids != actual_ids
        mutated = False

        if fingerprint_mismatch or node_set_mismatch:
            if not self._allow_replan:
                if node_set_mismatch:
                    raise SchedulerError(
                        "CHECKPOINT_NODE_SET_MISMATCH",
                        "checkpoint node set does not match execution graph",
                    )
                raise SchedulerError(
                    "CHECKPOINT_FINGERPRINT_MISMATCH",
                    (
                        f"checkpoint graph_fingerprint {state.graph_fingerprint!r} "
                        f"!= {current_fingerprint!r}; pass allow_replan=True to migrate"
                    ),
                )
            if state.graph_snapshot is None:
                raise SchedulerError(
                    "CHECKPOINT_REPLAN_UNSUPPORTED",
                    "checkpoint lacks graph_snapshot required for replan migration",
                )
            old_graph = ExecutionGraph.from_dict(state.graph_snapshot)
            to_invalidate = invalidation_set(
                old_graph,
                self.graph,
                content_changed=self._content_changed,
            )
            migrated_nodes: dict[str, NodeRecord] = {}
            now = _utc_now()
            for node_id in self.graph.topological_ids():
                if node_id not in state.nodes or node_id in to_invalidate:
                    migrated_nodes[node_id] = NodeRecord(
                        node_id=node_id,
                        status=NodeStatus.PENDING,
                        attempts=state.nodes[node_id].attempts
                        if node_id in state.nodes
                        else 0,
                        output={},
                        error=None,
                        updated_at=now,
                    )
                else:
                    migrated_nodes[node_id] = state.nodes[node_id]
            state.nodes = migrated_nodes
            state.graph_fingerprint = current_fingerprint
            state.graph_snapshot = self.graph.to_dict()
            mutated = True
        elif self._content_changed:
            to_invalidate = downstream_closure(self.graph, self._content_changed)
            now = _utc_now()
            for node_id in self.graph.topological_ids():
                if node_id in to_invalidate:
                    record = state.nodes[node_id]
                    record.status = NodeStatus.PENDING
                    record.output = {}
                    record.error = None
                    record.updated_at = now
                    mutated = True

        # Recover from crash: reclaim leases and mark stale RUNNING nodes INTERRUPTED.
        self.isolation.reclaim(state.run_id)
        for node_id in self.graph.topological_ids():
            record = state.nodes[node_id]
            if record.status == NodeStatus.RUNNING:
                record.status = NodeStatus.INTERRUPTED
                record.error = "previous run interrupted while node was running"
                record.updated_at = _utc_now()
                mutated = True
        if mutated:
            self._persist(state)
        return state

    def _persist(self, state: GraphState) -> None:
        state.revision += 1
        state.updated_at = _utc_now()
        state.save_atomic(self.checkpoint_path)

    def _transition(
        self,
        node_id: str,
        new_status: NodeStatus,
        *,
        output: dict[str, Any] | None = None,
        error: str | None = None,
        increment_attempt: bool = False,
    ) -> None:
        record = self.state.nodes[node_id]
        record.status = new_status
        if increment_attempt:
            record.attempts += 1
        if output is not None:
            record.output = dict(output)
        record.error = error
        record.updated_at = _utc_now()
        self._persist(self.state)
        if self._on_transition is not None:
            self._on_transition(node_id, new_status, self.state)

    def invalidate(
        self,
        node_ids: Iterable[str],
        *,
        propagate: bool = True,
    ) -> list[str]:
        """Atomically reset specified nodes (and by default all downstream dependents) to PENDING."""
        if isinstance(node_ids, str):
            raise TypeError("node_ids must be an iterable of node IDs, not str")
        requested = list(node_ids)
        for node_id in requested:
            self.graph.get_node(node_id)

        if propagate:
            targets = downstream_closure(self.graph, requested)
        else:
            targets = {
                node_id for node_id in requested if node_id in self.state.nodes
            }

        reset: list[str] = []
        now = _utc_now()
        for node_id in self.graph.topological_ids():
            if node_id in targets:
                record = self.state.nodes[node_id]
                record.status = NodeStatus.PENDING
                record.output = {}
                record.error = None
                record.updated_at = now
                reset.append(node_id)

        if reset:
            self._persist(self.state)
            if self._on_transition is not None:
                for node_id in reset:
                    self._on_transition(node_id, NodeStatus.PENDING, self.state)
        return reset

    def run(
        self,
        executor: NodeExecutor,
        *,
        retry_failed: bool = False,
    ) -> GraphState:
        """Execute all runnable nodes in topological order until completion or block."""
        if retry_failed:
            failed_roots = [
                node_id
                for node_id in self.graph.topological_ids()
                if self.state.status_of(node_id) == NodeStatus.FAILED
            ]
            if failed_roots:
                self.invalidate(failed_roots, propagate=True)

        runnable_states = {NodeStatus.PENDING, NodeStatus.INTERRUPTED}
        rerun_or_reset: set[str] = set()

        for node_id in self.graph.topological_ids():
            current_status = self.state.status_of(node_id)
            deps = self.graph.dependencies_of(node_id)

            # If an upstream dependency re-ran in this pass, stale DONE nodes must recompute.
            if current_status == NodeStatus.DONE and any(
                dep_id in rerun_or_reset for dep_id in deps
            ):
                self._transition(
                    node_id,
                    NodeStatus.PENDING,
                    output={},
                    error=None,
                )
                current_status = NodeStatus.PENDING

            # If all upstream dependencies are now DONE, previously BLOCKED nodes can run again.
            if current_status == NodeStatus.BLOCKED and all(
                self.state.status_of(dep_id) == NodeStatus.DONE for dep_id in deps
            ):
                self._transition(
                    node_id,
                    NodeStatus.PENDING,
                    output={},
                    error=None,
                )
                current_status = NodeStatus.PENDING

            if current_status == NodeStatus.DONE:
                continue

            unmet_blocking = [
                dep_id
                for dep_id in deps
                if self.state.status_of(dep_id)
                in {NodeStatus.FAILED, NodeStatus.BLOCKED, NodeStatus.INTERRUPTED}
            ]
            if unmet_blocking:
                if current_status != NodeStatus.BLOCKED:
                    self._transition(
                        node_id,
                        NodeStatus.BLOCKED,
                        error=(
                            "blocked by upstream node(s): "
                            + ", ".join(unmet_blocking)
                        ),
                    )
                continue

            if current_status not in runnable_states:
                continue

            if any(self.state.status_of(dep_id) != NodeStatus.DONE for dep_id in deps):
                self._transition(
                    node_id,
                    NodeStatus.BLOCKED,
                    error="upstream dependencies did not complete",
                )
                continue

            node = self.graph.get_node(node_id)
            lease = self.isolation.acquire(node_id, run_id=self.state.run_id)
            try:
                self.isolation.prepare(lease)
                self._transition(
                    node_id,
                    NodeStatus.RUNNING,
                    error=None,
                    increment_attempt=True,
                )
                raw_result = executor(node, lease, self.state)
                result = _normalize_result(raw_result)
                self._transition(
                    node_id,
                    result.status,
                    output=result.output,
                    error=result.error,
                )
                rerun_or_reset.add(node_id)
            except KeyboardInterrupt:
                self._transition(
                    node_id,
                    NodeStatus.INTERRUPTED,
                    error="execution interrupted by signal",
                )
                raise
            except Exception as error:
                self._transition(
                    node_id,
                    NodeStatus.FAILED,
                    error=f"{type(error).__name__}: {error}",
                )
            finally:
                self.isolation.release(lease)

        return self.state


def _normalize_result(
    raw: NodeRunResult | dict[str, Any] | None,
) -> NodeRunResult:
    if raw is None:
        return NodeRunResult(status=NodeStatus.DONE)
    if isinstance(raw, NodeRunResult):
        return raw
    if isinstance(raw, dict):
        return NodeRunResult(status=NodeStatus.DONE, output=dict(raw))
    raise SchedulerError(
        "INVALID_NODE_RESULT",
        f"unsupported node result type: {type(raw).__name__}",
    )


def _utc_now() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


def _fsync_directory(path: Path) -> None:
    try:
        descriptor = os.open(path, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)

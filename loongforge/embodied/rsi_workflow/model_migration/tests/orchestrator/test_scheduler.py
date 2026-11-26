import json
from pathlib import Path

import pytest

from core.orchestrator.graph import ExecutionGraph, Node
from core.orchestrator.isolation import IsolationLease, NoOpIsolation
from core.orchestrator.scheduler import (
    GraphState,
    NodeRunResult,
    NodeScheduler,
    NodeStatus,
    SchedulerError,
)


def _linear_graph(*, anatomy_hash: str = "v1") -> ExecutionGraph:
    return ExecutionGraph(
        [
            Node(id="intake", stage="intake", content_hash="v1"),
            Node(
                id="anatomy",
                stage="anatomy",
                depends_on=("intake",),
                content_hash=anatomy_hash,
            ),
            Node(id="scaffold", stage="scaffold", depends_on=("anatomy",), content_hash="v1"),
            Node(id="verify", stage="verify", depends_on=("scaffold",), content_hash="v1"),
        ],
        graph_id="linear",
    )


def test_scheduler_executes_dag_in_order_and_persists_atomic_checkpoints(
    tmp_path: Path,
) -> None:
    graph = _linear_graph()
    checkpoint = tmp_path / "state" / "graph_state.json"
    isolation = NoOpIsolation(base_dir=tmp_path)
    observed_transitions: list[tuple[str, NodeStatus, int]] = []
    executed: list[str] = []

    def on_transition(node_id: str, status: NodeStatus, state: GraphState) -> None:
        disk_state = GraphState.load(checkpoint)
        assert disk_state.revision == state.revision
        assert disk_state.status_of(node_id) == status
        observed_transitions.append((node_id, status, state.revision))

    def executor(
        node: Node,
        lease: IsolationLease,
        state: GraphState,
    ) -> NodeRunResult:
        assert lease.node_id == node.id
        assert lease.lease_id in isolation.active_leases
        executed.append(node.id)
        return NodeRunResult(status=NodeStatus.DONE, output={"ran": node.id})

    scheduler = NodeScheduler(
        graph,
        checkpoint_path=checkpoint,
        run_id="run-ok",
        isolation=isolation,
        on_transition=on_transition,
    )
    final_state = scheduler.run(executor)

    assert executed == ["intake", "anatomy", "scaffold", "verify"]
    assert all(
        final_state.status_of(node_id) == NodeStatus.DONE
        for node_id in graph.topological_ids()
    )
    assert isolation.active_leases == {}
    # Each node transitions RUNNING -> DONE on disk.
    assert len(observed_transitions) == 8
    assert not list(checkpoint.parent.glob("*.tmp"))


def test_scheduler_blocks_downstream_nodes_on_failure_and_keeps_independent_branch(
    tmp_path: Path,
) -> None:
    graph = ExecutionGraph(
        [
            Node(id="root", stage="intake"),
            Node(id="branch_fail", stage="anatomy", depends_on=("root",)),
            Node(id="downstream", stage="scaffold", depends_on=("branch_fail",)),
            Node(id="branch_ok", stage="verify", depends_on=("root",)),
        ],
        graph_id="branching",
    )
    checkpoint = tmp_path / "checkpoint.json"
    executed: list[str] = []

    def executor(
        node: Node,
        lease: IsolationLease,
        state: GraphState,
    ) -> NodeRunResult:
        executed.append(node.id)
        if node.id == "branch_fail":
            raise RuntimeError("boom")
        return NodeRunResult(status=NodeStatus.DONE)

    scheduler = NodeScheduler(graph, checkpoint_path=checkpoint, run_id="run-fail")
    state = scheduler.run(executor)

    assert executed == ["root", "branch_fail", "branch_ok"]
    assert state.status_of("root") == NodeStatus.DONE
    assert state.status_of("branch_fail") == NodeStatus.FAILED
    assert state.status_of("downstream") == NodeStatus.BLOCKED
    assert state.status_of("branch_ok") == NodeStatus.DONE
    assert "branch_fail" in (state.nodes["downstream"].error or "")


def test_scheduler_resumes_from_crash_checkpoint_and_skips_done_nodes(
    tmp_path: Path,
) -> None:
    graph = _linear_graph()
    checkpoint = tmp_path / "crash_checkpoint.json"

    # Simulate a previous run that finished `intake`, crashed mid-`anatomy` (still RUNNING).
    crashed_state = GraphState.initial(graph, run_id="run-crash")
    crashed_state.nodes["intake"].status = NodeStatus.DONE
    crashed_state.nodes["intake"].attempts = 1
    crashed_state.nodes["intake"].output = {"artifact": "intake-v1"}
    crashed_state.nodes["anatomy"].status = NodeStatus.RUNNING
    crashed_state.nodes["anatomy"].attempts = 1
    crashed_state.save_atomic(checkpoint)

    class TrackingIsolation(NoOpIsolation):
        def __init__(self) -> None:
            super().__init__(base_dir=tmp_path)
            self.reclaimed_runs: list[str] = []

        def reclaim(self, run_id: str) -> list[str]:
            self.reclaimed_runs.append(run_id)
            return super().reclaim(run_id)

    isolation = TrackingIsolation()
    resumed = NodeScheduler(
        graph,
        checkpoint_path=checkpoint,
        run_id="run-crash",
        isolation=isolation,
    )
    assert isolation.reclaimed_runs == ["run-crash"]
    assert resumed.state.status_of("intake") == NodeStatus.DONE
    assert resumed.state.status_of("anatomy") == NodeStatus.INTERRUPTED

    executed: list[str] = []

    def executor(
        node: Node,
        lease: IsolationLease,
        state: GraphState,
    ) -> NodeRunResult:
        executed.append(node.id)
        return NodeRunResult(status=NodeStatus.DONE, output={"resumed": node.id})

    final_state = resumed.run(executor)

    # `intake` was already DONE and must NOT re-execute.
    assert executed == ["anatomy", "scaffold", "verify"]
    assert final_state.nodes["intake"].attempts == 1
    assert final_state.nodes["anatomy"].attempts == 2
    assert final_state.nodes["intake"].output == {"artifact": "intake-v1"}
    persisted = json.loads(checkpoint.read_text(encoding="utf-8"))
    assert persisted["nodes"]["verify"]["status"] == "done"


def test_invalidate_propagates_transitively_and_reruns_all_downstream_nodes(
    tmp_path: Path,
) -> None:
    graph = _linear_graph()
    checkpoint = tmp_path / "invalidate.json"
    scheduler = NodeScheduler(graph, checkpoint_path=checkpoint, run_id="run-inv")

    version = {"n": 1}
    ran_first: list[str] = []

    def executor(node: Node, lease: IsolationLease, state: GraphState) -> NodeRunResult:
        ran_first.append(node.id)
        return NodeRunResult(status=NodeStatus.DONE, output={"v": version["n"]})

    scheduler.run(executor)
    assert ran_first == ["intake", "anatomy", "scaffold", "verify"]

    rev_before = scheduler.state.revision
    reset = scheduler.invalidate(["anatomy"])
    assert reset == ["anatomy", "scaffold", "verify"]
    # Single atomic persist for the entire transitive batch.
    assert scheduler.state.revision == rev_before + 1
    assert scheduler.state.status_of("intake") == NodeStatus.DONE
    for nid in ("anatomy", "scaffold", "verify"):
        assert scheduler.state.status_of(nid) == NodeStatus.PENDING
        assert scheduler.state.nodes[nid].output == {}

    version["n"] = 2
    ran_second: list[str] = []

    def executor_v2(
        node: Node, lease: IsolationLease, state: GraphState
    ) -> NodeRunResult:
        ran_second.append(node.id)
        return NodeRunResult(status=NodeStatus.DONE, output={"v": version["n"]})

    final_state = scheduler.run(executor_v2)
    assert ran_second == ["anatomy", "scaffold", "verify"]
    assert final_state.nodes["intake"].output == {"v": 1}
    for nid in ("anatomy", "scaffold", "verify"):
        assert final_state.nodes[nid].output == {"v": 2}

    with pytest.raises(TypeError):
        scheduler.invalidate("anatomy")  # type: ignore[arg-type]


def test_blocked_downstream_recomputes_when_upstream_recovers_via_retry_failed(
    tmp_path: Path,
) -> None:
    graph = _linear_graph()
    checkpoint = tmp_path / "retry.json"
    scheduler = NodeScheduler(graph, checkpoint_path=checkpoint, run_id="run-retry")

    should_fail = {"anatomy": True}
    ran: list[str] = []

    def executor(node: Node, lease: IsolationLease, state: GraphState) -> NodeRunResult:
        ran.append(node.id)
        if node.id == "anatomy" and should_fail["anatomy"]:
            raise RuntimeError("transient gpu error")
        return NodeRunResult(status=NodeStatus.DONE, output={"ok": node.id})

    state1 = scheduler.run(executor)
    assert state1.status_of("intake") == NodeStatus.DONE
    assert state1.status_of("anatomy") == NodeStatus.FAILED
    assert state1.status_of("scaffold") == NodeStatus.BLOCKED
    assert state1.status_of("verify") == NodeStatus.BLOCKED

    should_fail["anatomy"] = False
    ran.clear()
    state2 = scheduler.run(executor, retry_failed=True)
    assert ran == ["anatomy", "scaffold", "verify"]
    assert all(
        state2.status_of(nid) == NodeStatus.DONE for nid in graph.topological_ids()
    )


def test_checkpoint_fingerprint_mismatch_fails_closed_by_default_and_replans_when_allowed(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "replan.json"
    g_v1 = _linear_graph(anatomy_hash="v1")
    s1 = NodeScheduler(g_v1, checkpoint_path=checkpoint, run_id="run-1")
    s1.run(lambda node, lease, state: NodeRunResult(output={"ver": 1}))

    g_v2 = _linear_graph(anatomy_hash="v2")
    with pytest.raises(SchedulerError) as raised:
        NodeScheduler(g_v2, checkpoint_path=checkpoint, run_id="run-1")
    assert raised.value.reason_code == "CHECKPOINT_FINGERPRINT_MISMATCH"

    # Explicit replan resets changed node + downstream while keeping untouched upstream DONE.
    s2 = NodeScheduler(
        g_v2,
        checkpoint_path=checkpoint,
        run_id="run-1",
        allow_replan=True,
    )
    assert s2.state.status_of("intake") == NodeStatus.DONE
    assert s2.state.status_of("anatomy") == NodeStatus.PENDING
    assert s2.state.status_of("scaffold") == NodeStatus.PENDING
    assert s2.state.status_of("verify") == NodeStatus.PENDING

    ran: list[str] = []
    s2.run(
        lambda node, lease, state: (
            ran.append(node.id),
            NodeRunResult(output={"ver": 2}),
        )[1]
    )
    assert ran == ["anatomy", "scaffold", "verify"]
    assert s2.state.nodes["intake"].output == {"ver": 1}
    assert s2.state.nodes["verify"].output == {"ver": 2}


def test_checkpoint_replan_handles_added_and_removed_nodes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "add_remove.json"
    g_old = ExecutionGraph(
        [
            Node(id="intake", stage="intake", content_hash="v1"),
            Node(id="anatomy", stage="anatomy", depends_on=("intake",), content_hash="v1"),
            Node(id="obsolete", stage="verify", depends_on=("anatomy",), content_hash="v1"),
        ],
        graph_id="evolving",
    )
    s_old = NodeScheduler(g_old, checkpoint_path=checkpoint, run_id="run-evolve")
    s_old.run(lambda node, lease, state: NodeRunResult(output={"node": node.id}))

    g_new = ExecutionGraph(
        [
            Node(id="intake", stage="intake", content_hash="v1"),
            Node(id="anatomy", stage="anatomy", depends_on=("intake",), content_hash="v1"),
            Node(id="scaffold", stage="scaffold", depends_on=("anatomy",), content_hash="v1"),
        ],
        graph_id="evolving",
    )

    # Default fail-closed on node set mismatch.
    with pytest.raises(SchedulerError) as mismatch:
        NodeScheduler(g_new, checkpoint_path=checkpoint, run_id="run-evolve")
    assert mismatch.value.reason_code == "CHECKPOINT_NODE_SET_MISMATCH"

    # With allow_replan=True: obsolete dropped, intake/anatomy kept DONE, scaffold PENDING.
    s_new = NodeScheduler(
        g_new,
        checkpoint_path=checkpoint,
        run_id="run-evolve",
        allow_replan=True,
    )
    assert set(s_new.state.nodes.keys()) == {"intake", "anatomy", "scaffold"}
    assert s_new.state.status_of("intake") == NodeStatus.DONE
    assert s_new.state.status_of("anatomy") == NodeStatus.DONE
    assert s_new.state.status_of("scaffold") == NodeStatus.PENDING

    ran: list[str] = []
    s_new.run(
        lambda node, lease, state: (
            ran.append(node.id),
            NodeRunResult(output={"node": node.id}),
        )[1]
    )
    assert ran == ["scaffold"]


def test_corrupt_checkpoint_json_and_isolation_prepare_failure(tmp_path: Path) -> None:
    graph = _linear_graph()
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(SchedulerError) as err:
        NodeScheduler(graph, checkpoint_path=corrupt)
    assert err.value.reason_code == "CHECKPOINT_JSON_INVALID"

    class FailingPrepareIsolation(NoOpIsolation):
        def prepare(self, lease: IsolationLease) -> None:
            if lease.node_id == "anatomy":
                raise OSError("worktree creation failed")

    checkpoint = tmp_path / "prepare_fail.json"
    scheduler = NodeScheduler(
        graph,
        checkpoint_path=checkpoint,
        isolation=FailingPrepareIsolation(base_dir=tmp_path),
    )
    state = scheduler.run(lambda node, lease, state: NodeRunResult())
    assert state.status_of("intake") == NodeStatus.DONE
    assert state.status_of("anatomy") == NodeStatus.FAILED
    assert state.status_of("scaffold") == NodeStatus.BLOCKED
    assert "worktree creation failed" in (state.nodes["anatomy"].error or "")

"""Transitive invalidation calculation across ExecutionGraph revisions.

Inspired by Apache-2.0 NVIDIA TensorRT-LLM `agent-flow/orchestration/invalidation.py`.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable

from .graph import ExecutionGraph


def downstream_closure(
    graph: ExecutionGraph,
    seed_ids: Iterable[str],
) -> set[str]:
    """Compute transitive closure of affected leaf nodes in `graph` starting from `seed_ids`.

    Accepts both leaf node IDs and enclosing group node IDs. Unknown IDs in
    `seed_ids` are ignored when computing closures across graph revisions, or
    validated by callers that require strict membership.
    """
    seeds = set(seed_ids)
    directly_affected: set[str] = set()
    for leaf_id in graph.leaf_ids:
        if leaf_id in seeds:
            directly_affected.add(leaf_id)
            continue
        parent_id = graph.parent_of(leaf_id)
        while parent_id is not None:
            if parent_id in seeds:
                directly_affected.add(leaf_id)
                break
            parent_id = graph.parent_of(parent_id)

    affected: set[str] = set(directly_affected)
    queue: deque[str] = deque(directly_affected)
    while queue:
        current = queue.popleft()
        for dependent_id in graph.dependents_of(current):
            if dependent_id not in affected:
                affected.add(dependent_id)
                queue.append(dependent_id)
    return affected


def invalidation_set(
    old_graph: ExecutionGraph,
    new_graph: ExecutionGraph,
    *,
    content_changed: Iterable[str] = (),
) -> set[str]:
    """Compute the transitive set of leaf node IDs invalidated in `new_graph`.

    A leaf node in `new_graph` is directly invalidated if:
    1. It is newly introduced (not present as a leaf in `old_graph`);
    2. Its effective signature changed (kind, content_hash, metadata, enclosing
       group metadata/hash, or effective dependencies changed);
    3. Its ID (or any of its enclosing group IDs) appears in `content_changed`.

    Invalidation then propagates transitively to all downstream dependents in
    `new_graph`.
    """
    if isinstance(content_changed, str):
        raise TypeError("content_changed must be an iterable of node IDs, not str")
    changed_ids = set(content_changed)
    old_leaves = set(old_graph.leaf_ids)
    new_leaves = set(new_graph.leaf_ids)

    directly_invalidated: set[str] = set()

    for leaf_id in new_leaves:
        if leaf_id not in old_leaves:
            directly_invalidated.add(leaf_id)
            continue
        if leaf_id in changed_ids:
            directly_invalidated.add(leaf_id)
            continue
        # Also propagate if an enclosing group ID is marked content-changed.
        parent_id = new_graph.parent_of(leaf_id)
        while parent_id is not None:
            if parent_id in changed_ids:
                directly_invalidated.add(leaf_id)
                break
            parent_id = new_graph.parent_of(parent_id)
        if leaf_id in directly_invalidated:
            continue
        if old_graph.effective_signature(leaf_id) != new_graph.effective_signature(
            leaf_id
        ):
            directly_invalidated.add(leaf_id)

    return downstream_closure(new_graph, directly_invalidated)

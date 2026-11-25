"""Nested DAG data model, validation, cycle detection, and topological flattening.

Inspired by Apache-2.0 NVIDIA TensorRT-LLM `agent-flow/orchestration/graph.py`.
Uses only Python standard library + PyYAML.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

VALID_STAGES: frozenset[str] = frozenset(
    {
        "intake",
        "anatomy",
        "scaffold",
        "adapt",
        "verify",
        "deliver",
    }
)
GROUP_STAGE = "group"
SUPPORTED_SCHEMA_VERSION = 1

_NODE_KNOWN_KEYS: frozenset[str] = frozenset(
    {"id", "stage", "depends_on", "children", "content_hash", "metadata"}
)
_STAGE_EXTRA_METADATA_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "description",
        "isolation",
        "skills",
        "extension_points",
        "checks",
        "tools",
    }
)


def _load_yaml_unique_keys(raw_text: str) -> Any:
    """Parse YAML while rejecting duplicate mapping keys (fail-closed)."""

    class _UniqueKeyLoader(yaml.SafeLoader):
        pass

    def _construct_mapping(
        loader: Any,
        node: Any,
        deep: bool = False,
    ) -> dict[Any, Any]:
        mapping: dict[Any, Any] = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node, deep=deep)
            if key in mapping:
                raise GraphError(
                    "DUPLICATE_YAML_KEY",
                    f"duplicate YAML key: {key!r}",
                )
            mapping[key] = loader.construct_object(value_node, deep=deep)
        return mapping

    _UniqueKeyLoader.add_constructor(
        yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
        _construct_mapping,
    )
    return yaml.load(raw_text, Loader=_UniqueKeyLoader)


class GraphError(ValueError):
    """Structured error raised when constructing or validating an ExecutionGraph."""

    def __init__(
        self,
        reason_code: str,
        message: str,
        *,
        node_id: str | None = None,
    ) -> None:
        self.reason_code = reason_code
        self.message = message
        self.node_id = node_id
        super().__init__(message)

    def to_dict(self) -> dict[str, Any]:
        return {
            "reason_code": self.reason_code,
            "message": self.message,
            "node_id": self.node_id,
        }


@dataclass(frozen=True, slots=True)
class Node:
    """A node in a declarative, optionally nested execution DAG."""

    id: str
    stage: str = "intake"
    depends_on: tuple[str, ...] = ()
    children: tuple[Node, ...] = ()
    content_hash: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not self.id.strip():
            raise GraphError("INVALID_NODE_ID", "node id must be a non-empty string")
        if self.id != self.id.strip():
            raise GraphError(
                "INVALID_NODE_ID",
                f"node id must not contain leading or trailing whitespace: {self.id!r}",
                node_id=self.id,
            )
        children = tuple(self.children)
        if not isinstance(self.stage, str) or not self.stage.strip():
            raise GraphError(
                "INVALID_STAGE",
                "node stage must be a non-empty string",
                node_id=self.id,
            )
        if children:
            if self.stage != GROUP_STAGE and self.stage not in VALID_STAGES:
                raise GraphError(
                    "INVALID_STAGE",
                    f"group node {self.id!r} has invalid stage {self.stage!r}",
                    node_id=self.id,
                )
        else:
            if self.stage not in VALID_STAGES:
                raise GraphError(
                    "INVALID_STAGE",
                    (
                        f"leaf node {self.id!r} has invalid stage {self.stage!r}; "
                        f"expected one of {sorted(VALID_STAGES)}"
                    ),
                    node_id=self.id,
                )
        deps = tuple(self.depends_on)
        if len(deps) != len(set(deps)):
            raise GraphError(
                "DUPLICATE_DEPENDENCY",
                f"node {self.id!r} contains duplicate depends_on entries",
                node_id=self.id,
            )
        if self.id in deps:
            raise GraphError(
                "SELF_DEPENDENCY",
                f"node {self.id!r} cannot depend on itself",
                node_id=self.id,
            )
        object.__setattr__(self, "depends_on", deps)
        object.__setattr__(self, "children", children)
        object.__setattr__(self, "metadata", copy.deepcopy(dict(self.metadata)))

    @property
    def is_group(self) -> bool:
        return len(self.children) > 0

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Node:
        if not isinstance(data, dict):
            raise GraphError("INVALID_NODE_SPEC", "node specification must be a mapping")
        if "kind" in data and "stage" not in data:
            raise GraphError(
                "INVALID_STAGE",
                "node specification must use 'stage' instead of legacy 'kind'",
                node_id=data.get("id") if isinstance(data.get("id"), str) else None,
            )
        allowed_keys = _NODE_KNOWN_KEYS | _STAGE_EXTRA_METADATA_KEYS
        unknown_keys = sorted(set(data.keys()) - allowed_keys)
        if unknown_keys:
            raise GraphError(
                "UNKNOWN_NODE_KEY",
                f"node specification contains unknown key(s): {', '.join(unknown_keys)}",
                node_id=data.get("id") if isinstance(data.get("id"), str) else None,
            )
        node_id = data.get("id")
        if not isinstance(node_id, str):
            raise GraphError("INVALID_NODE_ID", "node 'id' must be a string")
        raw_deps = data.get("depends_on", [])
        if not isinstance(raw_deps, list) or not all(
            isinstance(item, str) for item in raw_deps
        ):
            raise GraphError(
                "INVALID_DEPENDENCY_LIST",
                f"node {node_id!r} depends_on must be a list of strings",
                node_id=node_id,
            )
        raw_children = data.get("children", [])
        if not isinstance(raw_children, list):
            raise GraphError(
                "INVALID_CHILDREN_LIST",
                f"node {node_id!r} children must be a list",
                node_id=node_id,
            )
        children = tuple(cls.from_dict(child) for child in raw_children)
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise GraphError(
                "INVALID_METADATA",
                f"node {node_id!r} metadata must be a mapping",
                node_id=node_id,
            )
        merged_metadata = copy.deepcopy(metadata)
        for extra_key in sorted(_STAGE_EXTRA_METADATA_KEYS):
            if extra_key in data:
                merged_metadata[extra_key] = copy.deepcopy(data[extra_key])
        content_hash = data.get("content_hash")
        if content_hash is not None and not isinstance(content_hash, str):
            raise GraphError(
                "INVALID_CONTENT_HASH",
                f"node {node_id!r} content_hash must be a string or null",
                node_id=node_id,
            )
        if "stage" in data:
            stage = data["stage"]
        elif children:
            stage = GROUP_STAGE
        else:
            raise GraphError(
                "MISSING_STAGE",
                f"leaf node {node_id!r} must declare 'stage'",
                node_id=node_id,
            )
        return cls(
            id=node_id,
            stage=stage,
            depends_on=tuple(raw_deps),
            children=children,
            content_hash=content_hash,
            metadata=merged_metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "id": self.id,
            "stage": self.stage,
            "depends_on": list(self.depends_on),
            "metadata": copy.deepcopy(self.metadata),
        }
        if self.content_hash is not None:
            payload["content_hash"] = self.content_hash
        if self.children:
            payload["children"] = [child.to_dict() for child in self.children]
        return payload


class ExecutionGraph:
    """Validated, immutable nested DAG with global ID uniqueness and cycle checks."""

    def __init__(
        self,
        nodes: list[Node] | tuple[Node, ...],
        *,
        graph_id: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if not isinstance(graph_id, str) or not graph_id.strip():
            raise GraphError("INVALID_GRAPH_ID", "graph_id must be a non-empty string")
        self.graph_id = graph_id
        self.metadata: dict[str, Any] = dict(metadata or {})
        self.roots: tuple[Node, ...] = tuple(nodes)

        self._nodes_by_id: dict[str, Node] = {}
        self._parent_by_id: dict[str, str | None] = {}
        self._collect_unique_nodes(self.roots, parent_id=None)

        if not self._nodes_by_id:
            raise GraphError("EMPTY_GRAPH", "execution graph must contain at least one node")

        self._effective_deps: dict[str, tuple[str, ...]] = {}
        self._leaf_ids: tuple[str, ...] = ()
        self._build_effective_dependencies()
        self._topological_order: tuple[str, ...] = self._compute_topological_order()

    def _collect_unique_nodes(
        self,
        nodes: tuple[Node, ...],
        *,
        parent_id: str | None,
    ) -> None:
        for node in nodes:
            if node.id in self._nodes_by_id:
                raise GraphError(
                    "DUPLICATE_NODE_ID",
                    f"duplicate node id in graph: {node.id!r}",
                    node_id=node.id,
                )
            self._nodes_by_id[node.id] = node
            self._parent_by_id[node.id] = parent_id
            if node.children:
                self._collect_unique_nodes(node.children, parent_id=node.id)

    def _descendant_leaves(self, node_id: str) -> tuple[str, ...]:
        node = self._nodes_by_id[node_id]
        if not node.is_group:
            return (node.id,)
        leaves: list[str] = []
        for child in node.children:
            leaves.extend(self._descendant_leaves(child.id))
        return tuple(leaves)

    def _ancestor_ids(self, node_id: str) -> list[str]:
        ancestors: list[str] = []
        current = self._parent_by_id[node_id]
        while current is not None:
            ancestors.append(current)
            current = self._parent_by_id[current]
        return ancestors

    def _build_effective_dependencies(self) -> None:
        for node in self._nodes_by_id.values():
            for dep_id in node.depends_on:
                if dep_id not in self._nodes_by_id:
                    raise GraphError(
                        "UNKNOWN_DEPENDENCY",
                        f"node {node.id!r} depends on unknown node {dep_id!r}",
                        node_id=node.id,
                    )
                if dep_id in self._ancestor_ids(node.id):
                    raise GraphError(
                        "ANCESTOR_DEPENDENCY",
                        f"node {node.id!r} cannot depend on its own ancestor {dep_id!r}",
                        node_id=node.id,
                    )
                if node.id in self._ancestor_ids(dep_id):
                    raise GraphError(
                        "DESCENDANT_DEPENDENCY",
                        f"group node {node.id!r} cannot depend on its own descendant {dep_id!r}",
                        node_id=node.id,
                    )

        leaves = [
            node_id
            for node_id, node in self._nodes_by_id.items()
            if not node.is_group
        ]
        self._leaf_ids = tuple(leaves)

        for leaf_id in self._leaf_ids:
            raw_dep_targets: list[str] = []
            # Inherit dependencies from enclosing group nodes as well as own dependencies.
            chain = [*reversed(self._ancestor_ids(leaf_id)), leaf_id]
            for member_id in chain:
                raw_dep_targets.extend(self._nodes_by_id[member_id].depends_on)

            expanded: list[str] = []
            seen: set[str] = set()
            for target_id in raw_dep_targets:
                for dep_leaf in self._descendant_leaves(target_id):
                    if dep_leaf == leaf_id:
                        raise GraphError(
                            "CYCLE_DETECTED",
                            f"effective self-cycle detected for node {leaf_id!r}",
                            node_id=leaf_id,
                        )
                    if dep_leaf not in seen:
                        seen.add(dep_leaf)
                        expanded.append(dep_leaf)
            self._effective_deps[leaf_id] = tuple(expanded)

    def _compute_topological_order(self) -> tuple[str, ...]:
        in_degree: dict[str, int] = {
            leaf_id: len(self._effective_deps[leaf_id]) for leaf_id in self._leaf_ids
        }
        dependents: dict[str, list[str]] = {leaf_id: [] for leaf_id in self._leaf_ids}
        for leaf_id in self._leaf_ids:
            for dep_id in self._effective_deps[leaf_id]:
                dependents[dep_id].append(leaf_id)

        queue: deque[str] = deque(
            leaf_id for leaf_id in self._leaf_ids if in_degree[leaf_id] == 0
        )
        order: list[str] = []
        while queue:
            current = queue.popleft()
            order.append(current)
            for nxt in dependents[current]:
                in_degree[nxt] -= 1
                if in_degree[nxt] == 0:
                    queue.append(nxt)

        if len(order) != len(self._leaf_ids):
            remaining = [
                leaf_id for leaf_id in self._leaf_ids if in_degree[leaf_id] > 0
            ]
            raise GraphError(
                "CYCLE_DETECTED",
                f"cycle detected involving nodes: {', '.join(remaining)}",
                node_id=remaining[0],
            )
        return tuple(order)

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(self._nodes_by_id.keys())

    @property
    def leaf_ids(self) -> tuple[str, ...]:
        return self._leaf_ids

    def get_node(self, node_id: str) -> Node:
        try:
            return self._nodes_by_id[node_id]
        except KeyError as error:
            raise GraphError(
                "UNKNOWN_NODE",
                f"unknown node id: {node_id!r}",
                node_id=node_id,
            ) from error

    def parent_of(self, node_id: str) -> str | None:
        self.get_node(node_id)
        return self._parent_by_id[node_id]

    def dependencies_of(self, leaf_id: str) -> tuple[str, ...]:
        self.get_node(leaf_id)
        if leaf_id not in self._effective_deps:
            raise GraphError(
                "NOT_A_LEAF_NODE",
                f"node {leaf_id!r} is not an executable leaf node",
                node_id=leaf_id,
            )
        return self._effective_deps[leaf_id]

    def dependents_of(self, leaf_id: str) -> tuple[str, ...]:
        self.get_node(leaf_id)
        if leaf_id not in self._effective_deps:
            raise GraphError(
                "NOT_A_LEAF_NODE",
                f"node {leaf_id!r} is not an executable leaf node",
                node_id=leaf_id,
            )
        return tuple(
            candidate
            for candidate in self._leaf_ids
            if leaf_id in self._effective_deps[candidate]
        )

    def flatten(self) -> list[Node]:
        """Return executable leaf nodes in deterministic topological order."""
        return [self._nodes_by_id[node_id] for node_id in self._topological_order]

    def topological_ids(self) -> tuple[str, ...]:
        return self._topological_order

    def effective_signature(self, leaf_id: str) -> str:
        """Signature capturing the leaf node, its enclosing groups, and effective deps."""
        self.dependencies_of(leaf_id)
        chain = [*reversed(self._ancestor_ids(leaf_id)), leaf_id]
        payload = {
            "leaf_id": leaf_id,
            "effective_depends_on": list(self._effective_deps[leaf_id]),
            "chain": [
                {
                    "id": member_id,
                    "stage": self._nodes_by_id[member_id].stage,
                    "content_hash": self._nodes_by_id[member_id].content_hash,
                    "metadata": self._nodes_by_id[member_id].metadata,
                }
                for member_id in chain
            ],
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def fingerprint(self) -> str:
        """Deterministic SHA-256 fingerprint of the entire flattened DAG."""
        payload = {
            "graph_id": self.graph_id,
            "nodes": [
                {
                    "id": leaf_id,
                    "signature": self.effective_signature(leaf_id),
                }
                for leaf_id in self._topological_order
            ],
        }
        encoded = json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SUPPORTED_SCHEMA_VERSION,
            "graph_id": self.graph_id,
            "metadata": copy.deepcopy(self.metadata),
            "nodes": [node.to_dict() for node in self.roots],
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        require_schema_version: bool = False,
    ) -> ExecutionGraph:
        if not isinstance(data, dict):
            raise GraphError("INVALID_GRAPH_SPEC", "graph document must be a mapping")
        if "schema_version" in data:
            schema_version = data["schema_version"]
            if (
                type(schema_version) is not int
                or schema_version != SUPPORTED_SCHEMA_VERSION
            ):
                raise GraphError(
                    "UNSUPPORTED_SCHEMA_VERSION",
                    f"unsupported schema_version: {schema_version!r}",
                )
        elif require_schema_version:
            raise GraphError(
                "MISSING_SCHEMA_VERSION",
                "graph document must declare 'schema_version: 1'",
            )
        raw_nodes = data.get("nodes")
        if raw_nodes is None and isinstance(data.get("stages"), list):
            raw_nodes = data["stages"]
        if not isinstance(raw_nodes, list):
            raise GraphError(
                "INVALID_GRAPH_NODES",
                "graph specification must contain a 'nodes' or 'stages' list",
            )
        graph_id = data.get("graph_id", data.get("profile_id", "default"))
        metadata = data.get("metadata", {})
        if not isinstance(metadata, dict):
            raise GraphError("INVALID_GRAPH_METADATA", "graph metadata must be a mapping")
        nodes = [Node.from_dict(item) for item in raw_nodes]
        return cls(nodes, graph_id=graph_id, metadata=metadata)

    @classmethod
    def from_yaml(cls, path: str | Path) -> ExecutionGraph:
        try:
            raw_text = Path(path).read_text(encoding="utf-8")
        except OSError as error:
            raise GraphError("GRAPH_READ_FAILED", str(error)) from error
        try:
            document = _load_yaml_unique_keys(raw_text)
        except GraphError:
            raise
        except yaml.YAMLError as error:
            raise GraphError("GRAPH_YAML_ERROR", str(error)) from error
        if not isinstance(document, dict):
            raise GraphError("INVALID_GRAPH_SPEC", "YAML graph document must be a mapping")
        return cls.from_dict(document, require_schema_version=True)

from pathlib import Path

import pytest

from core.orchestrator.graph import ExecutionGraph, GraphError, Node


REPO_ROOT = Path(__file__).parents[2]


def test_nested_dag_parses_flattens_and_inherits_group_dependencies() -> None:
    graph = ExecutionGraph.from_dict(
        {
            "graph_id": "nested-sample",
            "nodes": [
                {"id": "intake", "stage": "intake"},
                {
                    "id": "adapt_group",
                    "stage": "group",
                    "depends_on": ["intake"],
                    "children": [
                        {"id": "scaffold", "stage": "scaffold"},
                        {
                            "id": "adapt",
                            "stage": "adapt",
                            "depends_on": ["scaffold"],
                        },
                    ],
                },
                {
                    "id": "verify",
                    "stage": "verify",
                    "depends_on": ["adapt_group"],
                },
            ],
        }
    )

    assert graph.topological_ids() == ("intake", "scaffold", "adapt", "verify")
    assert [node.id for node in graph.flatten()] == [
        "intake",
        "scaffold",
        "adapt",
        "verify",
    ]
    assert graph.dependencies_of("scaffold") == ("intake",)
    assert graph.dependencies_of("adapt") == ("intake", "scaffold")
    assert graph.dependencies_of("verify") == ("scaffold", "adapt")
    assert graph.parent_of("scaffold") == "adapt_group"
    assert graph.parent_of("intake") is None


def test_duplicate_node_id_across_nesting_levels_is_rejected() -> None:
    with pytest.raises(GraphError) as raised:
        ExecutionGraph(
            [
                Node(id="shared", stage="intake"),
                Node(
                    id="group",
                    stage="group",
                    children=(Node(id="shared", stage="anatomy"),),
                ),
            ]
        )

    assert raised.value.reason_code == "DUPLICATE_NODE_ID"
    assert raised.value.node_id == "shared"


def test_direct_and_indirect_cycles_are_detected() -> None:
    with pytest.raises(GraphError) as self_cycle:
        Node(id="a", stage="intake", depends_on=("a",))
    assert self_cycle.value.reason_code == "SELF_DEPENDENCY"

    with pytest.raises(GraphError) as cycle:
        ExecutionGraph(
            [
                Node(id="a", stage="intake", depends_on=("c",)),
                Node(id="b", stage="anatomy", depends_on=("a",)),
                Node(id="c", stage="scaffold", depends_on=("b",)),
            ]
        )
    assert cycle.value.reason_code == "CYCLE_DETECTED"


def test_group_cycle_and_unknown_dependency_are_rejected() -> None:
    with pytest.raises(GraphError) as unknown:
        ExecutionGraph([Node(id="a", stage="intake", depends_on=("missing",))])
    assert unknown.value.reason_code == "UNKNOWN_DEPENDENCY"

    with pytest.raises(GraphError) as group_cycle:
        ExecutionGraph(
            [
                Node(
                    id="g1",
                    stage="group",
                    depends_on=("g2",),
                    children=(Node(id="a", stage="intake"),),
                ),
                Node(
                    id="g2",
                    stage="group",
                    depends_on=("g1",),
                    children=(Node(id="b", stage="anatomy"),),
                ),
            ]
        )
    assert group_cycle.value.reason_code == "CYCLE_DETECTED"


def test_default_loongforge_embodied_profile_loads_as_valid_6_stage_dag() -> None:
    profile_path = REPO_ROOT / "core" / "profiles" / "loongforge_embodied.yaml"
    graph = ExecutionGraph.from_yaml(profile_path)

    assert graph.graph_id == "loongforge_embodied"
    assert graph.topological_ids() == (
        "intake",
        "anatomy",
        "scaffold",
        "adapt",
        "verify",
        "deliver",
    )


def test_invalid_stage_and_legacy_kind_are_rejected() -> None:
    with pytest.raises(GraphError) as bad_stage:
        Node(id="typo", stage="intak")
    assert bad_stage.value.reason_code == "INVALID_STAGE"

    with pytest.raises(GraphError) as legacy_kind:
        ExecutionGraph.from_dict({"nodes": [{"id": "n1", "kind": "intake"}]})
    assert legacy_kind.value.reason_code == "INVALID_STAGE"

    with pytest.raises(GraphError) as unknown_key:
        ExecutionGraph.from_dict(
            {
                "nodes": [
                    {"id": "n1", "stage": "intake", "depend_on": ["other"]},
                ]
            }
        )
    assert unknown_key.value.reason_code == "UNKNOWN_NODE_KEY"


def test_yaml_loader_rejects_duplicate_keys_and_bad_schema_version(
    tmp_path: Path,
) -> None:
    dup_yaml = tmp_path / "dup.yaml"
    dup_yaml.write_text(
        "schema_version: 1\nprofile_id: first\nprofile_id: second\n"
        "stages:\n  - id: intake\n    stage: intake\n",
        encoding="utf-8",
    )
    with pytest.raises(GraphError) as dup_err:
        ExecutionGraph.from_yaml(dup_yaml)
    assert dup_err.value.reason_code == "DUPLICATE_YAML_KEY"

    missing_ver = tmp_path / "missing_ver.yaml"
    missing_ver.write_text(
        "profile_id: test\nstages:\n  - id: intake\n    stage: intake\n",
        encoding="utf-8",
    )
    with pytest.raises(GraphError) as missing_err:
        ExecutionGraph.from_yaml(missing_ver)
    assert missing_err.value.reason_code == "MISSING_SCHEMA_VERSION"

    bad_ver = tmp_path / "bad_ver.yaml"
    bad_ver.write_text(
        "schema_version: 99\nprofile_id: test\n"
        "stages:\n  - id: intake\n    stage: intake\n",
        encoding="utf-8",
    )
    with pytest.raises(GraphError) as ver_err:
        ExecutionGraph.from_yaml(bad_ver)
    assert ver_err.value.reason_code == "UNSUPPORTED_SCHEMA_VERSION"


def test_graph_edge_error_codes_empty_duplicate_dep_ancestor_and_unknown_node() -> None:
    with pytest.raises(GraphError) as empty_err:
        ExecutionGraph([])
    assert empty_err.value.reason_code == "EMPTY_GRAPH"

    with pytest.raises(GraphError) as dup_dep:
        Node(id="b", stage="anatomy", depends_on=("a", "a"))
    assert dup_dep.value.reason_code == "DUPLICATE_DEPENDENCY"

    with pytest.raises(GraphError) as anc_err:
        ExecutionGraph(
            [
                Node(
                    id="g",
                    stage="group",
                    children=(Node(id="child", stage="intake", depends_on=("g",)),),
                )
            ]
        )
    assert anc_err.value.reason_code == "ANCESTOR_DEPENDENCY"

    with pytest.raises(GraphError) as desc_err:
        ExecutionGraph(
            [
                Node(
                    id="g",
                    stage="group",
                    depends_on=("child",),
                    children=(Node(id="child", stage="intake"),),
                )
            ]
        )
    assert desc_err.value.reason_code == "DESCENDANT_DEPENDENCY"

    graph = ExecutionGraph(
        [
            Node(
                id="g",
                stage="group",
                children=(Node(id="leaf", stage="intake"),),
            )
        ]
    )
    with pytest.raises(GraphError) as unknown_err:
        graph.dependencies_of("nonexistent")
    assert unknown_err.value.reason_code == "UNKNOWN_NODE"

    with pytest.raises(GraphError) as not_leaf:
        graph.dependencies_of("g")
    assert not_leaf.value.reason_code == "NOT_A_LEAF_NODE"

import json
from pathlib import Path

from core.contracts import export_json_schemas


def test_json_schemas_are_exported_for_every_public_contract(
    tmp_path: Path,
) -> None:
    exported_paths = export_json_schemas(tmp_path)

    assert {path.name for path in exported_paths} == {
        "artifact-manifest.schema.json",
        "f1-task-pack.schema.json",
        "intake-task-pack-v2.schema.json",
        "model-analysis-v2.schema.json",
        "skill-handoff-request-v2.schema.json",
        "skill-handoff-receipt-v2.schema.json",
        "f2-behavior-contract.schema.json",
        "gate-decision.schema.json",
        "gate-result.schema.json",
        "handoff-result.schema.json",
        "stage-spec.schema.json",
    }

    f1_schema = json.loads(
        (tmp_path / "f1-task-pack.schema.json").read_text(encoding="utf-8")
    )
    assert f1_schema["title"] == "F1TaskPack"
    assert set(f1_schema["required"]) >= {
        "schema_version",
        "source",
        "target",
        "environment",
        "entrypoint",
        "assets",
        "scope",
    }

    stage_schema = json.loads(
        (tmp_path / "stage-spec.schema.json").read_text(encoding="utf-8")
    )
    assert stage_schema["title"] == "StageSpec"
    gate_result_schema = json.loads(
        (tmp_path / "gate-result.schema.json").read_text(encoding="utf-8")
    )
    assert gate_result_schema["title"] == "GateResult"
    gate_decision_schema = json.loads(
        (tmp_path / "gate-decision.schema.json").read_text(encoding="utf-8")
    )
    assert gate_decision_schema["title"] == "GateDecision"

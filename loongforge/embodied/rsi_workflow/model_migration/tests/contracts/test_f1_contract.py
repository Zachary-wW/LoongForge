from pathlib import Path

import pytest
import yaml

from core.contracts import (
    ContractParseError,
    ContractValidationError,
    load_f1_task_pack,
)


FIXTURES = Path(__file__).parents[1] / "fixtures" / "contracts"


def test_valid_f1_task_pack_loads_as_strict_model() -> None:
    task_pack = load_f1_task_pack(FIXTURES / "f1-task-pack.valid.yaml")

    assert task_pack.schema_version == 1
    assert task_pack.source.model_variant == "pi05"
    assert task_pack.source.commit == "7f3c2a9d4e5b6f70128394aa65b7c8d9e0f1a2b3"
    assert task_pack.scope.asset_mode == "manifest_only"
    assert task_pack.assets[0].id == "checkpoint"


def test_placeholder_commit_is_rejected_with_machine_readable_error(
    tmp_path: Path,
) -> None:
    document = yaml.safe_load(
        (FIXTURES / "f1-task-pack.valid.yaml").read_text(encoding="utf-8")
    )
    document["source"]["commit"] = "REPLACE_WITH_SHA"
    request_path = tmp_path / "f1-task-pack.invalid.yaml"
    request_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ContractValidationError) as raised:
        load_f1_task_pack(request_path)

    error = raised.value
    assert error.reason_code == "CONTRACT_VALIDATION_FAILED"
    assert error.to_dict()["errors"][0]["loc"] == ["source", "commit"]


def test_duplicate_yaml_keys_are_rejected(tmp_path: Path) -> None:
    source = (FIXTURES / "f1-task-pack.valid.yaml").read_text(encoding="utf-8")
    request_path = tmp_path / "f1-task-pack.duplicate-key.yaml"
    request_path.write_text(
        source.replace("schema_version: 1", "schema_version: 1\nschema_version: 1", 1),
        encoding="utf-8",
    )

    with pytest.raises(ContractParseError) as raised:
        load_f1_task_pack(request_path)

    assert raised.value.reason_code == "DUPLICATE_YAML_KEY"
    assert raised.value.to_dict()["path"] == "schema_version"


def test_duplicate_asset_ids_are_rejected(tmp_path: Path) -> None:
    source = (FIXTURES / "f1-task-pack.valid.yaml").read_text(encoding="utf-8")
    document = yaml.safe_load(source)
    document["assets"].append(dict(document["assets"][0]))
    invalid_path = tmp_path / "f1-task-pack.duplicate-asset.yaml"
    invalid_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ContractValidationError) as raised:
        load_f1_task_pack(invalid_path)

    assert "asset ids must be unique" in (raised.value.to_dict()["errors"][0]["msg"])


def test_entrypoint_path_cannot_escape_the_source_root(tmp_path: Path) -> None:
    source = (FIXTURES / "f1-task-pack.valid.yaml").read_text(encoding="utf-8")
    document = yaml.safe_load(source)
    document["entrypoint"]["path"] = "../outside.py"
    invalid_path = tmp_path / "f1-task-pack.escaping-entrypoint.yaml"
    invalid_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ContractValidationError) as raised:
        load_f1_task_pack(invalid_path)

    assert raised.value.to_dict()["errors"][0]["loc"] == ["entrypoint", "path"]
    assert "relative and normalized" in (raised.value.to_dict()["errors"][0]["msg"])


def test_missing_contract_file_returns_structured_error(tmp_path: Path) -> None:
    missing_path = tmp_path / "missing.yaml"

    with pytest.raises(ContractParseError) as raised:
        load_f1_task_pack(missing_path)

    assert raised.value.reason_code == "CONTRACT_READ_FAILED"
    assert raised.value.path == str(missing_path)


def test_unknown_asset_kind_is_rejected(tmp_path: Path) -> None:
    source = (FIXTURES / "f1-task-pack.valid.yaml").read_text(encoding="utf-8")
    document = yaml.safe_load(source)
    document["assets"][0]["kind"] = "weights"
    invalid_path = tmp_path / "f1-task-pack.unknown-asset-kind.yaml"
    invalid_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ContractValidationError) as raised:
        load_f1_task_pack(invalid_path)

    assert raised.value.to_dict()["errors"][0]["loc"] == ["assets", 0, "kind"]

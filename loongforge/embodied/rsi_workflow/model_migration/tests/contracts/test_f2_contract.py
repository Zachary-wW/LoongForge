from pathlib import Path

import pytest
import yaml

from core.contracts import ContractValidationError, load_f2_behavior_contract


FIXTURES = Path(__file__).parents[1] / "fixtures" / "contracts"


def test_valid_f2_contract_binds_the_frozen_f1_package() -> None:
    contract = load_f2_behavior_contract(FIXTURES / "f2-behavior-contract.valid.yaml")

    assert contract.case.id == "pi05-libero-step-1"
    assert contract.case.f1_manifest_sha256 == (
        "13579bdf02468ace13579bdf02468ace13579bdf02468ace13579bdf02468ace"
    )
    assert contract.validator.id == "f2-two-attempt-comparator"
    assert [phase.id for phase in contract.phases][-1] == "optimizer_step"


def test_duplicate_phase_ids_are_rejected(tmp_path: Path) -> None:
    contract_path = FIXTURES / "f2-behavior-contract.valid.yaml"
    document = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    document["phases"].append({"id": "loss", "required": True})
    invalid_path = tmp_path / "f2.behavior-contract.duplicate-phase.yaml"
    invalid_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ContractValidationError) as raised:
        load_f2_behavior_contract(invalid_path)

    error = raised.value
    assert error.reason_code == "CONTRACT_VALIDATION_FAILED"
    assert "phase ids must be unique" in error.to_dict()["errors"][0]["msg"]


def test_duplicate_assertion_ids_are_rejected(tmp_path: Path) -> None:
    contract_path = FIXTURES / "f2-behavior-contract.valid.yaml"
    document = yaml.safe_load(contract_path.read_text(encoding="utf-8"))
    document["assertions"].append(dict(document["assertions"][0]))
    invalid_path = tmp_path / "f2.behavior-contract.duplicate-assertion.yaml"
    invalid_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ContractValidationError) as raised:
        load_f2_behavior_contract(invalid_path)

    assert (
        "assertion ids must be unique" in (raised.value.to_dict()["errors"][0]["msg"])
    )

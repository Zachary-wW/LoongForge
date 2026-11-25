from pathlib import Path

import pytest
import yaml

from core.contracts import ContractValidationError, load_handoff_result


FIXTURES = Path(__file__).parents[1] / "fixtures" / "contracts"


def test_completed_handoff_result_binds_request_scope_and_validator() -> None:
    result = load_handoff_result(FIXTURES / "handoff-result.valid.yaml")

    assert result.stage == "f1"
    assert result.execution_status == "completed"
    assert result.observed_scope == ["static", "startup_probe"]
    assert result.validator_verdict is not None
    assert result.validator_verdict.path == "evidence/validate/verdict.json"


def test_completed_handoff_without_validator_verdict_is_rejected(
    tmp_path: Path,
) -> None:
    source = (FIXTURES / "handoff-result.valid.yaml").read_text(encoding="utf-8")
    document = yaml.safe_load(source)
    document["validator_verdict"] = None
    invalid_path = tmp_path / "handoff-result.missing-verdict.yaml"
    invalid_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ContractValidationError) as raised:
        load_handoff_result(invalid_path)

    assert (
        "completed handoff requires a validator verdict"
        in (raised.value.to_dict()["errors"][0]["msg"])
    )


def test_blocked_handoff_must_state_a_reason(tmp_path: Path) -> None:
    source = (FIXTURES / "handoff-result.valid.yaml").read_text(encoding="utf-8")
    document = yaml.safe_load(source)
    document["execution_status"] = "blocked"
    document["validator_verdict"] = None
    document["blocked_reasons"] = []
    invalid_path = tmp_path / "handoff-result.blocked-without-reason.yaml"
    invalid_path.write_text(yaml.safe_dump(document), encoding="utf-8")

    with pytest.raises(ContractValidationError) as raised:
        load_handoff_result(invalid_path)

    assert (
        "blocked handoff requires at least one reason"
        in (raised.value.to_dict()["errors"][0]["msg"])
    )

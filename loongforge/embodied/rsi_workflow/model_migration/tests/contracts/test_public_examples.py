import hashlib
from pathlib import Path

import pytest

from core.contracts import ContractValidationError, load_f1_task_pack, load_f2_behavior_contract
from validate.anatomy_check import AnatomyChecker, REQUIRED_ASSERTION_IDS

ROOT = Path(__file__).parents[2]


def test_public_f1_example_strictly_loads() -> None:
    task = load_f1_task_pack(ROOT / "core/contracts/f1-task-pack.example.yaml")
    assert isinstance(task.assets, list)
    assert task.scope.startup_probe
    assert len(task.entrypoint.argv) > 0


def test_public_f2_example_is_schema_only_not_forged_validator_acceptance() -> None:
    contract = load_f2_behavior_contract(ROOT / "core/contracts/f2-behavior-contract.example.yaml")
    assert set(REQUIRED_ASSERTION_IDS) <= {x.id for x in contract.assertions if x.required}
    assert contract.validator.id == "schema-only:core.contracts.f2"
    assert contract.validator.sha256 == hashlib.sha256((ROOT / "core/contracts/f2.py").read_bytes()).hexdigest()
    assert contract.validator.sha256 != AnatomyChecker().validator_sha256()
    assert contract.blocked_reasons == ["EXAMPLE_NOT_EXECUTABLE"]


def test_public_example_does_not_relax_strict_model(tmp_path: Path) -> None:
    text = (ROOT / "core/contracts/f1-task-pack.example.yaml").read_text()
    path = tmp_path / "invalid.yaml"
    path.write_text(text + "unrecognized_field: true\n")
    with pytest.raises(ContractValidationError):
        load_f1_task_pack(path)

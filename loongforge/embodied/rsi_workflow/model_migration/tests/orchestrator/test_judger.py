import json
from pathlib import Path

import pytest

from core.contracts import (
    ContractValidationError,
    GateDecision as CoreGateDecision,
    GateResult as CoreGateResult,
    load_gate_decision,
    load_gate_result,
)
from core.orchestrator.judger import (
    GateBinding,
    GateDecision,
    GateEvidence,
    GateResult,
    decide,
)


COMMIT_A = "7f3c2a9d4e5b6f70128394aa65b7c8d9e0f1a2b3"
COMMIT_B = "1a2b3c4d5e6f708192a3b4c5d6e7f8091a2b3c4d"
MANIFEST_A = "13579bdf02468ace13579bdf02468ace13579bdf02468ace13579bdf02468ace"
MANIFEST_B = "2468ace013579bdf2468ace013579bdf2468ace013579bdf2468ace013579bdf"
VAL_SHA_1 = "abcdef0123456789abcdef0123456789abcdef0123456789abcdef0123456789"
VAL_SHA_2 = "9876543210fedcba9876543210fedcba9876543210fedcba9876543210fedcba"


def _binding(
    *,
    commit: str = COMMIT_A,
    manifest_sha256: str = MANIFEST_A,
    validator_id: str = "validate.intake_check",
    validator_sha256: str = VAL_SHA_1,
) -> GateBinding:
    return GateBinding(
        commit=commit,
        manifest_sha256=manifest_sha256,
        validator_id=validator_id,
        validator_sha256=validator_sha256,
    )


def test_decide_returns_pass_when_all_gates_pass_and_bindings_match() -> None:
    results = [
        GateResult(
            gate_id="intake",
            check="intake",
            status="pass",
            binding=_binding(validator_id="v.intake", validator_sha256=VAL_SHA_1),
            summary="intake verified",
        ),
        GateResult(
            gate_id="anatomy",
            check="anatomy",
            status="pass",
            binding=_binding(validator_id="v.anatomy", validator_sha256=VAL_SHA_2),
            summary="anatomy verified",
        ),
    ]

    decision = decide(
        results,
        expected_commit=COMMIT_A,
        expected_manifest_sha256=MANIFEST_A,
        trusted_validators={"v.intake": VAL_SHA_1, "v.anatomy": VAL_SHA_2},
        required_gates=("intake", "anatomy"),
    )

    assert decision.status == "pass"
    assert decision.failing_gates == ()
    assert decision.reason_codes == ()
    assert decision.first_mismatch is None


def test_decide_precedence_error_gt_blocked_gt_repairable_gt_pass() -> None:
    repairable = GateResult(
        gate_id="parity",
        check="parity",
        status="repairable",
        binding=_binding(validator_id="v.parity", validator_sha256=VAL_SHA_1),
        reason_codes=("LOSS_DRIFT",),
        first_mismatch="forward.action_head",
        summary="action head diverged",
    )
    blocked = GateResult(
        gate_id="task_eval",
        check="task_eval",
        status="blocked",
        binding=_binding(validator_id="v.eval", validator_sha256=VAL_SHA_2),
        reason_codes=("WAITING_ASSET",),
        summary="simulator license missing",
    )
    err = GateResult(
        gate_id="train_resume",
        check="train_resume",
        status="error",
        binding=_binding(validator_id="v.train", validator_sha256=VAL_SHA_1),
        reason_codes=("VALIDATOR_CRASH",),
        summary="checker raised unhandled error",
    )

    d_repairable = decide(
        [repairable],
        expected_commit=COMMIT_A,
        expected_manifest_sha256=MANIFEST_A,
    )
    assert d_repairable.status == "repairable"
    assert d_repairable.first_mismatch == "forward.action_head"
    assert d_repairable.reason_codes == ("LOSS_DRIFT",)

    d_blocked = decide(
        [repairable, blocked],
        expected_commit=COMMIT_A,
        expected_manifest_sha256=MANIFEST_A,
    )
    assert d_blocked.status == "blocked"
    assert d_blocked.failing_gates == ("parity", "task_eval")
    assert d_blocked.reason_codes == ("LOSS_DRIFT", "WAITING_ASSET")

    d_error = decide(
        [repairable, blocked, err],
        expected_commit=COMMIT_A,
        expected_manifest_sha256=MANIFEST_A,
    )
    assert d_error.status == "error"
    assert "VALIDATOR_CRASH" in d_error.reason_codes


def test_decide_fails_closed_on_commit_manifest_or_validator_sha_mismatch() -> None:
    bad_commit = GateResult(
        gate_id="intake",
        check="intake",
        status="pass",
        binding=_binding(commit=COMMIT_B),
        summary="stale commit pass",
    )
    d1 = decide(
        [bad_commit],
        expected_commit=COMMIT_A,
        expected_manifest_sha256=MANIFEST_A,
    )
    assert d1.status == "error"
    assert "COMMIT_BINDING_MISMATCH" in d1.reason_codes

    bad_manifest = GateResult(
        gate_id="intake",
        check="intake",
        status="pass",
        binding=_binding(manifest_sha256=MANIFEST_B),
        summary="stale manifest pass",
    )
    d2 = decide(
        [bad_manifest],
        expected_commit=COMMIT_A,
        expected_manifest_sha256=MANIFEST_A,
    )
    assert d2.status == "error"
    assert "MANIFEST_BINDING_MISMATCH" in d2.reason_codes

    bad_validator = GateResult(
        gate_id="intake",
        check="intake",
        status="pass",
        binding=_binding(validator_id="v.intake", validator_sha256=VAL_SHA_2),
        summary="tampered validator pass",
    )
    d3 = decide(
        [bad_validator],
        expected_commit=COMMIT_A,
        expected_manifest_sha256=MANIFEST_A,
        trusted_validators={"v.intake": VAL_SHA_1},
    )
    assert d3.status == "error"
    assert "VALIDATOR_SHA_MISMATCH" in d3.reason_codes


def test_decide_fails_closed_on_empty_or_missing_required_gates() -> None:
    empty_decision = decide(
        [],
        expected_commit=COMMIT_A,
        expected_manifest_sha256=MANIFEST_A,
    )
    assert empty_decision.status == "error"
    assert "EMPTY_GATE_RESULTS" in empty_decision.reason_codes

    partial = GateResult(
        gate_id="intake",
        check="intake",
        status="pass",
        binding=_binding(),
        summary="intake ok",
    )
    missing_decision = decide(
        [partial],
        expected_commit=COMMIT_A,
        expected_manifest_sha256=MANIFEST_A,
        required_gates=("intake", "anatomy"),
    )
    assert missing_decision.status == "error"
    assert "MISSING_REQUIRED_GATE" in missing_decision.reason_codes
    assert "anatomy" in missing_decision.failing_gates


def test_decide_rejects_duplicate_gate_ids_untrusted_validators_and_bad_expected_ids() -> None:
    g1 = GateResult(
        gate_id="intake",
        check="intake",
        status="pass",
        binding=_binding(validator_id="v.intake", validator_sha256=VAL_SHA_1),
        summary="first pass",
    )
    g2 = GateResult(
        gate_id="intake",
        check="intake",
        status="pass",
        binding=_binding(validator_id="v.intake", validator_sha256=VAL_SHA_1),
        summary="duplicate gate id",
    )
    dup = decide(
        [g1, g2],
        expected_commit=COMMIT_A,
        expected_manifest_sha256=MANIFEST_A,
        trusted_validators={"v.intake": VAL_SHA_1},
    )
    assert dup.status == "error"
    assert "DUPLICATE_GATE_ID" in dup.reason_codes

    untrusted = decide(
        [g1],
        expected_commit=COMMIT_A,
        expected_manifest_sha256=MANIFEST_A,
        trusted_validators={"v.other": VAL_SHA_1},
    )
    assert untrusted.status == "error"
    assert "UNTRUSTED_VALIDATOR" in untrusted.reason_codes

    bad_expected = decide(
        [g1],
        expected_commit="0" * 40,
        expected_manifest_sha256="0" * 64,
    )
    assert bad_expected.status == "error"
    assert "INVALID_EXPECTED_COMMIT" in bad_expected.reason_codes
    assert "INVALID_EXPECTED_MANIFEST_SHA" in bad_expected.reason_codes


def test_decide_rejects_string_required_gates() -> None:
    g1 = GateResult(
        gate_id="intake",
        check="intake",
        status="pass",
        binding=_binding(),
        summary="ok",
    )
    with pytest.raises(TypeError):
        decide(
            [g1],
            expected_commit=COMMIT_A,
            expected_manifest_sha256=MANIFEST_A,
            required_gates="intake",  # type: ignore[arg-type]
        )


def test_engine_and_core_verdict_models_match_constraints_and_roundtrip_on_disk(
    tmp_path: Path,
) -> None:
    # Drift guards: invalid check, duplicate reason_codes, empty evidence detail rejected on both sides.
    with pytest.raises(ValueError):
        GateResult(
            gate_id="bad",
            check="not_a_real_check",  # type: ignore[arg-type]
            status="pass",
            binding=_binding(),
            summary="bad check",
        )
    with pytest.raises(ValueError):
        GateResult(
            gate_id="dup",
            check="intake",
            status="repairable",
            binding=_binding(),
            reason_codes=("DRIFT", "DRIFT"),
            summary="duplicate reasons",
        )
    with pytest.raises(ValueError):
        GateEvidence(key="k", detail="   ")

    result = GateResult(
        gate_id="parity",
        check="parity",
        status="repairable",
        binding=_binding(validator_id="v.parity", validator_sha256=VAL_SHA_1),
        reason_codes=("LOSS_DRIFT",),
        first_mismatch="forward.action_head",
        summary="action head diverged",
        evidence=(
            GateEvidence(
                key="max_abs_diff",
                detail="0.012 > 1e-3",
                path="evidence/parity.json",
                digest=MANIFEST_A,
            ),
        ),
        metrics={"max_abs_diff": 0.012},
    )
    decision = decide(
        [result],
        expected_commit=COMMIT_A,
        expected_manifest_sha256=MANIFEST_A,
        trusted_validators={"v.parity": VAL_SHA_1},
        required_gates=("parity",),
    )

    # Round-trip to disk and validate via strict core.contracts loaders AND engine from_dict.
    result_path = tmp_path / "gate_result.json"
    decision_path = tmp_path / "gate_decision.json"
    result_path.write_text(
        json.dumps(result.to_dict(), indent=2), encoding="utf-8"
    )
    decision_path.write_text(
        json.dumps(decision.to_dict(), indent=2), encoding="utf-8"
    )

    loaded_core_result = load_gate_result(result_path)
    loaded_core_decision = load_gate_decision(decision_path)
    assert isinstance(loaded_core_result, CoreGateResult)
    assert isinstance(loaded_core_decision, CoreGateDecision)
    assert loaded_core_result.first_mismatch == "forward.action_head"
    assert loaded_core_decision.status == "repairable"

    reloaded_engine_result = GateResult.from_dict(
        loaded_core_result.model_dump(mode="json")
    )
    reloaded_engine_decision = GateDecision.from_dict(
        loaded_core_decision.model_dump(mode="json")
    )
    assert reloaded_engine_result == result
    assert reloaded_engine_decision == decision

    # Negative: tampered/invalid check on disk is rejected by strict loader.
    tampered = result.to_dict()
    tampered["check"] = "unknown_check"
    bad_path = tmp_path / "bad_result.json"
    bad_path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ContractValidationError):
        load_gate_result(bad_path)

from __future__ import annotations

import pytest

from core.stages.f2_anatomy.adapters.fake import FakeStepAdapter, FakeStepConfig
from core.stages.f2_anatomy.compare import compare_attempts, diff_weight_snapshots


def test_compare_attempts_passes_for_identical_clean_attempts() -> None:
    adapter = FakeStepAdapter()
    a1 = adapter.execute_step(FakeStepConfig(seed=7, world_size=2))
    a2 = adapter.execute_step(FakeStepConfig(seed=7, world_size=2))

    report = compare_attempts(a1, a2)
    assert report.status == "pass"
    assert report.first_mismatch is None
    assert report.reason_codes == ()


@pytest.mark.parametrize(
    ("injection", "expected_reason", "expected_mismatch_prefix"),
    [
        ("skip_forward", "FORWARD_CALL_SKIPPED", "forward.call_count"),
        ("swap_cameras", "CAMERA_ORDER_MISMATCH", "input.cameras"),
        ("swap_samples", "SAMPLE_ORDER_MISMATCH", "input.sample_ids"),
        ("swap_rng_consumers", "RNG_CONSUMER_MISMATCH", "rng.events"),
        ("step_counter_only", "NO_REAL_PARAMETER_UPDATE", "optimizer.weight_update"),
        ("missing_rank_evidence", "MISSING_RANK_EVIDENCE", "distributed.ranks"),
    ],
)
def test_all_six_negative_injections_pinpoint_exact_first_mismatch(
    injection: str,
    expected_reason: str,
    expected_mismatch_prefix: str,
) -> None:
    adapter = FakeStepAdapter()
    world_size = 2 if injection == "missing_rank_evidence" else 1
    baseline = adapter.execute_step(FakeStepConfig(seed=11, world_size=world_size))
    injected = adapter.execute_step(
        FakeStepConfig(seed=11, world_size=world_size, inject=injection)
    )

    report = compare_attempts(baseline, injected)
    assert report.status == "fail"
    assert expected_reason in report.reason_codes
    assert report.first_mismatch is not None
    assert report.first_mismatch.startswith(expected_mismatch_prefix)
    assert report.expected is not None
    assert report.actual is not None
    assert report.evidence_path is not None


def test_compare_marks_nondeterministic_numerical_drift_explicitly_never_pass() -> None:
    adapter = FakeStepAdapter()
    a1 = adapter.execute_step(FakeStepConfig(seed=13))
    a2 = adapter.execute_step(FakeStepConfig(seed=13))
    # Simulate subtle numerical drift on loss/weights with identical exact metadata.
    a2["loss"] = float(a1["loss"]) + 1e-4
    a2["weights_after"]["action_head.weight"]["sha256"] = "9" * 64
    a2["weights_after_digest"] = "8" * 64

    report = compare_attempts(a1, a2)
    assert report.status == "non_deterministic"
    assert "NON_DETERMINISTIC_ENDPOINT" in report.reason_codes
    assert report.first_mismatch is not None


def test_hierarchical_weight_probe_diff_prioritizes_name_shape_dtype_value() -> None:
    base = {
        "encoder.w": {"kind": "P", "shape": [2, 2], "dtype": "float32", "sha256": "a" * 64},
        "head.w": {"kind": "P", "shape": [2, 1], "dtype": "float32", "sha256": "b" * 64},
    }
    # Name difference comes first
    missing = {"encoder.w": base["encoder.w"]}
    d_name = diff_weight_snapshots(base, missing)
    assert d_name["identical"] is False
    assert d_name["first_category"] == "name"
    assert d_name["only_a"] == ["head.w"]

    # Shape difference
    bad_shape = {
        "encoder.w": base["encoder.w"],
        "head.w": {"kind": "P", "shape": [4, 1], "dtype": "float32", "sha256": "b" * 64},
    }
    d_shape = diff_weight_snapshots(base, bad_shape)
    assert d_shape["first_category"] == "shape"

    # Dtype difference
    bad_dtype = {
        "encoder.w": base["encoder.w"],
        "head.w": {"kind": "P", "shape": [2, 1], "dtype": "bfloat16", "sha256": "b" * 64},
    }
    d_dtype = diff_weight_snapshots(base, bad_dtype)
    assert d_dtype["first_category"] == "dtype"

    # Value difference
    bad_val = {
        "encoder.w": base["encoder.w"],
        "head.w": {"kind": "P", "shape": [2, 1], "dtype": "float32", "sha256": "c" * 64},
    }
    d_val = diff_weight_snapshots(base, bad_val)
    assert d_val["first_category"] == "value"


def test_compare_rejects_structural_drift_as_fail_and_common_missing_evidence_and_metadata_drift() -> None:
    adapter = FakeStepAdapter()
    a1 = adapter.execute_step(FakeStepConfig(seed=31))
    a2 = adapter.execute_step(FakeStepConfig(seed=31))

    # P2-2: post-step weight shape/dtype drift must be `fail` (NOT `non_deterministic`)
    a2_shape = dict(a2)
    a2_shape["weights_after"] = {
        **a2["weights_after"],
        "action_head.weight": {**a2["weights_after"]["action_head.weight"], "shape": [4, 2]},
    }
    r_shape = compare_attempts(a1, a2_shape)
    assert r_shape.status == "fail"
    assert "WEIGHT_STRUCTURE_MISMATCH" in r_shape.reason_codes

    a2_dtype = dict(a2)
    a2_dtype["weights_after"] = {
        **a2["weights_after"],
        "action_head.weight": {**a2["weights_after"]["action_head.weight"], "dtype": "bfloat16"},
    }
    r_dtype = compare_attempts(a1, a2_dtype)
    assert r_dtype.status == "fail"
    assert "WEIGHT_STRUCTURE_MISMATCH" in r_dtype.reason_codes

    # P1-1: attempt-2 step_before drift (10 -> 11 vs 0 -> 1) or label drift rejected
    a2_step = dict(a2)
    a2_step["step_before"] = 10
    a2_step["step_after"] = 11
    r_step = compare_attempts(a1, a2_step)
    assert r_step.status == "fail"
    assert "ATTEMPT_METADATA_MISMATCH" in r_step.reason_codes
    assert r_step.first_mismatch == "init.step_before"

    a2_env = dict(a2)
    a2_env["execution_environment"] = "real_gpu"
    r_env = compare_attempts(a1, a2_env)
    assert r_env.status == "fail"
    assert "ATTEMPT_METADATA_MISMATCH" in r_env.reason_codes

    # P1-2 / P2-2: BOTH attempts missing cameras or phases simultaneously must FAIL (not pass)
    a1_no_cam = {**a1, "cameras": {}}
    a2_no_cam = {**a2, "cameras": {}}
    r_no_cam = compare_attempts(a1_no_cam, a2_no_cam)
    assert r_no_cam.status == "fail"
    assert "MISSING_CAMERA_EVIDENCE" in r_no_cam.reason_codes

    a1_no_phase = {**a1, "phases": []}
    a2_no_phase = {**a2, "phases": []}
    r_no_phase = compare_attempts(a1_no_phase, a2_no_phase)
    assert r_no_phase.status == "fail"
    assert "MISSING_PHASE_EVIDENCE" in r_no_phase.reason_codes

    # P1-1: deleting a SINGLE required camera role (`right_wrist_0_rgb`) on BOTH attempts fails!
    a1_drop_cam = {**a1, "cameras": {k: v for k, v in a1["cameras"].items() if k != "right_wrist_0_rgb"}}
    a2_drop_cam = {**a2, "cameras": {k: v for k, v in a2["cameras"].items() if k != "right_wrist_0_rgb"}}
    r_drop_cam = compare_attempts(a1_drop_cam, a2_drop_cam)
    assert r_drop_cam.status == "fail"
    assert "MISSING_CAMERA_ROLE" in r_drop_cam.reason_codes
    assert r_drop_cam.first_mismatch == "input.cameras.right_wrist_0_rgb"

    # P1-1: deleting a SINGLE required RNG consumer (`flow_time`) on BOTH attempts fails!
    a1_drop_rng = {**a1, "rng_events": [e for e in a1["rng_events"] if e["consumer"] != "flow_time"]}
    a2_drop_rng = {**a2, "rng_events": [e for e in a2["rng_events"] if e["consumer"] != "flow_time"]}
    r_drop_rng = compare_attempts(a1_drop_rng, a2_drop_rng)
    assert r_drop_rng.status == "fail"
    assert "MISSING_RNG_CONSUMER" in r_drop_rng.reason_codes
    assert r_drop_rng.first_mismatch == "rng.events.flow_time"

    # P2-1: setting `weights_after={}` on BOTH attempts cannot forge a real parameter update!
    a1_empty_after = {**a1, "weights_after": {}}
    a2_empty_after = {**a2, "weights_after": {}}
    r_empty_after = compare_attempts(a1_empty_after, a2_empty_after)
    assert r_empty_after.status == "fail"
    assert "MISSING_WEIGHTS_AFTER" in r_empty_after.reason_codes

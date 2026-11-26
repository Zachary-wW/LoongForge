from __future__ import annotations

from pathlib import Path

from core.stages.f2_anatomy.adapters.fake import (
    FAKE_NEGATIVE_INJECTIONS,
    FakeStepAdapter,
    FakeStepConfig,
)
from core.stages.f2_anatomy.adapters.openpi_jax import OpenPIJaxAdapter, OpenPIJaxConfig


def test_fake_adapter_produces_deterministic_step_with_mandatory_mock_labels() -> None:
    adapter = FakeStepAdapter()
    cfg = FakeStepConfig(seed=42, world_size=1)

    rec1 = adapter.execute_step(cfg)
    rec2 = adapter.execute_step(cfg)

    assert rec1["execution_environment"] == "cpu_mock"
    assert rec1["evidence_scope"] == "orchestration_only"
    assert rec1["step_before"] == 0
    assert rec1["step_after"] == 1
    assert rec1["forward_call_count"] == 1
    assert rec1["weights_before_digest"] != rec1["weights_after_digest"]
    assert rec1["weights_before"] == rec2["weights_before"]
    assert rec1["weights_after"] == rec2["weights_after"]
    assert rec1["loss"] == rec2["loss"]
    assert rec1["rng_events"] == rec2["rng_events"]
    assert set(FAKE_NEGATIVE_INJECTIONS) == {
        "skip_forward",
        "swap_cameras",
        "swap_samples",
        "swap_rng_consumers",
        "step_counter_only",
        "missing_rank_evidence",
        "swap_action_tokens",
        "swap_action_masks",
    }


def test_openpi_jax_adapter_preflight_never_fakes_pass_without_assets_or_gpu(
    tmp_path: Path,
) -> None:
    adapter = OpenPIJaxAdapter()
    preflight = adapter.preflight(
        OpenPIJaxConfig(
            upstream_repo=tmp_path / "openpi",
            checkpoint_path=tmp_path / "missing_ckpt",
            dataset_path=tmp_path / "missing_dataset",
            tokenizer_path=None,
            norm_stats_path=None,
            require_gpu=True,
        )
    )
    assert preflight.status in ("not_run", "waiting")
    assert "WAITING_ASSET" in preflight.reason_codes
    assert preflight.execution_environment == "openpi_jax"

    # Executing without ready assets/GPU must return structured not_run/WAITING_ASSET, never pass.
    result = adapter.execute_step(
        OpenPIJaxConfig(
            upstream_repo=tmp_path / "openpi",
            checkpoint_path=tmp_path / "missing_ckpt",
            dataset_path=tmp_path / "missing_dataset",
            require_gpu=True,
        )
    )
    assert result["status"] == "not_run"
    assert "WAITING_ASSET" in result["reason_codes"]

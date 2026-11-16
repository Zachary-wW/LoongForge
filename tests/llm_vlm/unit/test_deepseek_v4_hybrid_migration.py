"""Unit coverage for the DeepSeek-V4 HybridModel migration contract."""

import sys
from pathlib import Path

import pytest

from loongforge.models.foundation.deepseek_v4.deepseek_v4_layer_spec import (
    build_deepseek_v4_hybrid_layer_pattern,
    expand_deepseek_v4_hybrid_compress_ratios,
)

TOOLS_DIR = Path(__file__).resolve().parents[3] / "tools"
sys.path.insert(0, str(TOOLS_DIR))
from mcore_checkpoint_convert.mcore.mcore_checkpoint import McoreCheckpoint  # noqa: E402


def test_builds_native_pattern_and_expands_physical_ratios():
    pattern = build_deepseek_v4_hybrid_layer_pattern(
        csa_compress_ratios=[0, 0, 4, 128, 0],
        num_layers=4,
        mtp_num_layers=1,
    )

    assert pattern == "WEWECEHE/WE"
    assert expand_deepseek_v4_hybrid_compress_ratios(
        pattern, [0, 0, 4, 128, 0]
    ) == [0, 0, 0, 0, 4, 0, 128, 0, 0, 0]


def test_honors_per_layer_dense_and_moe_frequency():
    pattern = build_deepseek_v4_hybrid_layer_pattern(
        csa_compress_ratios=[0, 4, 128],
        num_layers=3,
        moe_layer_freq=[1, 0, 1],
    )

    assert pattern == "WEC-HE"


@pytest.mark.parametrize(
    ("ratios", "message"),
    [
        ([0, 8, 128], "Unsupported DeepSeek-V4 compress ratio 8"),
        ([0, 4], "must equal num_layers"),
    ],
)
def test_rejects_invalid_pattern_inputs(ratios, message):
    with pytest.raises(ValueError, match=message):
        build_deepseek_v4_hybrid_layer_pattern(
            csa_compress_ratios=ratios,
            num_layers=3,
        )


def test_maps_logical_v4_layer_to_attention_and_ffn_physical_layers():
    checkpoint = object.__new__(McoreCheckpoint)
    checkpoint.dsv4_hybrid_has_mhc = True

    assert checkpoint._resolve_dsv4_hybrid_main_layer("input_layernorm", 2) == (
        4,
        "inner_layer",
    )
    assert checkpoint._resolve_dsv4_hybrid_main_layer("moe.gate", 2) == (
        5,
        "inner_layer",
    )
    assert checkpoint._resolve_dsv4_hybrid_main_layer("hc_attn_fn", 2) == (4, None)
    assert checkpoint._resolve_dsv4_hybrid_main_layer("hc_ffn_fn", 2) == (5, None)

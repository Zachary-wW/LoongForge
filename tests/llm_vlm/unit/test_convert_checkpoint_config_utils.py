# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
# Purpose: Verify VLM checkpoint path scoping.

"""Checks for VLM checkpoint conversion configuration."""

import sys
from pathlib import Path
from types import SimpleNamespace
import torch

# tests/llm_vlm/unit -> tests/llm_vlm -> tests -> repo root, then repo-root tools/
sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))

from mcore_checkpoint_convert.common.common_config import CommonConfig
from mcore_checkpoint_convert.utils.config_utils import convert_vlm_config
from mcore_bridge.core.parser import Parser
from mcore_bridge.checkpoint import hf_checkpoint_loader


def test_vlm_conversion_accepts_root_relative_paths():
    """Root-relative, legacy, and scoped paths must all map to the foundation."""
    config = CommonConfig()
    config.update({
        "name_map": {
            "huggingface": {"word_embeddings": "model.embed_tokens"},
            "mcore": {
                "layer_prefix": "language_model.decoder.layers",
                "word_embeddings": "embedding.word_embeddings",
                "vision_word_embeddings": None,
                "final_layernorm": "foundation_model.decoder.final_layernorm",
                "word_embeddings_for_head": "output_layer",
            },
        },
    })

    name_map = convert_vlm_config(config, for_vlm=True).get("name_map")["mcore"]

    assert name_map["layer_prefix"] == "foundation_model.decoder.layers"
    assert name_map["word_embeddings"] == "foundation_model.embedding.word_embeddings"
    assert name_map["vision_word_embeddings"] == "encoder_model.text_encoder.word_embeddings"
    assert (
        config.get("name_map")["huggingface"]["vision_word_embeddings"]
        == "model.embed_tokens"
    )
    assert name_map["final_layernorm"] == "foundation_model.decoder.final_layernorm"
    assert name_map["word_embeddings_for_head"] == "foundation_model.output_layer"


def test_online_vlm_sharding_uses_resolved_encoder_tp(monkeypatch):
    repo_root = Path(__file__).resolve().parents[3]
    monkeypatch.setenv("LOONGFORGE_PATH", str(repo_root))
    args = SimpleNamespace(
        model_name=None,
        config_file=str(repo_root / "configs/models/qwen2.5vl/qwen2_5_vl_7b.yaml"),
        tensor_model_parallel_size=2,
        encoder_tensor_model_parallel_size=None,
        num_experts=None,
    )
    parser = Parser(args)
    model_config = SimpleNamespace(
        image_encoder=SimpleNamespace(tensor_model_parallel_size=1),
    )

    parallel_config = parser.get_parallel_config(model_config=model_config)

    assert parser.type == "vlm"
    assert parallel_config.tp_size == 2
    assert parallel_config.encoder_tp_size == 1
    assert parallel_config.hf_checkpoint_device == "cpu"
    assert parser.get_parallel_config().encoder_tp_size is None


def test_checkpoint_diagnostics_sample_only_small_parameter_views(monkeypatch):
    messages = []
    monkeypatch.setattr(hf_checkpoint_loader.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(hf_checkpoint_loader, "print_rank_0", messages.append)
    model = torch.nn.Module()
    model.output_layer = torch.nn.Linear(128, 128, bias=False)
    with torch.no_grad():
        model.output_layer.weight.fill_(5)
        model.output_layer.weight[:64, :64].fill_(2)

    hf_checkpoint_loader._log_loaded_param_stats([model])

    assert messages == [
        "[CKPT_LOAD] total params=16,384",
        "[CKPT_LOAD]   output_layer.weight: numel=16,384 sampled=4,096 "
        "sample_norm=128.0000 sample_mean_abs=2.000000",
    ]

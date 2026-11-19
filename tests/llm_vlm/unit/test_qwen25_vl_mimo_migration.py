"""Unit coverage for the Qwen2.5-VL Omni to MiMo migration seam."""

import torch

from loongforge.models.mimo_models.mimo_vlm_model import (
    remap_legacy_omni_state_dict,
)
from loongforge.models.foundation.qwen2.qwen_model import (
    use_native_mrope_for_qwen2_vl,
)
from loongforge.models.mimo_models.qwen25_vl import (
    normalize_encoder_inputs,
    split_image_inputs_for_fan_in,
)


class _Config:
    rotary_emb_func = "RotaryEmbedding"
    position_embedding_type = "rope"
    multi_latent_attention = False
    mrope_section = None


def test_remap_legacy_omni_state_dict():
    legacy = {
        "foundation_model.embedding.word_embeddings.weight": "lm-embedding",
        "encoder_model.image_encoder.patch_embed.proj.weight": "patch",
        "encoder_model.image_projector.linear_fc1.weight": "projector",
        "encoder_model.text_encoder.word_embeddings.weight": "legacy-alias",
    }

    assert remap_legacy_omni_state_dict(legacy) == {
        "language_model.embedding.word_embeddings.weight": "lm-embedding",
        "modality_submodules.images.encoders.image_encoder."
        "patch_embed.proj.weight": "patch",
        "modality_submodules.images.input_projections.0."
        "linear_fc1.weight": "projector",
    }


def test_native_mimo_state_dict_is_unchanged():
    native = {
        "language_model.decoder.layers.0.self_attention.linear_qkv.weight": "qkv",
        "modality_submodules.images.input_projections.0.linear_fc1.weight": "projector",
    }

    assert remap_legacy_omni_state_dict(native) is native


def test_qwen2_vl_legacy_rope_marker_maps_to_native_mrope():
    config = _Config()
    config.rotary_emb_func = "Qwen2VLRotaryEmbedding"

    assert use_native_mrope_for_qwen2_vl(config)
    assert config.position_embedding_type == "mrope"
    assert config.mrope_section == [16, 24, 24]


def test_non_qwen2_vl_rope_is_unchanged():
    config = _Config()

    assert not use_native_mrope_for_qwen2_vl(config)
    assert config.position_embedding_type == "rope"
    assert config.mrope_section is None


def test_qwen25_vision_inputs_accept_mimo_named_encoder_payload():
    payload = {"images": "pixels", "image_grid_thw": "grid"}

    assert normalize_encoder_inputs({"image_encoder": payload}) is payload
    assert normalize_encoder_inputs(payload) is payload


def test_vision_fan_in_assigns_whole_images_with_uneven_lengths():
    patches = torch.arange(20).unsqueeze(1)
    grid = torch.tensor([[1, 2, 4], [1, 2, 6]])
    image_inputs = {"images": patches, "image_grid_thw": grid}

    first, first_sizes = split_image_inputs_for_fan_in(image_inputs, 0, 3, 2)
    second, second_sizes = split_image_inputs_for_fan_in(image_inputs, 1, 3, 2)
    empty, empty_sizes = split_image_inputs_for_fan_in(image_inputs, 2, 3, 2)

    assert first_sizes == second_sizes == empty_sizes == (2, 3, 0)
    torch.testing.assert_close(first["images"], patches[:8])
    torch.testing.assert_close(first["image_grid_thw"], grid[:1])
    torch.testing.assert_close(second["images"], patches[8:])
    torch.testing.assert_close(second["image_grid_thw"], grid[1:])
    assert empty is None

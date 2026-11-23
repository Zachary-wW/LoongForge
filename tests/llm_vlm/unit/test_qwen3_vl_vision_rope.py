from types import SimpleNamespace

import torch

from loongforge.models.encoder.qwen3_vl_vision_models import rope_utils


def test_vision_rope_omits_unsupported_mscale_kwarg(monkeypatch):
    calls = []

    def fake_te_rope(t, _freqs, **kwargs):
        calls.append(kwargs)
        return t

    monkeypatch.setattr(rope_utils, "te_apply_rotary_pos_emb", fake_te_rope)
    monkeypatch.setattr(rope_utils, "_TE_ROPE_HAS_MSCALE", False)
    config = SimpleNamespace(
        apply_rope_fusion=True,
        rotary_interleaved=False,
        multi_latent_attention=False,
    )

    output = rope_utils.apply_rotary_pos_emb(
        torch.zeros(2, 3, 4),
        torch.zeros(2, 1, 4),
        config,
    )

    assert output.shape == (2, 3, 4)
    assert len(calls) == 1
    assert "mscale" not in calls[0]
    assert calls[0]["fused"] is True


def test_vision_rope_accepts_mcore_max_seqlen_kwarg():
    config = SimpleNamespace(
        apply_rope_fusion=False,
        rotary_interleaved=False,
        multi_latent_attention=False,
    )

    output = rope_utils.apply_rotary_pos_emb(
        torch.zeros(2, 3, 4),
        torch.zeros(2, 1, 1, 4),
        config,
        max_seqlen=4096,
    )

    assert output.shape == (2, 3, 4)


def test_vision_rope_preserves_non_unit_mscale_without_te_support(monkeypatch):
    def fail_if_fused_rope_is_called(*_args, **_kwargs):
        raise AssertionError("fused TE RoPE must not run without mscale support")

    monkeypatch.setattr(rope_utils, "te_apply_rotary_pos_emb", fail_if_fused_rope_is_called)
    monkeypatch.setattr(rope_utils, "_TE_ROPE_HAS_MSCALE", False)
    config = SimpleNamespace(
        apply_rope_fusion=True,
        rotary_interleaved=False,
        multi_latent_attention=False,
    )

    output = rope_utils.apply_rotary_pos_emb(
        torch.ones(2, 3, 4),
        torch.zeros(2, 1, 1, 4),
        config,
        mscale=1.5,
        max_seqlen=4096,
    )

    assert output.shape == (2, 3, 4)
    assert torch.all(output == 1.5)


def test_qwen3_vl_mrope_accepts_mcore_max_seqlen_kwarg():
    from loongforge.models.foundation.qwen3 import qwen_layer_spec

    config = SimpleNamespace()
    output = qwen_layer_spec.apply_mrope(
        torch.zeros(2, 1, 3, 4),
        torch.zeros(2, 1, 1, 4),
        config,
        max_seqlen=4096,
    )

    assert output.shape == (2, 1, 3, 4)

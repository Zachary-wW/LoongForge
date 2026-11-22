import inspect
from types import SimpleNamespace

import loongforge.models.encoder.internvl_vision_models.internvl_config as internvl_config
from loongforge.models.encoder.internvl_vision_models.intern_vision_attention import (
    InternSelfAttention,
)
from loongforge.models.encoder.internvl_vision_models.internvl_config import (
    InternVisionConfig,
    resolve_image_token_id,
)


def _config():
    return InternVisionConfig(
        num_layers=1,
        hidden_size=8,
        ffn_hidden_size=16,
        num_attention_heads=2,
        kv_channels=4,
        normalization="LayerNorm",
    )


def test_intern_vision_config_defers_image_token_resolution(monkeypatch):
    def fail_if_called():
        raise AssertionError("tokenizer must not be queried during config construction")

    monkeypatch.setattr(internvl_config, "is_tokenizer_initialized", lambda: False)
    monkeypatch.setattr(internvl_config, "get_tokenizer", fail_if_called)

    config = _config()

    assert config.image_token_id is None


def test_resolve_image_token_id_after_tokenizer_initialization(monkeypatch):
    tokenizer = SimpleNamespace(
        tokenizer=SimpleNamespace(convert_tokens_to_ids=lambda token: 151671)
    )
    monkeypatch.setattr(internvl_config, "get_tokenizer", lambda: tokenizer)

    config = _config()
    assert config.image_token_id is None

    resolve_image_token_id(config)

    assert config.image_token_id == 151671


def test_resolve_image_token_id_preserves_explicit_id(monkeypatch):
    def fail_if_called():
        raise AssertionError("explicit image_token_id must not be overwritten")

    monkeypatch.setattr(internvl_config, "get_tokenizer", fail_if_called)
    config = _config()
    config.image_token_id = 7

    resolve_image_token_id(config)

    assert config.image_token_id == 7


def test_resolve_image_token_id_does_not_touch_non_internvl_config(monkeypatch):
    def fail_if_called():
        raise AssertionError("non-InternVL image encoder must not use InternVL token")

    monkeypatch.setattr(internvl_config, "get_tokenizer", fail_if_called)
    config = SimpleNamespace(image_token_id=None, model_type="qwen2_5_vl")

    resolve_image_token_id(config)

    assert config.image_token_id is None


def test_intern_self_attention_forwards_mcore_context_args(monkeypatch):
    import loongforge.models.encoder.internvl_vision_models.intern_vision_attention as attention_module
    from megatron.core.transformer.attention import Attention

    call = {}

    def fake_attention_init(self, **kwargs):
        call.update(kwargs)
        self.query_projection_size = 8
        self.kv_projection_size = 8
        self.hidden_size_per_attention_head = 4
        self.num_attention_heads_per_partition = 2

    monkeypatch.setattr(Attention, "__init__", fake_attention_init)
    monkeypatch.setattr(
        attention_module,
        "build_module",
        lambda *args, **kwargs: SimpleNamespace(),
    )

    config = SimpleNamespace(
        hidden_size=8,
        num_attention_heads=2,
        kv_channels=4,
        num_query_groups=2,
        add_bias_linear=False,
        add_qkv_bias=False,
        sequence_parallel=False,
        init_method=None,
    )
    submodules = SimpleNamespace(
        linear_qkv=object(),
        q_layernorm=None,
        k_layernorm=None,
    )

    InternSelfAttention(
        config=config,
        submodules=submodules,
        layer_number=3,
        pp_layer_offset=4,
        is_mtp_layer=True,
        name="decoder.layers.3.self_attention",
    )

    assert call["pp_layer_offset"] == 4
    assert call["is_mtp_layer"] is True
    assert call["name"] == "decoder.layers.3.self_attention"


def test_intern_self_attention_signature_contains_mcore_020_context_args():
    parameters = inspect.signature(InternSelfAttention.__init__).parameters

    assert "pp_layer_offset" in parameters
    assert "is_mtp_layer" in parameters
    assert "name" in parameters

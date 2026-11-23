from types import SimpleNamespace

from mcore_checkpoint_convert.huggingface.util.hf_attn_converter import (
    HfAttnGateQkvConverter,
    HfAttnQkvConverter,
)


class _Config:
    def __init__(self):
        self._args = {
            "mcore": {},
            "common": {
                "hidden_size": 4096,
                "num_attention_heads": 32,
                "num_key_value_heads": None,
                "head_dim": 128,
            },
            "huggingface": {},
        }

    def get_args(self, scope):
        return self._args[scope]

    def get_name_map(self, name):
        return {}


def test_null_key_value_heads_defaults_to_mha():
    config = _Config()
    qkv = HfAttnQkvConverter(config)
    gate_qkv = HfAttnGateQkvConverter(config)

    assert qkv.num_key_value_heads == 32
    assert gate_qkv.num_key_value_heads == 32
    assert gate_qkv.num_querys_per_group == 1

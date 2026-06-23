# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.

"""DeepSeek-V4 layer spec.

V4 uses shared-KV MQA attention (not MLA's latent compression), grouped output
projection, attention sinks, inverse RoPE rotation, Manifold-Constrained
Hyper-Connections, and Hash-MoE bootstrap routing for the first N MoE layers.
All 43 layers are MoE (first 3 use hash_moe, remaining use topk moe).
"""

from typing import Tuple, Optional

from omegaconf import ListConfig

from megatron.core.models.gpt.gpt_layer_specs import get_gpt_mtp_block_spec
from megatron.core.transformer.enums import AttnMaskType, LayerType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    get_transformer_layer_offset,
    TransformerLayerSubmodules,
)
from megatron.core.transformer.moe.experts import SequentialMLP, TEGroupedMLP
from megatron.core.transformer.moe.shared_experts import SharedExpertMLP
from megatron.core.transformer.moe.moe_layer import MoELayer, MoESubmodules
from megatron.core.enums import Fp8Recipe

from loongforge.models.dispatch import multiacc_modules
from loongforge.models.foundation.deepseek_v4.deepseek_v4_attention import (
    DeepseekV4Attention,
    DeepseekV4AttentionSubmodules,
)
from loongforge.utils import get_args


def _get_deepseek_v4_layer_with_te_spec(
    num_experts: Optional[int] = None,
    moe_grouped_gemm: Optional[bool] = True,
) -> ModuleSpec:
    """Get the transformer layer spec for DeepSeek-V4.

    V4 uses shared-KV MQA attention (DeepseekV4Attention) instead of MLA.
    When enable_hyper_connections=True, the hyper connection slots are
    activated in TransformerLayerSubmodules.
    """
    mlp = _get_mlp_module_spec(
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
    )

    # V4 shared-KV MQA attention
    attention = ModuleSpec(
        module=DeepseekV4Attention,
        params={"attn_mask_type": AttnMaskType.causal},
        submodules=DeepseekV4AttentionSubmodules(
            linear_q_a_proj=multiacc_modules.TEColumnParallelLinear,
            linear_q_b_proj=multiacc_modules.TEColumnParallelLinear,
            linear_kv_proj=multiacc_modules.TEColumnParallelLinear,
            q_a_norm=IdentityOp,
            q_b_norm=IdentityOp,
            kv_norm=IdentityOp,
            linear_o_a_proj=IdentityOp,
            linear_o_b_proj=multiacc_modules.TERowParallelLinear,
            core_attention=IdentityOp,
        ),
    )

    return ModuleSpec(
        module=TransformerLayer,
        submodules=TransformerLayerSubmodules(
            input_layernorm=multiacc_modules.TENorm,
            self_attention=attention,
            self_attn_bda=multiacc_modules.get_bias_dropout_add,
            # V4 uses mHC: hyper connection slots are populated by TransformerLayer
            # when enable_hyper_connections=True in config
            self_attention_hyper_connection=IdentityOp,
            pre_mlp_layernorm=multiacc_modules.TENorm,
            mlp_hyper_connection=IdentityOp,
            mlp=mlp,
            mlp_bda=multiacc_modules.get_bias_dropout_add,
        ),
    )


def _get_mlp_module_spec(
    num_experts: int = None,
    moe_grouped_gemm: bool = False,
) -> ModuleSpec:
    """Helper function to get module spec for MLP/MoE.

    Keep this function even if the target model is dense-only.
    MoE variants of the same family reuse this file.
    """
    if num_experts is None:
        # Dense MLP w/ TE modules.
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=multiacc_modules.TEColumnParallelLinear,
                linear_fc2=multiacc_modules.TERowParallelLinear,
            ),
        )

    # MoE MLP
    if moe_grouped_gemm:
        assert multiacc_modules.TEColumnParallelGroupedLinear is not None
        expert_module = TEGroupedMLP
        linear_fc1 = multiacc_modules.TEColumnParallelGroupedLinear
        linear_fc2 = multiacc_modules.TERowParallelGroupedLinear
    else:
        expert_module = SequentialMLP
        linear_fc1 = multiacc_modules.TEColumnParallelLinear
        linear_fc2 = multiacc_modules.TERowParallelLinear

    # Shared expert
    shared_linear_fc1 = multiacc_modules.TEColumnParallelLinear
    shared_linear_fc2 = multiacc_modules.TERowParallelLinear

    return ModuleSpec(
        module=MoELayer,
        submodules=MoESubmodules(
            shared_experts=ModuleSpec(
                module=SharedExpertMLP,
                params={"gate": False},
                submodules=MLPSubmodules(
                    linear_fc1=shared_linear_fc1,
                    linear_fc2=shared_linear_fc2,
                ),
            ),
            experts=ModuleSpec(
                module=expert_module,
                submodules=MLPSubmodules(
                    linear_fc1=linear_fc1,
                    linear_fc2=linear_fc2,
                ),
            ),
        ),
    )


def get_deepseek_v4_decoder_block_and_mtp_spec(
    config: TransformerConfig,
    vp_stage: int = None,
) -> Tuple[TransformerBlockSubmodules, Optional[ModuleSpec]]:
    """Get the DeepSeek-V4 decoder block and multi-token prediction layer spec.

    All 43 layers are MoE in V4. The first num_hash_layers (default 3) use
    hash_moe routing, the rest use topk moe.
    """
    assert config.num_moe_experts > 0, "DeepSeek-V4 requires MoE (all layers are MoE)"
    assert config.multi_latent_attention, "DeepSeek-V4 requires multi_latent_attention=True"

    block_spec = None
    mtp_block_spec = None
    use_te = config.transformer_impl == "transformer_engine"

    # All layers are MoE in V4 — no separate dense layer spec
    moe_layer_spec = _get_deepseek_v4_layer_with_te_spec(
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
    )

    # Parse moe_layer_freq to determine dense/MoE pattern.
    # V4: all layers are MoE, so moe_layer_freq is all-ones.
    # But we support three formats per R020.
    if config.moe_layer_freq is not None:
        # compatibility for hydra config
        if isinstance(config.moe_layer_freq, ListConfig):
            config.moe_layer_freq = list(config.moe_layer_freq)

        if isinstance(config.moe_layer_freq, int):
            moe_layer_pattern = [
                1 if (i % config.moe_layer_freq == 0) else 0
                for i in range(config.num_layers)
            ]
        elif isinstance(config.moe_layer_freq, list):
            moe_layer_pattern = config.moe_layer_freq
            assert len(moe_layer_pattern) == config.num_layers, (
                f"Invalid length of moe_layer_pattern: {len(moe_layer_pattern)}, "
                f"expected {config.num_layers}, "
                f"current moe layer pattern: {config.moe_layer_freq}"
            )
        elif isinstance(config.moe_layer_freq, str):
            import ast
            freq_list = ast.literal_eval(config.moe_layer_freq)
            moe_layer_pattern = [bool(freq_list[i % len(freq_list)]) for i in range(config.num_layers)]
        else:
            raise ValueError(
                f"Invalid moe_layer_freq: {type(config.moe_layer_freq)}, {config.moe_layer_freq}"
            )
    else:
        # Default: all layers are MoE
        moe_layer_pattern = [1] * config.num_layers

    # Create the layer specs for the model.
    layer_specs = []
    for layer_number in range(config.num_layers):
        if moe_layer_pattern[layer_number] == 1:
            layer_specs.append(moe_layer_spec)
        elif moe_layer_pattern[layer_number] == 0:
            # Dense fallback (should not happen in V4, but supported for compatibility)
            dense_layer_spec = _get_deepseek_v4_layer_with_te_spec(
                num_experts=None,
                moe_grouped_gemm=False,
            )
            layer_specs.append(dense_layer_spec)
        else:
            raise ValueError(f"Invalid layer pattern: {moe_layer_pattern}")

    # Slice the layer specs to only include the layers built in this pipeline stage.
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)

    if config.pipeline_model_parallel_layout is not None:
        local_layer_specs = [
            layer_specs[layer_id]
            for layer_id in config.pipeline_model_parallel_layout.get_layer_id_list(
                layer_type=LayerType.decoder, vp_stage=vp_stage
            )
        ]
    else:
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        local_layer_specs = layer_specs[offset : offset + num_layers_to_build]

    # Block spec.
    block_spec = TransformerBlockSubmodules(
        layer_specs=local_layer_specs,
        layer_norm=multiacc_modules.TENorm,
    )

    # MTP spec
    if config.mtp_num_layers is not None:
        if hasattr(block_spec, "layer_specs") and len(block_spec.layer_specs) == 0:
            transformer_layer_spec_for_mtp = _get_deepseek_v4_layer_with_te_spec(
                num_experts=config.num_moe_experts,
                moe_grouped_gemm=config.moe_grouped_gemm,
            )
        else:
            transformer_layer_spec_for_mtp = block_spec

        mtp_block_spec = get_gpt_mtp_block_spec(
            config, transformer_layer_spec_for_mtp,
            use_transformer_engine=use_te, vp_stage=vp_stage
        )

    return block_spec, mtp_block_spec

# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 layer spec using native Megatron Core 0.20 attention."""

from typing import Optional, Tuple

from omegaconf import ListConfig

from megatron.core.extensions.transformer_engine_spec_provider import TESpecProvider
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.backends import BackendSpecProvider
from megatron.core.models.gpt.moe_module_specs import get_moe_module_spec_for_backend
from megatron.core.models.backends import get_backend_from_config
from megatron.core.transformer.enums import LayerType
from megatron.core.transformer.experimental_attention_variant.deepseek_v4_hybrid_attention_module_specs import (
    get_dsv4_hybrid_module_spec_for_backend,
)
from megatron.core.transformer.hyper_connection import HyperConnectionModule
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.multi_token_prediction import (
    MultiTokenPredictionLayer,
    MultiTokenPredictionLayerSubmodules,
    MultiTokenPredictionBlockSubmodules,
    get_mtp_num_layers_to_build,
)
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_block import (
    TransformerBlockSubmodules,
    get_num_layers_to_build,
)
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.transformer.transformer_layer import (
    HyperConnectionTransformerLayer,
    TransformerLayer,
    TransformerLayerSubmodules,
    get_transformer_layer_offset,
)
from megatron.core.models.hybrid.hybrid_layer_specs import hybrid_stack_spec


def _get_mlp_module_spec_for_backend(
    backend: BackendSpecProvider,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: bool = False,
    use_te_activation_func: bool = False,
) -> ModuleSpec:
    """Build the dense or MoE MLP spec used by DeepSeek-V4."""
    if num_experts is None:
        return ModuleSpec(
            module=MLP,
            submodules=MLPSubmodules(
                linear_fc1=backend.column_parallel_linear(),
                linear_fc2=backend.row_parallel_linear(),
                activation_func=backend.activation_func() if use_te_activation_func else None,
            ),
        )

    return get_moe_module_spec_for_backend(
        backend=backend,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        use_te_activation_func=use_te_activation_func,
    )


def _get_deepseek_v4_layer_with_te_spec(
    config: TransformerConfig,
    backend: BackendSpecProvider,
    num_experts: Optional[int] = None,
    moe_grouped_gemm: bool = False,
    use_te_activation_func: bool = False,
) -> ModuleSpec:
    """Build one DSv4 TransformerLayer with native MCore attention modules."""
    mlp = _get_mlp_module_spec_for_backend(
        backend=backend,
        num_experts=num_experts,
        moe_grouped_gemm=moe_grouped_gemm,
        use_te_activation_func=use_te_activation_func,
    )
    hyper_connection = HyperConnectionModule if config.enable_mhc_connections else IdentityOp
    layer_module = HyperConnectionTransformerLayer if config.enable_mhc_connections else TransformerLayer
    attention = get_dsv4_hybrid_module_spec_for_backend(config=config, backend=backend)

    return ModuleSpec(
        module=layer_module,
        submodules=TransformerLayerSubmodules(
            input_layernorm=backend.layer_norm(),
            self_attention=attention,
            self_attn_bda=get_bias_dropout_add,
            self_attention_hyper_connection=hyper_connection,
            pre_mlp_layernorm=backend.layer_norm() if num_experts else IdentityOp,
            mlp=mlp,
            mlp_bda=get_bias_dropout_add,
            mlp_hyper_connection=hyper_connection,
        ),
    )


def _get_moe_layer_pattern(config: TransformerConfig):
    """Return the dense/MoE pattern for decoder layers."""
    if isinstance(config.moe_layer_freq, ListConfig):
        config.moe_layer_freq = list(config.moe_layer_freq)
    if isinstance(config.moe_layer_freq, int):
        return [1 if i % config.moe_layer_freq == 0 else 0 for i in range(config.num_layers)]
    if isinstance(config.moe_layer_freq, list):
        if len(config.moe_layer_freq) != config.num_layers:
            raise ValueError(
                f"Invalid moe_layer_freq length: {len(config.moe_layer_freq)}, "
                f"expected {config.num_layers}"
            )
        return config.moe_layer_freq
    raise ValueError(f"Invalid moe_layer_freq: {config.moe_layer_freq!r}")


def get_deepseek_v4_decoder_block_and_mtp_spec(
    config: TransformerConfig,
    vp_stage: Optional[int] = None,
) -> Tuple[TransformerBlockSubmodules, Optional[ModuleSpec]]:
    """Return decoder and MTP block specs backed by native MCore 0.20 modules."""
    if not config.num_moe_experts:
        raise ValueError("DeepSeek-V4 requires MoE")
    if not config.multi_latent_attention:
        raise ValueError("DeepSeek-V4 requires MLA")
    if config.experimental_attention_variant != "dsv4_hybrid":
        raise ValueError("DeepSeek-V4 requires experimental_attention_variant='dsv4_hybrid'")
    if config.transformer_impl != "transformer_engine":
        raise ValueError("DeepSeek-V4 currently requires Transformer Engine")

    backend = TESpecProvider()
    dense_layer_spec = _get_deepseek_v4_layer_with_te_spec(
        config=config,
        backend=backend,
        num_experts=None,
        use_te_activation_func=config.use_te_activation_func,
    )
    moe_layer_spec = _get_deepseek_v4_layer_with_te_spec(
        config=config,
        backend=backend,
        num_experts=config.num_moe_experts,
        moe_grouped_gemm=config.moe_grouped_gemm,
        use_te_activation_func=config.use_te_activation_func,
    )

    pattern = _get_moe_layer_pattern(config)
    layer_specs = [moe_layer_spec if is_moe else dense_layer_spec for is_moe in pattern]
    num_layers_to_build = get_num_layers_to_build(config, vp_stage=vp_stage)

    if config.pipeline_model_parallel_layout is not None:
        local_layer_specs = [
            layer_specs[layer_id]
            for layer_id in config.pipeline_model_parallel_layout.get_layer_id_list(
                layer_type=LayerType.decoder,
                vp_stage=vp_stage,
            )
        ]
    else:
        offset = get_transformer_layer_offset(config, vp_stage=vp_stage)
        local_layer_specs = layer_specs[offset : offset + num_layers_to_build]

    block_spec = TransformerBlockSubmodules(
        layer_specs=local_layer_specs,
        layer_norm=backend.layer_norm(),
    )

    mtp_block_spec = None
    num_mtp_layers = get_mtp_num_layers_to_build(config, vp_stage=vp_stage)
    if num_mtp_layers > 0:
        ratio_symbol = {0: "W", 4: "C", 128: "H"}[config.csa_compress_ratios[-1]]
        mtp_layer_pattern = ratio_symbol + "E"
        config.mtp_layer_pattern = mtp_layer_pattern
        config.mtp_hybrid_submodules = hybrid_stack_spec.submodules
        backend = get_backend_from_config(config, transformer_impl="transformer_engine")
        linear = backend.column_parallel_linear()
        norm = backend.layer_norm(rms_norm=config.normalization == "RMSNorm")
        mtp_layer_spec = ModuleSpec(
            module=MultiTokenPredictionLayer,
            submodules=MultiTokenPredictionLayerSubmodules(
                enorm=norm,
                hnorm=norm,
                eh_proj=linear,
                e_proj=linear,
                h_proj=linear,
                mtp_model_layer=None,
                layer_norm=norm,
            ),
        )
        mtp_block_spec = MultiTokenPredictionBlockSubmodules(
            layer_specs=[mtp_layer_spec] * num_mtp_layers
        )

    return block_spec, mtp_block_spec

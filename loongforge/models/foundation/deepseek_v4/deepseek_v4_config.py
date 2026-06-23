# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 model configuration."""

from typing import Optional, Union, List
from dataclasses import dataclass

from loongforge.models.common.base_model_config import BaseModelMLAConfig
from loongforge.utils.constants import LanguageModelFamilies


@dataclass
class DeepseekV4Config(BaseModelMLAConfig):
    """Configuration for DeepSeek-V4 model (shared-KV MQA + MoE + mHC).

    V4 extends the DeepSeek family with:
    - Shared-KV MQA (num_key_value_heads=1, K=V same tensor)
    - Grouped output projection (o_groups, o_lora_rank)
    - Per-head attention sinks
    - Manifold-Constrained Hyper-Connections (mHC)
    - Compressed Sparse Attention (CSA) / Heavily Compressed Attention (HCA)
    - Lightning Indexer for CSA layers
    - Hash-MoE bootstrap routing for first N layers
    - SwiGLU clamp limit on expert activations
    - Interleaved RoPE on trailing rope slice
    - Two rope type labels: main (plain theta) and compress (yarn theta)
    """

    # ── Required fields (NO default values, filled by YAML) ──────────────────
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int

    # ── V4-specific: Shared-KV MQA ──────────────────────────────────────────
    head_dim: int = None                     # explicitly 512 in V4
    q_lora_rank: int = None                 # Q compression rank (1024)
    qk_rope_head_dim: int = None            # rope dims per head (64)
    o_groups: int = None                    # grouped output groups (8)
    o_lora_rank: int = None                 # per-group intermediate dim (1024)

    # ── MoE (V4: all layers are MoE, first N use hash_moe) ──────────────────
    num_experts: int = None                 # total number of experts (256)
    moe_ffn_hidden_size: int = None         # per-expert FFN hidden size (2048)
    moe_shared_expert_intermediate_size: int = None  # shared expert size
    moe_layer_freq: Optional[Union[int, List[int]]] = None  # which layers are MoE
    moe_router_score_function: str = "sqrtsoftplus"  # V4 uses sqrtsoftplus scoring
    moe_router_load_balancing_type: str = "none"  # V4 uses noaux_tc
    moe_router_topk: int = 6
    moe_router_enable_expert_bias: bool = True   # e_score_correction_bias
    moe_grouped_gemm: bool = True

    # ── Hash-MoE bootstrap ───────────────────────────────────────────────────
    num_hash_layers: int = None             # first N layers use hash routing (3)

    # ── SwiGLU clamp ────────────────────────────────────────────────────────
    swiglu_limit: float = None             # clamp limit (10.0)

    # ── Manifold-Constrained Hyper-Connections (mHC) ─────────────────────────
    enable_hyper_connections: bool = True
    num_residual_streams: int = 4           # hc_mult
    mhc_sinkhorn_iterations: int = 20
    mhc_init_gating_factor: float = 0.01

    # ── Compressed Attention ─────────────────────────────────────────────────
    layer_types: Optional[List[str]] = None         # per-layer attn schedule
    mlp_layer_types: Optional[List[str]] = None      # per-layer MoE schedule
    compress_rates: Optional[dict] = None            # {"csa": 4, "hca": 128}
    compress_rope_theta: float = None                # 160000
    sliding_window: int = None                       # 128

    # ── Lightning Indexer ────────────────────────────────────────────────────
    index_n_heads: int = None               # 64
    index_head_dim: int = None              # 128
    index_topk: int = None                  # 512

    # ── GQA (V4 uses single KV head) ────────────────────────────────────────
    group_query_attention: bool = True
    num_query_groups: int = 1

    # ── MLA flag ─────────────────────────────────────────────────────────────
    multi_latent_attention: bool = True     # V4 reuses MLA config path

    # ── RoPE ─────────────────────────────────────────────────────────────────
    position_embedding_type: str = "rope"
    add_position_embedding: bool = False
    rotary_interleaved: bool = True         # V4 uses interleaved RoPE
    rotary_base: int = 10000                # main rope theta
    apply_rope_fusion: bool = True
    rotary_percent: float = None            # partial_rotary_factor (0.125)

    # ── Normalization ────────────────────────────────────────────────────────
    normalization: str = "RMSNorm"

    # ── FFN ──────────────────────────────────────────────────────────────────
    swiglu: bool = True

    # ── Dropout ──────────────────────────────────────────────────────────────
    attention_dropout: float = 0
    hidden_dropout: float = 0

    # ── Linear bias ──────────────────────────────────────────────────────────
    add_bias_linear: bool = False
    add_qkv_bias: bool = False
    qk_layernorm: bool = False

    # ── Vocab / Embedding ────────────────────────────────────────────────────
    untie_embeddings_and_output_weights: bool = True
    vocab_size_in_config_file: int = None
    make_vocab_size_divisible_by: int = 128

    # ── KV channels ──────────────────────────────────────────────────────────
    kv_channels: int = None

    # ── MTP ──────────────────────────────────────────────────────────────────
    mtp_num_layers: int = 0
    mtp_loss_coef: float = 0.1

    # ── Routing ──────────────────────────────────────────────────────────────
    routed_scaling_factor: float = 1.5

    model_type = LanguageModelFamilies.DEEPSEEK_V4
    model_spec = [
        "loongforge.models.foundation.deepseek_v4.deepseek_v4_layer_spec",
        "get_deepseek_v4_decoder_block_and_mtp_spec",
    ]

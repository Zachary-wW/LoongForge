# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 model configuration (shared-KV MQA + MoE + mHC).

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

from typing import Optional, Union, List
from dataclasses import dataclass

from loongforge.models.common.base_model_config import BaseModelMLAConfig
from loongforge.utils.constants import LanguageModelFamilies


@dataclass
class DeepseekV4Config(BaseModelMLAConfig):
    """Configuration for DeepSeek-V4 model.

    HF-facing names are used as dataclass field names; Megatron internal
    names are mapped in __post_init__.

    G2 field mapping (HF-facing -> Megatron internal):
        hc_mult             -> num_residual_streams
        hc_sinkhorn_iters   -> mhc_sinkhorn_iterations
        swiglu_limit        -> activation_func_clamp_value
    """

    # ── Required fields (NO default values, filled by YAML) ──────────────────
    num_layers: int
    hidden_size: int
    ffn_hidden_size: int
    num_attention_heads: int

    # ── V4-specific: Shared-KV MQA ──────────────────────────────────────────
    head_dim: int = None                     # explicitly 512 in V4
    q_lora_rank: int = None                  # Q compression rank (1024)
    qk_rope_head_dim: int = None             # rope dims per head (64)
    o_groups: int = None                     # grouped output groups (8)
    o_lora_rank: int = None                  # per-group intermediate dim (1024)

    # ── MoE (V4: all layers are MoE, first N use hash_moe) ──────────────────
    num_experts: int = None                  # total number of experts (256)
    moe_ffn_hidden_size: int = None          # per-expert FFN hidden size (2048)
    moe_shared_expert_intermediate_size: int = None  # shared expert size
    moe_layer_freq: Optional[Union[int, List[int]]] = None
    moe_router_score_function: str = "sqrtsoftplus"   # V4 uses sqrtsoftplus scoring
    moe_router_load_balancing_type: str = "none"      # V4 uses noaux_tc
    moe_router_topk: int = 6
    moe_router_enable_expert_bias: bool = True        # e_score_correction_bias
    moe_grouped_gemm: bool = True

    # ── Hash-MoE bootstrap ───────────────────────────────────────────────────
    num_hash_layers: int = None              # first N layers use hash routing (3)

    # ── SwiGLU clamp ────────────────────────────────────────────────────────
    swiglu_limit: float = None               # clamp limit (10.0)

    # ── Manifold-Constrained Hyper-Connections (mHC) ─────────────────────────
    enable_hyper_connections: bool = True
    hc_mult: int = 4
    hc_sinkhorn_iters: int = 20
    hc_eps: float = 1e-6

    # ── Compressed Attention ─────────────────────────────────────────────────
    layer_types: Optional[List[str]] = None          # per-layer attn schedule
    mlp_layer_types: Optional[List[str]] = None       # per-layer MoE schedule
    compress_rates: Optional[dict] = None             # {"csa": 4, "hca": 128}
    compress_rope_theta: float = None                 # 160000
    sliding_window: int = None                        # 128

    # ── Lightning Indexer ────────────────────────────────────────────────────
    index_n_heads: int = None                # 64
    index_head_dim: int = None               # 128
    index_topk: int = None                   # 512

    # ── GQA (V4 uses single KV head) ────────────────────────────────────────
    group_query_attention: bool = True
    num_query_groups: int = 1

    # ── MLA flag ─────────────────────────────────────────────────────────────
    multi_latent_attention: bool = True      # V4 reuses MLA config path

    # ── RoPE ─────────────────────────────────────────────────────────────────
    position_embedding_type: str = "rope"
    add_position_embedding: bool = False
    rotary_interleaved: bool = True          # V4 uses interleaved RoPE
    rotary_base: int = 10000                 # main rope theta
    apply_rope_fusion: bool = True
    rotary_percent: float = None             # partial_rotary_factor (0.125)

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

    # ── Vocab / Embedding ───────────────────────────────────────────────────
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

    def __post_init__(self):
        # ── (a) Field mapping: HF-facing -> Megatron internal ──────────────
        # G2: Map HF-facing field names to Megatron internal names.
        self.num_residual_streams = self.hc_mult
        self.mhc_sinkhorn_iterations = self.hc_sinkhorn_iters
        self.activation_func_clamp_value = self.swiglu_limit if self.swiglu_limit is not None else None

        # ── (b) Validation assertions ──────────────────────────────────────
        assert self.num_experts is not None and self.num_experts > 0, (
            "DeepSeek-V4 requires num_experts > 0 (all layers are MoE)"
        )
        assert self.multi_latent_attention, (
            "DeepSeek-V4 requires multi_latent_attention=True"
        )
        assert self.enable_hyper_connections, (
            "DeepSeek-V4 requires enable_hyper_connections=True"
        )
        if self.head_dim is not None:
            assert self.head_dim > 256, (
                "V4 head_dim=512 exceeds FA2/3/4 limit; eager-only required"
            )
        if self.moe_layer_freq is not None:
            if isinstance(self.moe_layer_freq, list):
                assert len(self.moe_layer_freq) == self.num_layers, (
                    f"moe_layer_freq length ({len(self.moe_layer_freq)}) "
                    f"must equal num_layers ({self.num_layers})"
                )
        if self.compress_rates is not None:
            for key in ["compressed_sparse_attention", "heavily_compressed_attention"]:
                if key in self.compress_rates:
                    assert self.compress_rates[key] > 0, (
                        f"compress_rate for {key} must be positive"
                    )

        # ── (c) Derived field computation ─────────────────────────────────
        # qk_nope_head_dim derived from head_dim and qk_rope_head_dim
        if self.head_dim is not None and self.qk_rope_head_dim is not None:
            self.qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim

        # rotary_percent derived from partial_rotary_factor
        if self.rotary_percent is None and self.qk_rope_head_dim is not None and self.head_dim is not None:
            self.rotary_percent = self.qk_rope_head_dim / self.head_dim

        # kv_channels derived from head_dim
        if self.kv_channels is None and self.head_dim is not None:
            self.kv_channels = self.head_dim

        super().__post_init__()

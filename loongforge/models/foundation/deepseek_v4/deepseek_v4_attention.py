# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 Shared-KV MQA Attention with grouped output projection,
per-head attention sinks, and inverse RoPE rotation on the output.

Key differences from MLA (DeepSeek-V3):
- Shared-KV MQA: single kv_proj projects to head_dim for the single KV head;
  the same tensor is used as both key and value (K=V).
- head_dim=512 exceeds Flash Attention 2/3/4 limit of 256; eager-only.
- Grouped output projection: o_a_proj (GroupedLinear) + o_b_proj.
- Per-head learnable attention sinks (prevents attention collapse at seq start).
- Inverse RoPE rotation on attention output before grouped output projection.
- q_b_norm uses UnweightedRMSNorm (no learnable weight).
"""

import math
from dataclasses import dataclass
from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core.tensor_parallel import ColumnParallelLinear, RowParallelLinear
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.spec_utils import ModuleSpec
from megatron.core.transformer.transformer_config import TransformerConfig

from loongforge.models.dispatch import multiacc_modules


class DeepseekV4UnweightedRMSNorm(torch.nn.Module):
    """RMSNorm without learnable weight — pure variance normalization."""

    def __init__(self, eps: float = 1e-6):
        super().__init__()
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return x * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps).to(x.dtype)


def _rotate_half_interleaved(x: Tensor) -> Tensor:
    """V4 interleaved RoPE rotate_half: pairs (0::2, 1::2) channels."""
    x1 = x[..., 0::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).flatten(-2)


def _apply_interleaved_rope(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    unsqueeze_dim: int = 1,
) -> Tensor:
    """Apply interleaved RoPE on the trailing rope slice of x.

    cos/sin are half-sized (rope_head_dim // 2); expanded with repeat_interleave(2).
    The leading nope channels are untouched.
    """
    cos = cos.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    sin = sin.repeat_interleave(2, dim=-1).unsqueeze(unsqueeze_dim)
    rope_dim = cos.shape[-1]
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rotated = ((rope.float() * cos) + (_rotate_half_interleaved(rope).float() * sin)).to(x.dtype)
    return torch.cat([nope, rotated], dim=-1)


def _apply_inverse_interleaved_rope(
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    unsqueeze_dim: int = 1,
) -> Tensor:
    """Inverse interleaved RoPE: use (cos, -sin) to undo the rotation on the rope slice."""
    return _apply_interleaved_rope(x, cos, -sin, unsqueeze_dim=unsqueeze_dim)


@dataclass
class DeepseekV4AttentionSubmodules:
    """Submodule slots for DeepSeek-V4 shared-KV MQA attention.

    V4 attention does not follow the standard MLA submodule pattern.
    Instead it uses q_a_proj -> q_b_proj for Q compression, a single
    kv_proj for shared K=V, and grouped output projection.
    """
    linear_q_a_proj: Union[ModuleSpec, type] = IdentityOp
    linear_q_b_proj: Union[ModuleSpec, type] = IdentityOp
    linear_kv_proj: Union[ModuleSpec, type] = IdentityOp
    q_a_norm: Union[ModuleSpec, type] = IdentityOp
    q_b_norm: Union[ModuleSpec, type] = IdentityOp
    kv_norm: Union[ModuleSpec, type] = IdentityOp
    linear_o_a_proj: Union[ModuleSpec, type] = IdentityOp
    linear_o_b_proj: Union[ModuleSpec, type] = IdentityOp
    core_attention: Union[ModuleSpec, type] = IdentityOp


class DeepseekV4Attention(MegatronModule):
    """DeepSeek-V4 Shared-KV Multi-Query Attention.

    Uses single kv_proj for K=V, grouped output projection, attention sinks,
    and inverse RoPE rotation on the attention output.

    This module follows the Megatron spec-driven assembly pattern:
    it accepts submodules via DeepseekV4AttentionSubmodules, and
    TransformerLayer builds it via ModuleSpec.
    """

    def __init__(
        self,
        config: TransformerConfig,
        submodules: DeepseekV4AttentionSubmodules,
        layer_number: int = None,
        attn_mask_type: AttnMaskType = AttnMaskType.causal,
        attention_type: str = None,
    ):
        super().__init__(config=config)
        self.config = config
        self.layer_number = layer_number
        self.attn_mask_type = attn_mask_type

        # V4 attention parameters
        self.num_heads = config.num_attention_heads
        self.head_dim = getattr(config, "head_dim", config.hidden_size // self.num_heads)
        self.qk_rope_head_dim = getattr(config, "qk_rope_head_dim", 64)
        self.qk_nope_head_dim = self.head_dim - self.qk_rope_head_dim
        self.q_lora_rank = getattr(config, "q_lora_rank", None)
        self.o_groups = getattr(config, "o_groups", 8)
        self.o_lora_rank = getattr(config, "o_lora_rank", 1024)
        self.scaling = self.head_dim ** -0.5

        # Q compression: q_a_proj -> q_a_norm -> q_b_proj -> q_b_norm
        self.linear_q_a_proj = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=self.q_lora_rank,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_weight_param_allocation=True,
            skip_bias_add=True,
        )
        self.q_a_norm = DeepseekV4UnweightedRMSNorm(eps=config.layernorm_epsilon)

        self.linear_q_b_proj = ColumnParallelLinear(
            input_size=self.q_lora_rank,
            output_size=self.num_heads * self.head_dim,
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            skip_weight_param_allocation=True,
            skip_bias_add=True,
        )
        self.q_b_norm = DeepseekV4UnweightedRMSNorm(eps=config.layernorm_epsilon)

        # Shared-KV projection (single head, K=V)
        self.linear_kv_proj = ColumnParallelLinear(
            input_size=config.hidden_size,
            output_size=self.head_dim,
            config=config,
            init_method=config.init_method,
            bias=False,
            skip_weight_param_allocation=True,
            skip_bias_add=True,
        )
        self.kv_norm = multiacc_modules.TENorm(
            hidden_size=self.head_dim,
            config=config,
            eps=config.layernorm_epsilon,
        )

        # Grouped output projection
        self.linear_o_a_proj = _GroupedLinear(
            input_size_per_group=self.num_heads * self.head_dim // self.o_groups,
            output_size=self.o_lora_rank,
            n_groups=self.o_groups,
            config=config,
            init_method=config.output_layer_init_method,
        )
        self.linear_o_b_proj = RowParallelLinear(
            input_size=self.o_groups * self.o_lora_rank,
            output_size=config.hidden_size,
            config=config,
            init_method=config.output_layer_init_method,
            bias=False,
            skip_weight_param_allocation=True,
            skip_bias_add=True,
        )

        # Per-head attention sinks
        self.sinks = torch.nn.Parameter(torch.empty(self.num_heads))
        torch.nn.init.zeros_(self.sinks)

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor = None,
        rotary_emb_cos: Tensor = None,
        rotary_emb_sin: Tensor = None,
        encoder_output: Tensor = None,
        attention_type: str = None,
        packed_seq_params=None,
    ) -> Tuple[Tensor, Tensor]:
        """Forward pass for V4 shared-KV MQA attention.

        Args:
            hidden_states: [S, B, H] (Megatron convention)
            rotary_emb_cos/sin: cos/sin for interleaved RoPE (optional)
            attention_mask: attention mask tensor
        """
        # Q compression path
        q_residual = self.q_a_norm(self.linear_q_a_proj(hidden_states)[0])
        q = self.linear_q_b_proj(q_residual)[0]
        # Reshape Q: [S, B, num_heads * head_dim] -> [B, num_heads, S, head_dim]
        seq_len, batch_size = q.shape[0], q.shape[1]
        q = q.view(seq_len, batch_size, self.num_heads, self.head_dim)
        q = q.permute(1, 2, 0, 3).contiguous()  # [B, num_heads, S, head_dim]
        q = self.q_b_norm(q)

        # Apply interleaved RoPE on Q (trailing rope slice)
        if rotary_emb_cos is not None and rotary_emb_sin is not None:
            q = _apply_interleaved_rope(q, rotary_emb_cos, rotary_emb_sin)

        # Shared-KV projection
        kv = self.kv_norm(self.linear_kv_proj(hidden_states)[0])
        # Reshape KV: [S, B, head_dim] -> [B, 1, S, head_dim]
        kv = kv.view(seq_len, batch_size, 1, self.head_dim)
        kv = kv.permute(1, 2, 0, 3).contiguous()  # [B, 1, S, head_dim]

        # Apply interleaved RoPE on KV
        if rotary_emb_cos is not None and rotary_emb_sin is not None:
            kv = _apply_interleaved_rope(kv, rotary_emb_cos, rotary_emb_sin)

        # Broadcast KV to all heads: [B, 1, S, head_dim] -> [B, num_heads, S, head_dim]
        key_states = kv.expand(-1, self.num_heads, -1, -1).contiguous()
        value_states = key_states  # K = V in shared-KV MQA

        # Eager attention with sinks
        attn_weights = torch.matmul(q, key_states.transpose(2, 3)) * self.scaling
        if attention_mask is not None:
            attn_weights = attn_weights + attention_mask

        # Append attention sinks to logits
        sinks = self.sinks.reshape(1, -1, 1, 1).expand(
            attn_weights.shape[0], -1, attn_weights.shape[2], -1
        )
        combined_logits = torch.cat([attn_weights, sinks], dim=-1)
        combined_logits = combined_logits - combined_logits.max(dim=-1, keepdim=True).values

        probs = F.softmax(combined_logits.float(), dim=-1).to(attn_weights.dtype)
        scores = probs[..., :-1]  # drop sink column

        attn_output = torch.matmul(scores, value_states)
        # [B, num_heads, S, head_dim] -> [B, S, num_heads, head_dim]
        attn_output = attn_output.transpose(1, 2).contiguous()

        # Inverse RoPE rotation on the attention output's rope slice
        if rotary_emb_cos is not None and rotary_emb_sin is not None:
            attn_output = _apply_inverse_interleaved_rope(
                attn_output, rotary_emb_cos, rotary_emb_sin
            )

        # Grouped output projection
        # [B, S, num_heads, head_dim] -> [B, S, o_groups, num_heads*head_dim/o_groups]
        grouped = attn_output.reshape(
            *attn_output.shape[:2], self.o_groups, -1
        )
        grouped = self.linear_o_a_proj(grouped).flatten(2)
        output = self.linear_o_b_proj(grouped)[0]

        return output, None


class _GroupedLinear(torch.nn.Module):
    """Block-diagonal grouped linear for V4's output projection.

    Internal helper — not part of the Megatron spec-driven submodule system.
    """

    def __init__(
        self,
        input_size_per_group: int,
        output_size: int,
        n_groups: int,
        config: TransformerConfig,
        init_method=None,
    ):
        super().__init__()
        self.n_groups = n_groups
        self.input_size_per_group = input_size_per_group
        self.output_size = output_size
        self.weight = torch.nn.Parameter(
            torch.empty(n_groups, output_size, input_size_per_group)
        )
        self.register_parameter("bias", None)
        self._init_weight(init_method, config)

    def _init_weight(self, init_method, config):
        if init_method is not None:
            for g in range(self.n_groups):
                init_method(self.weight[g])
        else:
            std = getattr(config, "init_method_std", 0.02)
            torch.nn.init.normal_(self.weight, mean=0.0, std=std)

    def forward(self, x: Tensor) -> Tensor:
        """x: [B, S, o_groups, input_size_per_group] -> [B, S, o_groups, output_size]"""
        input_shape = x.shape[:-2]
        hidden_dim = x.shape[-1]
        w = self.weight.transpose(1, 2)  # [n_groups, input_size_per_group, output_size]
        x = x.reshape(-1, self.n_groups, hidden_dim).transpose(0, 1)
        y = torch.bmm(x, w).transpose(0, 1)
        return y.reshape(*input_shape, self.n_groups, -1)

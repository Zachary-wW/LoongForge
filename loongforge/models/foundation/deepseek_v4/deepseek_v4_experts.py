# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 Experts with SwiGLU clamp limit.

V4 experts use packed gate_up_proj (2 * intermediate_dim) + down_proj,
with swiglu_limit=10.0 clamp on gate and up pre-activations to prevent
numerical overflow in BF16/FP16 with large expert counts.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


class DeepseekV4Experts(MegatronModule):
    """Collection of expert weights stored as 3D tensors with SwiGLU clamp."""

    def __init__(
        self,
        config: TransformerConfig,
        num_local_experts: int,
        moe_ffn_hidden_size: int,
    ):
        super().__init__(config=config)
        self.num_experts = num_local_experts
        self.hidden_dim = config.hidden_size
        self.intermediate_dim = moe_ffn_hidden_size
        self.limit = getattr(config, "swiglu_limit", 10.0)

        # Packed gate_up_proj: [num_experts, 2 * intermediate_dim, hidden_dim]
        self.gate_up_proj = torch.nn.Parameter(
            torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim)
        )
        # down_proj: [num_experts, hidden_dim, intermediate_dim]
        self.down_proj = torch.nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim)
        )

    def _apply_gate(self, gate_up: Tensor) -> Tensor:
        """Apply SwiGLU with clamp limit on gate and up pre-activations."""
        gate, up = gate_up.chunk(2, dim=-1)
        gate = gate.clamp(max=self.limit)
        up = up.clamp(min=-self.limit, max=self.limit)
        return F.silu(gate) * up

    def forward(
        self,
        hidden_states: Tensor,
        top_k_index: Tensor,
        top_k_weights: Tensor,
    ) -> Tensor:
        """Forward pass for routed experts.

        Args:
            hidden_states: [B * S, H]
            top_k_index: [B * S, topk] expert indices
            top_k_weights: [B * S, topk] expert weights

        Returns:
            [B * S, H] weighted expert outputs
        """
        final = torch.zeros_like(hidden_states)
        with torch.no_grad():
            mask = F.one_hot(top_k_index, num_classes=self.num_experts).permute(2, 1, 0)
            hit = torch.greater(mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(mask[expert_idx])
            current = self._apply_gate(
                F.linear(hidden_states[token_idx], self.gate_up_proj[expert_idx])
            )
            current = F.linear(current, self.down_proj[expert_idx]) * top_k_weights[
                token_idx, top_k_pos, None
            ]
            final.index_add_(0, token_idx, current.to(final.dtype))
        return final

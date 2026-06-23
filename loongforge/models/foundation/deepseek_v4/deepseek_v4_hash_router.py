# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""DeepSeek-V4 Hash Router for the first N MoE layers.

Expert selection via frozen tid2eid[input_ids] lookup table instead of
learned argmax. The learned gate.weight still produces per-expert scores
for activation weighting of the hash-selected experts.
"""

from typing import Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor

from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig


class DeepseekV4HashRouter(MegatronModule):
    """Hash routing for first N MoE layers (paper section 2.1).

    Expert indices come from frozen tid2eid[input_ids] lookup (not learned
    argmax). gate.weight produces scores only for activation weighting of
    the hash-selected experts.
    """

    def __init__(
        self,
        config: TransformerConfig,
        num_local_experts: int,
        topk: int,
    ):
        super().__init__(config=config)
        self.topk = topk
        self.num_experts = num_local_experts
        self.hidden_dim = config.hidden_size

        # Learnable gate weights for scoring (not for expert selection)
        self.weight = torch.nn.Parameter(
            torch.empty(num_local_experts, self.hidden_dim)
        )

        # Frozen token-id -> expert-id lookup table
        vocab_size = config.padded_vocab_size if hasattr(config, "padded_vocab_size") else config.vocab_size
        self.register_buffer(
            "tid2eid",
            torch.zeros(vocab_size, self.topk, dtype=torch.long),
            persistent=True,
        )

    def forward(
        self,
        hidden_states: Tensor,
        input_ids: Tensor = None,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Route tokens via hash lookup.

        Args:
            hidden_states: [B * S, H] flattened hidden states
            input_ids: [B, S] token ids for tid2eid lookup

        Returns:
            logits, weights, indices
        """
        flat = hidden_states.reshape(-1, self.hidden_dim)
        logits = F.linear(flat, self.weight)
        scores = torch.sigmoid(logits)  # V4 hash gate scoring (same as TopKRouter)

        # Expert selection via frozen hash table
        indices = self.tid2eid[input_ids.reshape(-1)].long()
        weights = scores.gather(1, indices)
        weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-20)

        # Apply routed scaling factor
        routed_scaling_factor = getattr(self.config, "routed_scaling_factor", 1.5)
        return logits, weights * routed_scaling_factor, indices

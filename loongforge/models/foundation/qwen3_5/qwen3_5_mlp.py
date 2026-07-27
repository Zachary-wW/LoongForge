# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Qwen3.5 MLP variants."""

import torch
import torch.nn.functional as F

from megatron.core.transformer.mlp import MLP
from megatron.core.utils import nvtx_range_pop, nvtx_range_push


class Qwen35DenseMLP(MLP):
    """Dense MLP with HF-equivalent split gate/up projections.

    The checkpoint keeps Megatron's merged ``linear_fc1.weight`` layout
    ``[gate_proj; up_proj]``. During forward we run the two BF16 GEMMs
    separately to match HuggingFace Qwen3.5 numerics.
    """

    def forward(self, hidden_states, per_token_scale=None):
        """Perform the forward pass through the dense MLP block."""
        if per_token_scale is not None:
            raise NotImplementedError("Qwen35DenseMLP does not support per_token_scale yet.")
        if self.config.add_bias_linear:
            raise NotImplementedError("Qwen35DenseMLP expects bias-free Qwen3.5 dense MLP.")
        if self.config.activation_func_clamp_value is not None:
            raise NotImplementedError("Qwen35DenseMLP does not support activation clamping.")
        if not self.config.gated_linear_unit:
            return super().forward(hidden_states, per_token_scale=per_token_scale)

        nvtx_range_push(suffix="linear_fc1_split")
        gate_weight, up_weight = torch.split(
            self.linear_fc1.weight,
            self.config.ffn_hidden_size,
            dim=0,
        )
        gate = F.linear(hidden_states, gate_weight)
        up = F.linear(hidden_states, up_weight)
        intermediate_parallel = self.config.activation_func(gate) * (
            up + self.config.glu_linear_offset
        )
        nvtx_range_pop(suffix="linear_fc1_split")

        nvtx_range_push(suffix="linear_fc2")
        output, output_bias = self.linear_fc2(intermediate_parallel)
        nvtx_range_pop(suffix="linear_fc2")

        return output, output_bias

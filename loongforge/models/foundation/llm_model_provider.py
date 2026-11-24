# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
#
# Modified from Megatron-LM under the BSD 3-Clause License.
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

"""llm model provider"""

from typing import Optional

from loongforge.utils import (
    get_args, get_model_config, print_rank_0
)
from loongforge.models.model_backend import get_model_backend


def llm_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    parallel_output: bool = True,
    vp_stage: Optional[int] = None,
):
    """Generic LLM model provider.
    
    Args:
        pre_process (bool, optional): Set to true if you need to compute embedings. Defaults to True.
        post_process (bool, optional): Set to true if you need to want to compute output logits/loss. Defaults to True.
        parallel_output (bool): whether to allgather the output logits. Defaults to True.
    
    Returns:
        The corresponding LLM model.
    """
    args = get_args()

    config = get_model_config()

    print_rank_0(f"Building {config.model_type} model...")
    if args.use_legacy_models:
        raise ValueError("Classic Megatron-LM models are not supported.")
    
    # TODO: abstract this to a common dataclass
    config.padded_vocab_size = args.padded_vocab_size
    config.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
    config.rotary_seq_len_interpolation_factor = args.rotary_seq_len_interpolation_factor
    config.use_rope_scaling = args.use_rope_scaling
    config.rope_scaling_factor = args.rope_scaling_factor
    config.max_position_embeddings = args.max_position_embeddings
    config.rotary_base = args.rotary_base
    config.rotary_percent = args.rotary_percent
    config.use_rope_scaling = args.use_rope_scaling
    config.rope_scaling_factor = args.rope_scaling_factor
    config.rotary_seq_len_interpolation_factor = args.rotary_seq_len_interpolation_factor

    return get_model_backend(config).build_model(
        config=config,
        args=args,
        pre_process=pre_process,
        post_process=post_process,
        parallel_output=parallel_output,
        vp_stage=vp_stage,
    )

# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Model provider for the Phase 3 Qwen2.5-VL MiMo path."""

from __future__ import annotations

from typing import Optional

import torch
from megatron.core.models.mimo import MimoModel, MimoModelConfig
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.process_groups_config import ProcessGroupCollection
from megatron.core.transformer.spec_utils import ModuleSpec

from loongforge.models.encoder.qwen2_vl_vision_models.vision_model import (
    Qwen2VisionModelWithRMSNorm,
)
from loongforge.models.encoder.qwen2_vl_vision_models.adapter import Adapter
from loongforge.models.foundation import get_foundation_model_classes
from loongforge.models.mimo_models.mimo_vlm_model import LoongForgeMimoVLMModel
from loongforge.models.mimo_models.qwen25_vl import Qwen25VisionModalitySubmodules
from loongforge.utils import get_args, get_model_config


_TOPOLOGY_CACHE = {}


def _get_topology(args, image_tp: int):
    """Create one explicit PGC topology shared by all virtual PP chunks."""
    from examples.mimo.training.topology import ModuleGridSpec, create_topology

    world = torch.distributed.get_world_size()
    key = (
        world,
        args.tensor_model_parallel_size,
        args.context_parallel_size,
        args.pipeline_model_parallel_size,
        image_tp,
    )
    topology = _TOPOLOGY_CACHE.get(key)
    if topology is None:
        topology = create_topology(
            [
                ModuleGridSpec(
                    name=MIMO_LANGUAGE_MODULE_KEY,
                    num_ranks=world,
                    tp=args.tensor_model_parallel_size,
                    cp=args.context_parallel_size,
                    pp=args.pipeline_model_parallel_size,
                ),
                ModuleGridSpec(
                    name="images",
                    num_ranks=world,
                    tp=image_tp,
                    cp=1,
                    pp=1,
                ),
            ]
        )
        _TOPOLOGY_CACHE[key] = topology
    return topology


def _vision_spec(config, vision_pg_collection: ProcessGroupCollection) -> ModuleSpec:
    return ModuleSpec(
        module=Qwen25VisionModalitySubmodules,
        params={"pg_collection": vision_pg_collection},
        submodules={
            "encoders": {
                "image_encoder": ModuleSpec(
                    module=Qwen2VisionModelWithRMSNorm,
                    params={"config": config.image_encoder},
                )
            },
            "input_projections": [
                ModuleSpec(
                    module=Adapter,
                    params={
                        "config": config.image_projector,
                        "input_size": config.image_encoder.hidden_size,
                        "output_size": config.foundation.hidden_size,
                        "spatial_merge_size": getattr(
                            config.image_encoder, "spatial_merge_size", 2
                        ),
                        "use_postshuffle_norm": getattr(
                            config.image_encoder, "use_postshuffle_norm", False
                        ),
                    },
                )
            ],
        },
    )


def mimo_vlm_model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    parallel_output: bool = True,
    vp_stage: Optional[int] = None,
):
    args = get_args()
    config = get_model_config()
    if args.enable_encoder_hetero_dp or args.enable_full_hetero_dp:
        raise NotImplementedError("Heterogeneous encoder DP is deferred until Phase 6")
    if args.use_vit_dp_balance or args.use_vlm_dp_balance:
        raise NotImplementedError("VLM/ViT DP balance is deferred until Phase 6")
    if args.enable_chunkpipe:
        raise NotImplementedError("MiMo VLM ChunkPipe is deferred until Phase 6")

    foundation = config.foundation
    foundation.padded_vocab_size = args.padded_vocab_size
    foundation.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy
    foundation.rotary_seq_len_interpolation_factor = (
        args.rotary_seq_len_interpolation_factor
    )
    foundation.use_rope_scaling = args.use_rope_scaling
    foundation.rope_scaling_factor = args.rope_scaling_factor
    foundation.max_position_embeddings = args.max_position_embeddings
    foundation.rotary_base = args.rotary_base
    foundation.rotary_percent = args.rotary_percent

    image_encoder = config.image_encoder
    image_encoder.pipeline_model_parallel_size = 1
    image_encoder.context_parallel_size = 1
    image_encoder.sequence_parallel = False
    image_encoder.tp_comm_overlap = False

    _, language_class = get_foundation_model_classes(foundation.model_type)
    language_spec = ModuleSpec(
        module=language_class,
        params={
            "config": foundation,
            "pre_process": pre_process,
            "post_process": post_process,
            "parallel_output": parallel_output,
            "scatter_embedding_sequence_parallel": False,
            "vp_stage": vp_stage,
        },
    )

    image_token_id = getattr(image_encoder, "image_token_id", None)
    if image_token_id is None:
        raise ValueError("Qwen2.5-VL image_encoder config is missing image_token_id")

    image_tp = image_encoder.tensor_model_parallel_size
    world = torch.distributed.get_world_size()
    if world % image_tp:
        raise ValueError(
            f"world_size={world} is not divisible by image encoder TP={image_tp}"
        )
    # Vision is not VPP-sharded. Build it only in the first virtual chunk;
    # later chunks are language-only views over the same shared topology.
    topology = _get_topology(args, image_tp)
    language_pg = topology.module_pgs[MIMO_LANGUAGE_MODULE_KEY]
    vision_pg = topology.module_pgs["images"]
    language_spec.params["pg_collection"] = language_pg
    owns_vision = vp_stage in (None, 0) and pre_process
    modality_specs = {"images": _vision_spec(config, vision_pg)} if owns_vision else {}
    special_token_ids = {"images": image_token_id} if owns_vision else {}
    if args.pipeline_model_parallel_size == 1:
        module_to_grid_map = (
            topology.grids
            if owns_vision
            else {MIMO_LANGUAGE_MODULE_KEY: topology.grids[MIMO_LANGUAGE_MODULE_KEY]}
        )
    else:
        # ColocatedBridgeCommunicator currently requires language PP=1. With
        # language DP=1 every vision TP/DP replica sees the same microbatch, so
        # PP-first can consume its local projected image embeddings directly.
        # Later PP/VPP chunks dispatch to the MiMo-owned language stage in the
        # wrapper. Keep the explicit PGC on the module specs and wrapper.
        language_grid = topology.grids[MIMO_LANGUAGE_MODULE_KEY]
        language_dp = language_grid.shape[language_grid.dim_names.index("dp")]
        if language_dp != 1:
            raise NotImplementedError(
                "MiMo with language PP>1 currently requires language DP=1 "
                "so PP-first can consume local replicated vision embeddings"
            )
        module_to_grid_map = None
    mimo_config = MimoModelConfig(
        language_model_spec=language_spec,
        modality_submodules_spec=modality_specs,
        special_token_ids=special_token_ids,
        module_to_grid_map=module_to_grid_map,
        kv_format="sbhd",
    )
    mimo_model = MimoModel(
        mimo_config,
        cp_group=language_pg.cp,
        tp_group=language_pg.tp,
    )
    return LoongForgeMimoVLMModel(
        mimo_model=mimo_model,
        topology=topology,
        model_type=config.model_type,
        pre_process=pre_process,
        post_process=post_process,
    )

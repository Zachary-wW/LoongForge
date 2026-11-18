# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""LoongForge wrapper around the native MCore MiMo multimodal model."""

from __future__ import annotations

from typing import Any, Optional

import torch

from megatron.core.models.mimo import MimoModel
from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY
from megatron.core.process_groups_config import MultiModuleProcessGroupCollection


_LEGACY_OMNI_PREFIX_MAP = {
    "foundation_model.": "language_model.",
    "encoder_model.image_encoder.": (
        "modality_submodules.images.encoders.image_encoder."
    ),
    "encoder_model.image_projector.": (
        "modality_submodules.images.input_projections.0."
    ),
}
_LEGACY_OMNI_ALIASES = {
    # This weight aliases foundation_model.embedding.word_embeddings.weight in
    # OmniCombinationModel; MiMo has only the language model copy.
    "encoder_model.text_encoder.word_embeddings.weight",
}


def remap_legacy_omni_state_dict(state_dict):
    """Map a legacy Omni checkpoint into the native MiMo module tree."""
    if not any(
        key.startswith(tuple(_LEGACY_OMNI_PREFIX_MAP))
        or key in _LEGACY_OMNI_ALIASES
        for key in state_dict
    ):
        return state_dict

    remapped = {}
    for key, value in state_dict.items():
        if key in _LEGACY_OMNI_ALIASES:
            continue
        for source_prefix, target_prefix in _LEGACY_OMNI_PREFIX_MAP.items():
            if key.startswith(source_prefix):
                key = target_prefix + key[len(source_prefix) :]
                break
        remapped[key] = value
    return remapped


class LoongForgeMimoVLMModel(torch.nn.Module):
    """Expose the existing LoongForge VLM forward contract on top of MiMo."""

    def __init__(
        self,
        *,
        mimo_model: MimoModel,
        topology,
        model_type: str,
        pre_process: bool,
        post_process: bool,
    ) -> None:
        super().__init__()
        self.mimo_model = mimo_model
        self.topology = topology
        self.config = mimo_model.config
        self.model_type = model_type
        self.add_encoder = len(mimo_model.modality_submodules) > 0
        self.add_decoder = mimo_model.language_model is not None
        self.disable_param_offloading = True
        self.pre_process = pre_process
        self.post_process = post_process
        self.pg_collection = topology.module_pgs[MIMO_LANGUAGE_MODULE_KEY]
        self.share_embeddings_and_output_weights = getattr(
            getattr(mimo_model.language_model, "module", mimo_model.language_model),
            "share_embeddings_and_output_weights",
            False,
        )
        self.vp_stage = mimo_model.mimo_config.language_model_spec.params.get(
            "vp_stage"
        )
        active_module_names = [MIMO_LANGUAGE_MODULE_KEY]
        active_module_names.extend(mimo_model.modality_submodules.keys())
        active_module_pgs = {
            name: topology.module_pgs[name] for name in active_module_names
        }
        self.module_pg_collection = MultiModuleProcessGroupCollection(
            module_pgs=active_module_pgs,
            language_model_module_name=MIMO_LANGUAGE_MODULE_KEY,
        )

    def set_input_tensor(self, input_tensor) -> None:
        self.mimo_model.set_input_tensor(input_tensor)

    def shared_embedding_or_output_weight(self):
        language = self.mimo_model.language_model
        return language.shared_embedding_or_output_weight()

    def state_dict(self, *args, **kwargs):
        return self.mimo_model.state_dict(*args, **kwargs)

    def load_state_dict(self, state_dict, strict=True, assign=False):
        return self.mimo_model.load_state_dict(
            remap_legacy_omni_state_dict(state_dict),
            strict=strict,
            assign=assign,
        )

    def sharded_state_dict(self, prefix="", sharded_offsets=(), metadata=None):
        return self.mimo_model.sharded_state_dict(
            prefix=prefix,
            sharded_offsets=sharded_offsets,
            metadata=metadata,
        )

    def state_dict_for_save_checkpoint(self, prefix="", keep_vars=False):
        return self.mimo_model.state_dict(prefix=prefix, keep_vars=keep_vars)

    def train(self, mode: bool = True):
        super().train(mode)
        self.mimo_model.train(mode)
        return self

    def forward(
        self,
        image_inputs: Optional[dict[str, torch.Tensor]] = None,
        video_inputs: Optional[dict[str, torch.Tensor]] = None,
        audio_inputs: Optional[dict[str, torch.Tensor]] = None,
        *,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        packed_seq_params: Any = None,
        labels: Optional[torch.Tensor] = None,
        loss_mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        if video_inputs is not None:
            raise NotImplementedError("Phase 3 Qwen2.5-VL MiMo smoke supports images only")
        if audio_inputs is not None:
            raise NotImplementedError("Qwen2.5-VL does not support audio inputs")
        if packed_seq_params is not None:
            raise NotImplementedError(
                "Phase 3 MiMo smoke supports SBHD only; packed sequence is disabled"
            )

        language_model = self.mimo_model.language_model
        language_module = getattr(language_model, "module", language_model)
        if language_module is not None and not language_module.pre_process:
            # Non-first PP/VPP chunks contain only a language-model stage. The
            # MimoModel colocated front-end requires an embedding and therefore
            # cannot be called here; dispatch the language stage directly while
            # preserving the same MiMo-owned module and PGC.
            output = language_model(
                input_ids=input_ids,
                position_ids=position_ids,
                decoder_input=None,
                labels=labels,
                loss_mask=loss_mask,
                attention_mask=attention_mask,
                packed_seq_params=packed_seq_params,
            )
            if labels is not None:
                return output
            return output, loss_mask

        modality_inputs = None
        if image_inputs is not None:
            modality_inputs = {
                "images": {
                    "image_encoder": {
                        "images": image_inputs["images"],
                        "image_grid_thw": image_inputs["image_grid_thw"],
                    }
                }
            }

        output, output_loss_mask = self.mimo_model(
            input_ids=input_ids,
            position_ids=position_ids,
            attention_mask=attention_mask,
            labels=labels,
            loss_mask=loss_mask,
            modality_inputs=modality_inputs,
        )
        if labels is not None:
            return output
        return output, output_loss_mask

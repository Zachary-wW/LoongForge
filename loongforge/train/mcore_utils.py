# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Small helpers shared by the MCore runtime adapter."""

from __future__ import annotations

import os


def is_hf_checkpoint(load_path):
    """Return whether a path contains a supported HuggingFace checkpoint."""
    if load_path is None:
        return False
    names = (
        "model.safetensors.index.json",
        "model.safetensors",
        "pytorch_model.bin.index.json",
        "pytorch_model.bin",
    )
    return any(os.path.exists(os.path.join(load_path, name)) for name in names)


def resolve_convert_pp_layout(model_config):
    """Return the converter pipeline layout, including VLM foundation layouts."""
    layout = getattr(model_config, "pipeline_model_parallel_layout", None)
    if layout is None:
        foundation = getattr(model_config, "foundation", None)
        if foundation is not None:
            layout = getattr(foundation, "pipeline_model_parallel_layout", None)
    if layout is None or getattr(layout, "layout", None) is None:
        return None
    return layout


def configure_mimo_grad_finalization(wrappers):
    """Finalize each MiMo module over its own process groups."""
    from megatron.core.distributed.finalize_model_grads import finalize_model_grads
    from megatron.core.models.mimo.config.role import MIMO_LANGUAGE_MODULE_KEY

    module_pgs = wrappers[0].topology.module_pgs
    mimo_models = [wrapper.mimo_model for wrapper in wrappers]

    def finalize_grads_func(_model_list, _num_tokens, force_all_reduce=False, **_kwargs):
        finalize_model_grads(
            [mimo_model.language_model for mimo_model in mimo_models],
            num_tokens=None,
            pg_collection=module_pgs[MIMO_LANGUAGE_MODULE_KEY],
            force_all_reduce=force_all_reduce,
        )
        for mimo_model in mimo_models:
            for name, submodule in mimo_model.modality_submodules.items():
                finalize_model_grads(
                    [submodule],
                    num_tokens=None,
                    pg_collection=module_pgs[name],
                    force_all_reduce=force_all_reduce,
                )
                bridge = mimo_model.colocated_comms.get((name, MIMO_LANGUAGE_MODULE_KEY))
                if bridge is not None and bridge.is_fan_in():
                    submodule.scale_gradients(bridge.scale)

    config = mimo_models[0].config
    config.finalize_model_grads_func = finalize_grads_func

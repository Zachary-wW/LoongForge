# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Small helpers shared by the MCore runtime adapter."""

from __future__ import annotations

import os

from megatron.core.msc_utils import maybe_msc
from megatron.training.checkpointing import get_checkpoint_tracker_filename


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


def read_tracker_iteration(checkpoints_path):
    """Read a tracker without exiting on missing or malformed files."""
    tracker_filename = get_checkpoint_tracker_filename(checkpoints_path)
    if not os.path.isfile(tracker_filename):
        return None
    try:
        with maybe_msc.open(tracker_filename, "r") as stream:
            metastring = stream.read().strip()
    except OSError:
        return None
    if metastring == "release":
        return (0, True)
    try:
        iteration = int(metastring)
    except ValueError:
        return None
    if iteration <= 0:
        return None
    return (iteration, False)


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

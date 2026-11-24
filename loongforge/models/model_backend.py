# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Model backend seam for GPT, Hybrid, and future MiMo model families."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ClassVar, Mapping

import torch
from transformers import AutoModel

from loongforge.models.common import BaseMegatronLanguageModule


@dataclass(frozen=True)
class BackendCapabilities:
    """Features whose behavior is owned by a model backend."""

    hybrid_model: bool = False
    mtp: bool = True
    packed_sequence: bool = True
    chunkpipe: bool = True
    fine_grained_offload: bool = True
    direct_hf_checkpoint: bool = True
    checkpoint_state_layout: str = "gpt"


class ModelBackend(ABC):
    """Construct a model and translate the common training contract for it."""

    name: ClassVar[str]
    capabilities: ClassVar[BackendCapabilities]

    def build_model(
        self,
        *,
        config: Any,
        args: Any,
        pre_process: bool,
        post_process: bool,
        parallel_output: bool,
        vp_stage: int | None,
    ) -> BaseMegatronLanguageModule:
        """Build the backend model through the registered AutoModel class."""
        build_model_context = nullcontext
        build_model_context_args: dict[str, Any] = {}
        if args.fp8_param_gather and not getattr(config, "selective_fp8", False):
            try:
                from transformer_engine.pytorch import fp8_model_init
            except ImportError as error:
                raise RuntimeError(
                    "--fp8-param-gather requires `fp8_model_init` from TransformerEngine, "
                    "but it was not found."
                ) from error

            build_model_context = fp8_model_init
            build_model_context_args["enabled"] = True
            if "preserve_high_precision_init_val" in inspect.signature(fp8_model_init).parameters:
                build_model_context_args["preserve_high_precision_init_val"] = True

        with build_model_context(**build_model_context_args):
            model = AutoModel.from_config(
                config=config,
                pre_process=pre_process,
                post_process=post_process,
                parallel_output=parallel_output,
                vp_stage=vp_stage,
            )

        self.validate_model(model)
        self._validate_selective_fp8(model, config)
        return model

    @abstractmethod
    def validate_model(self, model: torch.nn.Module) -> None:
        """Reject models constructed by the wrong backend."""

    def pretrain_forward_kwargs(
        self,
        *,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        extra_block_kwargs: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Translate a pretrain LLM batch to model-forward arguments."""
        return {
            "input_ids": tokens,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "loss_mask": loss_mask,
            "extra_block_kwargs": dict(extra_block_kwargs) if extra_block_kwargs else None,
        }

    def sft_forward_kwargs(
        self,
        *,
        tokens: torch.Tensor,
        labels: torch.Tensor,
        loss_mask: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        packed_seq_params: Any,
        extra_block_kwargs: Mapping[str, Any] | None,
    ) -> dict[str, Any]:
        """Translate an SFT LLM batch to model-forward arguments."""
        return {
            "input_ids": tokens,
            "position_ids": position_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "packed_seq_params": packed_seq_params,
            "loss_mask": loss_mask,
            "extra_block_kwargs": dict(extra_block_kwargs) if extra_block_kwargs else None,
        }

    @staticmethod
    def extract_loss(output: Any) -> torch.Tensor:
        """Extract a loss tensor from a tensor or structured model output."""
        if isinstance(output, torch.Tensor):
            return output
        loss = getattr(output, "loss", None)
        if loss is None:
            raise TypeError(
                f"{type(output).__name__} model output is neither a tensor nor exposes .loss"
            )
        return loss

    @staticmethod
    def _has_selective_fp8(config: Any) -> bool:
        if getattr(config, "selective_fp8", False):
            return True
        for attribute in ("foundation", "image_encoder", "video_encoder", "audio_encoder"):
            subconfig = getattr(config, attribute, None)
            if subconfig is not None and getattr(subconfig, "selective_fp8", False):
                return True
        return False

    def _validate_selective_fp8(self, model: torch.nn.Module, config: Any) -> None:
        if not self._has_selective_fp8(config):
            return
        try:
            from megatron.core.fp8_utils import validate_selective_fp8_coverage
        except ImportError as error:
            raise RuntimeError(
                "selective_fp8 is enabled, but the installed MCore does not expose "
                "validate_selective_fp8_coverage"
            ) from error
        validate_selective_fp8_coverage(model)


class GPTBackend(ModelBackend):
    """Backend for the LoongForge GPT/BaseGPT family."""

    name = "gpt"
    capabilities = BackendCapabilities()

    def validate_model(self, model: torch.nn.Module) -> None:
        from megatron.core.models.hybrid.hybrid_model import HybridModel

        if isinstance(model, HybridModel):
            raise TypeError("GPTBackend cannot execute a HybridModel")


class HybridBackend(ModelBackend):
    """Backend for native MCore HybridModel families."""

    name = "hybrid"
    capabilities = BackendCapabilities(
        hybrid_model=True,
        packed_sequence=False,
        chunkpipe=False,
        fine_grained_offload=False,
        checkpoint_state_layout="hybrid",
    )

    def validate_model(self, model: torch.nn.Module) -> None:
        from megatron.core.models.hybrid.hybrid_model import HybridModel

        if not isinstance(model, HybridModel):
            raise TypeError("HybridBackend requires a HybridModel")


_BACKENDS: dict[str, ModelBackend] = {
    GPTBackend.name: GPTBackend(),
    HybridBackend.name: HybridBackend(),
}


def register_backend(backend: ModelBackend) -> None:
    """Register a backend by its stable config name."""
    if backend.name in _BACKENDS:
        raise ValueError(f"Model backend {backend.name!r} is already registered")
    _BACKENDS[backend.name] = backend


def get_model_backend(config: Any) -> ModelBackend:
    """Resolve the backend declared by a model config."""
    backend_name = getattr(config, "model_backend", None) or "gpt"
    try:
        return _BACKENDS[backend_name]
    except KeyError as error:
        raise ValueError(
            f"Unknown model backend {backend_name!r}; available: {sorted(_BACKENDS)}"
        ) from error

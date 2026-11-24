"""LoongForge model and HF weight adapters for MCore pretraining."""

from copy import copy
from dataclasses import dataclass
from functools import partial
from typing import Callable, ClassVar

from loongforge.train.parser import parse_args_from_config
from loongforge.train.mcore_utils import (
    configure_mimo_grad_finalization,
    is_hf_checkpoint,
    resolve_convert_pp_layout,
)
from loongforge.utils import constants
from loongforge.utils.global_vars import get_model_config, set_loongforge_extra_global_vars

from megatron.core.transformer import TransformerConfig
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.callbacks import CallbackManager
from megatron.training.global_vars import get_args, set_global_variables
from megatron.training.models.base import ModelBuilder, ModelConfig, compose_hooks
from megatron.training.models.dist_utils import (
    build_virtual_pipeline_stages,
    unimodal_build_distributed_models,
)
from megatron.training.training import pretrain


@dataclass(kw_only=True)
class LoongForgeModelConfig(ModelConfig):
    transformer: TransformerConfig
    provider: Callable
    builder: ClassVar[str] = "loongforge.train.mcore_runtime.LoongForgeModelBuilder"


@dataclass(kw_only=True)
class LoongForgeMimoConfig(LoongForgeModelConfig):
    builder: ClassVar[str] = "loongforge.train.mcore_runtime.LoongForgeMimoBuilder"


class LoongForgeModelBuilder(ModelBuilder):
    def build_model(self, pg_collection, pre_process=None, post_process=None, vp_stage=None):
        return self._model_config.provider(
            pre_process=pre_process, post_process=post_process, vp_stage=vp_stage
        )

    def build_distributed_models(self, pg_collection, **kwargs):
        model_config = self._model_config
        model = unimodal_build_distributed_models(
            self.build_model,
            model_config.transformer,
            pg_collection,
            pre_wrap_hook=compose_hooks(model_config.pre_wrap_hooks),
            **kwargs,
        )
        return compose_hooks(model_config.post_wrap_hooks)(model)


class LoongForgeMimoBuilder(LoongForgeModelBuilder):
    def build_distributed_models(
        self,
        pg_collection,
        *,
        ddp_config,
        wrap_with_ddp,
        data_parallel_random_init,
        use_layer_wise_distributed_optimizer,
        use_layer_wise_param_layout,
        use_megatron_fsdp,
        use_torch_fsdp2,
        overlap_param_gather_with_optimizer_step,
    ):
        import torch

        from examples.mimo.training.runtime import wrap_active_modules_with_ddp
        from megatron.training import get_args

        args = get_args()
        if use_megatron_fsdp or use_torch_fsdp2:
            raise NotImplementedError("The MiMo runtime supports per-module DDP")
        if args.calculate_per_token_loss:
            raise NotImplementedError("The MiMo runtime requires per-microbatch loss normalization")
        model_config = self._model_config
        models = build_virtual_pipeline_stages(
            self.build_model,
            pg_collection,
            model_config.transformer.virtual_pipeline_model_parallel_size,
        )
        models = compose_hooks(model_config.pre_wrap_hooks)(models)
        for model in models:
            model.cuda(torch.cuda.current_device())
            if wrap_with_ddp:
                wrap_active_modules_with_ddp(
                    args,
                    model.mimo_model,
                    model.topology,
                    ddp_config=ddp_config,
                    data_parallel_random_init=data_parallel_random_init,
                    use_layer_wise_distributed_optimizer=use_layer_wise_distributed_optimizer,
                    use_layer_wise_param_layout=use_layer_wise_param_layout,
                )
        if wrap_with_ddp:
            configure_mimo_grad_finalization(models)
        return compose_hooks(model_config.post_wrap_hooks)(models)

    def build_optimizer(self, config, model_chunks, **kwargs):
        from megatron.core.models.mimo import get_mimo_optimizer

        if not config.use_distributed_optimizer:
            raise NotImplementedError("The MiMo runtime requires a distributed optimizer")
        return get_mimo_optimizer(
            [chunk.mimo_model for chunk in model_chunks],
            config,
            module_to_grid_map=model_chunks[0].topology.grids,
            config_overrides=kwargs.get("config_overrides"),
            use_gloo_process_groups=False,
            dump_param_to_param_group_map=kwargs.get("dump_param_to_param_group_map"),
        )


def _load_hf_weights(context, *, args, model_config):
    from tools.mcore_bridge.checkpoint.hf_checkpoint_loader import load_hf_checkpoint_online

    layout = resolve_convert_pp_layout(model_config)
    if layout is not None:
        args.convert_pp_layout = layout
    load_hf_checkpoint_online(
        context.model, context.optimizer, context.scheduler, args, model_config=model_config
    )


def _restore_variable_seq_lengths(model_config):
    """Restore LoongForge's batch contract after MCore argument validation."""
    runtime_args = get_args()
    runtime_args.variable_seq_lengths = True
    for target in (
        model_config,
        getattr(model_config, "foundation", None),
        getattr(model_config, "image_encoder", None),
    ):
        if target is not None and hasattr(target, "variable_seq_lengths"):
            target.variable_seq_lengths = True


def normalize_training_args(train_args):
    """Map LoongForge training-phase semantics onto MCore flags."""
    if getattr(train_args, "training_phase", None) == constants.TrainingPhase.SFT:
        train_args.sft = True
        if getattr(train_args, "enable_chunkpipe", False):
            train_args.sft_chunkpipe_mode = True
        else:
            # MCore sets sft_chunkpipe_mode=True whenever --sft is present;
            # non-ChunkPipe SFT must not take the ChunkPipe model branches.
            train_args.sft_chunkpipe_mode = False
        train_args.sft_loss_log_mode = "microbatch"
        if _sft_uses_variable_seq_lengths(train_args):
            # MCore's SFT argument validation resets this after LoongForge's
            # validator runs. Restore the legacy SFT padding semantics: with
            # variable sequence lengths the collator pads to the longest
            # sequence in each microbatch, while fixed mode pads to seq_length.
            train_args.variable_seq_lengths = True
        else:
            train_args.variable_seq_lengths = False
    return train_args


def _sync_chunkpipe_model_config(model_config, enabled):
    """Keep MCore and LoongForge model views of the ChunkPipe mode aligned."""
    for target in (
        model_config,
        getattr(model_config, "foundation", None),
        getattr(model_config, "image_encoder", None),
    ):
        if target is not None and hasattr(target, "sft_chunkpipe_mode"):
            target.sft_chunkpipe_mode = enabled


def _validate_chunkpipe_entrypoint(train_args, model_config, *, allow_chunkpipe):
    """Fail fast unless this is the supported LLM-SFT ChunkPipe entrypoint."""
    if not getattr(train_args, "enable_chunkpipe", False):
        return
    if not allow_chunkpipe:
        raise NotImplementedError(
            "ChunkPipe is currently supported only by the LLM SFT trainer. "
            "Pretrain, VLM, InternVL, Ernie, MiMo and Omni entrypoints stay disabled."
        )
    if getattr(train_args, "training_phase", None) != constants.TrainingPhase.SFT:
        raise NotImplementedError("ChunkPipe is currently supported only for SFT.")
    if _is_vlm_config(model_config) or getattr(model_config, "model_backend", None) in (
        "mimo",
        "omni",
    ):
        raise NotImplementedError("ChunkPipe is currently supported only for LLM SFT.")
    if getattr(train_args, "rerun_mode", None) != "disabled":
        raise NotImplementedError(
            "SFT ChunkPipe requires --rerun-mode disabled until sampler metadata "
            "is replayed from the batch itself."
        )
    if getattr(train_args, "eval_iters", 0):
        raise NotImplementedError(
            "SFT ChunkPipe currently requires eval_iters=0; validation loss reduction "
            "for the per-sample path is not yet migrated."
        )
    if getattr(train_args, "calculate_per_token_loss", False):
        raise NotImplementedError(
            "SFT ChunkPipe currently supports the per-sample loss path only; "
            "disable --calculate-per-token-loss."
        )
    if getattr(train_args, "tensor_model_parallel_size", 1) != 1:
        raise NotImplementedError(
            "SFT ChunkPipe currently requires tensor_model_parallel_size=1; "
            "multi-TP metadata queue ownership is not yet validated."
        )
    if getattr(train_args, "num_virtual_stages_per_pipeline_rank", 1) not in (None, 1):
        raise NotImplementedError(
            "SFT ChunkPipe currently requires virtual pipeline size 1; "
            "multi-VPP metadata queue ownership is not yet validated."
        )


def _validate_linear_ce_entrypoint(train_args):
    """Reject the unsupported linear-CE + param-gather-overlap combination."""
    if not (
        getattr(train_args, "cross_entropy_loss_fusion", False)
        and getattr(train_args, "cross_entropy_fusion_impl", "native") == "linear"
    ):
        return
    if getattr(train_args, "overlap_param_gather", False):
        raise NotImplementedError(
            "cross_entropy_fusion_impl='linear' currently requires "
            "overlap_param_gather=False. MCore DDP's output-layer bucket pre-hook "
            "is bypassed by the linear loss, and the overlap dispatcher cannot "
            "preserve bucket order for this path."
        )


def _sft_uses_variable_seq_lengths(train_args):
    """Mirror LoongForge SFT validation for the MCore argument boundary."""
    if getattr(train_args, "use_fixed_seq_lengths", False):
        return False
    if getattr(train_args, "tp_comm_overlap", False):
        return False
    if getattr(train_args, "num_experts", 0) and getattr(
        train_args, "moe_token_dispatcher_type", None
    ) in ("allgather", "alltoall_seq"):
        return False
    return True


def _is_vlm_config(model_config):
    return (
        getattr(model_config, "image_encoder", None) is not None
        or getattr(model_config, "model_backend", None) == "mimo"
    )


def _is_llava_ov_config(model_config):
    """Return whether this is the LLaVA-OneVision/RiceViT pretrain family."""
    return (
        getattr(model_config, "model_type", None) == "llava_ov_1_5"
        or getattr(getattr(model_config, "image_encoder", None), "model_type", None)
        == "rice_vit"
    )


def _is_mimo_or_omni_config(model_config):
    """MiMo/Omni use the fixed-length pretrain contract."""
    return getattr(model_config, "model_backend", None) in ("mimo", "omni")


def _requires_variable_seq_lengths(train_args, model_config):
    """Preserve the legacy variable-length batch contract."""
    training_phase = getattr(train_args, "training_phase", None)
    if training_phase == constants.TrainingPhase.PRETRAIN:
        if _is_llava_ov_config(model_config):
            return True
        if _is_mimo_or_omni_config(model_config):
            return False
        return _is_vlm_config(model_config)
    if training_phase == constants.TrainingPhase.SFT:
        return _sft_uses_variable_seq_lengths(train_args)
    return False


def _requires_legacy_scalar_grad_norm(train_args, model_config):
    """Select the clip root used by the corresponding old-stack trainer."""
    training_phase = getattr(train_args, "training_phase", None)
    if training_phase == constants.TrainingPhase.PRETRAIN:
        if _is_mimo_or_omni_config(model_config):
            return False
        return _is_vlm_config(model_config) or _is_llava_ov_config(model_config)
    if training_phase == constants.TrainingPhase.SFT:
        # Legacy SFT starts from the scalar root; packed VLM batches override
        # this to the tensor root in pretrain_vlm.forward_step().
        return True
    return False


def _select_sft_clip_root(train_args, packed_seq_params):
    """Packed VLM SFT needs tensor root; non-packed SFT keeps scalar root."""
    if getattr(train_args, "training_phase", None) != constants.TrainingPhase.SFT:
        return
    if not _is_vlm_config(get_model_config()):
        return
    from megatron.core.optimizer.clip_grads import set_legacy_scalar_grad_norm

    set_legacy_scalar_grad_norm(packed_seq_params is None)


def pretrain_with_mcore(
    train_args,
    train_valid_test_dataset_provider,
    model_provider,
    model_type,
    forward_step_func,
    process_non_loss_data_func=None,
    get_embedding_ranks=None,
    get_position_embedding_ranks=None,
    allow_chunkpipe=False,
):
    normalize_training_args(train_args)
    parse_args_from_config(train_args)
    model_config = get_model_config()
    _validate_linear_ce_entrypoint(train_args)
    _validate_chunkpipe_entrypoint(
        train_args, model_config, allow_chunkpipe=allow_chunkpipe
    )
    from megatron.core.optimizer.clip_grads import set_legacy_scalar_grad_norm

    set_legacy_scalar_grad_norm(
        _requires_legacy_scalar_grad_norm(train_args, model_config)
    )
    if _requires_variable_seq_lengths(train_args, model_config):
        train_args.variable_seq_lengths = True
        for target in (model_config, getattr(model_config, "foundation", None)):
            if target is not None and hasattr(target, "variable_seq_lengths"):
                target.variable_seq_lengths = True
    if getattr(train_args, "training_phase", None) == constants.TrainingPhase.SFT:
        # The parser materializes the SFT mode into TransformerConfig before
        # the LoongForge adapter runs; keep the model and CLI views consistent.
        _sync_chunkpipe_model_config(
            model_config, getattr(train_args, "enable_chunkpipe", False)
        )
    if train_args.enable_ema or getattr(model_config, "peft_config", None) is not None:
        raise NotImplementedError("This MCore runtime adapter does not yet support EMA or PEFT")
    callbacks = CallbackManager()
    if _requires_variable_seq_lengths(train_args, model_config):
        callbacks.register(
            "on_data_init_start",
            lambda _context: _restore_variable_seq_lengths(model_config),
        )
    if getattr(model_config, "model_backend", None) == "mimo":
        from loongforge.train.initialize import initialize_mimo_vision_parallel_state

        cfg = LoongForgeMimoConfig(transformer=model_config.foundation, provider=model_provider)
        callbacks.register(
            "on_setup_start",
            lambda context: initialize_mimo_vision_parallel_state(model_config.image_encoder),
        )
    else:
        # VLM configs carry the language TransformerConfig under ``foundation``;
        # plain LLM configs already are TransformerConfig instances.
        transformer_config = getattr(model_config, "foundation", model_config)
        cfg = LoongForgeModelConfig(transformer=transformer_config, provider=model_provider)
        if getattr(model_config, "image_encoder", None) is not None:
            from loongforge.train.initialize import initialize_mimo_vision_parallel_state

            callbacks.register(
                "on_setup_start",
                lambda context: initialize_mimo_vision_parallel_state(model_config.image_encoder),
            )

    runtime_args = train_args
    if is_hf_checkpoint(train_args.load):
        if train_args.use_megatron_fsdp or train_args.moe_use_upcycling:
            raise NotImplementedError("Online HF loading does not support Megatron FSDP or upcycling")
        if train_args.pretrained_checkpoint is not None:
            raise ValueError("Specify one source when loading HF weights through --load")
        runtime_args = copy(train_args)
        runtime_args.load = None
        callbacks.register(
            "on_data_init_start",
            partial(_load_hf_weights, args=train_args, model_config=model_config),
        )

    if getattr(runtime_args, "training_phase", None) == constants.TrainingPhase.SFT:
        # Set these before the args are copied into the runtime config/global
        # state. The MCore training loop keys SFT loss reporting off these
        # fields, not only off training_phase.
        runtime_args.sft = True
        runtime_args.sft_chunkpipe_mode = getattr(
            runtime_args, "enable_chunkpipe", False
        )
        runtime_args.sft_loss_log_mode = "microbatch"
        runtime_args.variable_seq_lengths = _sft_uses_variable_seq_lengths(
            runtime_args
        )

    runtime_args.model_type = model_type
    config = pretrain_cfg_container_from_args(runtime_args, cfg)
    set_global_variables(runtime_args, config, build_tokenizer=False)
    set_loongforge_extra_global_vars(runtime_args, build_tokenizer=True)
    if getattr(model_config, "image_encoder", None) is not None:
        from loongforge.models.encoder.internvl_vision_models.internvl_config import (
            resolve_image_token_id,
        )

        resolve_image_token_id(model_config.image_encoder)
    return pretrain(
        config,
        train_valid_test_dataset_provider,
        model_type,
        forward_step_func,
        process_non_loss_data_func=process_non_loss_data_func,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
        callbacks=callbacks,
    )

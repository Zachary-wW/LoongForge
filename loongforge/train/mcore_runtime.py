"""LoongForge adapters for the MCore 0.20 training entrypoint."""

from copy import copy
from dataclasses import dataclass
from functools import partial
from typing import Callable, ClassVar

from loongforge.train.mcore_utils import is_hf_checkpoint, resolve_convert_pp_layout
from loongforge.train.parser import parse_args_from_config
from loongforge.utils import constants
from loongforge.utils.global_vars import get_model_config, set_loongforge_extra_global_vars

from megatron.core.transformer import TransformerConfig
from megatron.training.argument_utils import pretrain_cfg_container_from_args
from megatron.training.callbacks import CallbackManager
from megatron.training.global_vars import set_global_variables
from megatron.training.models.base import ModelBuilder, ModelConfig, compose_hooks
from megatron.training.models.dist_utils import unimodal_build_distributed_models
from megatron.training.training import pretrain


@dataclass(kw_only=True)
class LoongForgeModelConfig(ModelConfig):
    """Bridge a LoongForge model provider into the MCore model builder."""

    transformer: TransformerConfig
    provider: Callable
    builder: ClassVar[str] = "loongforge.train.mcore_runtime.LoongForgeModelBuilder"


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


def _load_hf_weights(context, *, args, model_config):
    from tools.mcore_bridge.checkpoint.hf_checkpoint_loader import load_hf_checkpoint_online

    layout = resolve_convert_pp_layout(model_config)
    if layout is not None:
        args.convert_pp_layout = layout
    load_hf_checkpoint_online(
        context.model, context.optimizer, context.scheduler, args, model_config=model_config
    )


def normalize_training_args(train_args):
    """Map LoongForge SFT flags onto the flags consumed by MCore."""
    if getattr(train_args, "training_phase", None) != constants.TrainingPhase.SFT:
        return train_args

    train_args.sft = True
    train_args.sft_chunkpipe_mode = False
    train_args.sft_loss_log_mode = "microbatch"
    return train_args


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
    """Run LoongForge through the MCore 0.20 config-container entrypoint."""
    if getattr(train_args, "enable_chunkpipe", False):
        raise NotImplementedError("ChunkPipe support is introduced in LF-C.")

    normalize_training_args(train_args)
    parse_args_from_config(train_args)
    model_config = get_model_config()

    if train_args.enable_ema or getattr(model_config, "peft_config", None) is not None:
        raise NotImplementedError("This MCore runtime adapter does not yet support EMA or PEFT")

    callbacks = CallbackManager()
    transformer_config = getattr(model_config, "foundation", model_config)
    cfg = LoongForgeModelConfig(transformer=transformer_config, provider=model_provider)

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
        runtime_args.sft = True
        runtime_args.sft_chunkpipe_mode = False
        runtime_args.sft_loss_log_mode = "microbatch"

    runtime_args.model_type = model_type
    config = pretrain_cfg_container_from_args(runtime_args, cfg)
    set_global_variables(runtime_args, config, build_tokenizer=False)
    set_loongforge_extra_global_vars(runtime_args, build_tokenizer=True)
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

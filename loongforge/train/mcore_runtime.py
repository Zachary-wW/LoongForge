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
from megatron.training.global_vars import set_global_variables
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


def normalize_training_args(train_args):
    """Map LoongForge training-phase semantics onto MCore flags."""
    if getattr(train_args, "training_phase", None) == constants.TrainingPhase.SFT:
        train_args.sft = True
        if not getattr(train_args, "enable_chunkpipe", False):
            # MCore sets sft_chunkpipe_mode=True whenever --sft is present;
            # non-ChunkPipe SFT must not take the ChunkPipe model branches.
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
):
    parse_args_from_config(train_args)
    normalize_training_args(train_args)
    model_config = get_model_config()
    if train_args.enable_ema or getattr(model_config, "peft_config", None) is not None:
        raise NotImplementedError("This MCore runtime adapter does not yet support EMA or PEFT")
    if train_args.enable_chunkpipe:
        raise NotImplementedError("This MCore runtime adapter does not yet support ChunkPipe")

    callbacks = CallbackManager()
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

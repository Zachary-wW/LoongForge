from types import SimpleNamespace

from loongforge.train.mcore_runtime import (
    _sync_chunkpipe_model_config,
    _validate_chunkpipe_entrypoint,
    _is_llava_ov_config,
    _is_vlm_config,
    _requires_legacy_scalar_grad_norm,
    _requires_variable_seq_lengths,
    _restore_variable_seq_lengths,
    _select_sft_clip_root,
    normalize_training_args,
)
from loongforge.train.validators import (
    _drop_custom_pipeline_recompute_layers,
    _prepare_sft_chunkpipe_validation,
)


def test_sft_training_phase_enables_mcore_sft_semantics():
    args = normalize_training_args(
        SimpleNamespace(
            training_phase="sft",
            sft=False,
            enable_chunkpipe=False,
            sft_chunkpipe_mode=True,
            sft_loss_log_mode="token-weighted",
            variable_seq_lengths=False,
            use_fixed_seq_lengths=False,
            apply_rope_fusion=True,
            tp_comm_overlap=False,
            num_experts=None,
            moe_token_dispatcher_type="alltoall",
        )
    )

    assert args.sft is True
    assert args.sft_chunkpipe_mode is False
    assert args.sft_loss_log_mode == "microbatch"
    assert args.variable_seq_lengths is True
    assert args.apply_rope_fusion is True


def test_pretrain_training_phase_does_not_enable_sft():
    args = normalize_training_args(SimpleNamespace(training_phase="pretrain", sft=False))

    assert args.sft is False


def test_sft_chunkpipe_normalization_keeps_chunkpipe_mode():
    args = normalize_training_args(
        SimpleNamespace(
            training_phase="sft",
            sft=False,
            enable_chunkpipe=True,
            sft_chunkpipe_mode=False,
            sft_loss_log_mode="token-weighted",
            variable_seq_lengths=False,
            use_fixed_seq_lengths=False,
            apply_rope_fusion=True,
            tp_comm_overlap=False,
            num_experts=None,
            moe_token_dispatcher_type="alltoall",
        )
    )

    assert args.sft is True
    assert args.sft_chunkpipe_mode is True


def test_sft_chunkpipe_is_enabled_before_mcore_validation():
    args = SimpleNamespace(
        training_phase="sft",
        enable_chunkpipe=True,
        sft=False,
        sft_chunkpipe_mode=False,
    )

    _prepare_sft_chunkpipe_validation(args)

    assert args.sft is True


def test_chunkpipe_model_config_syncs_both_views():
    foundation = SimpleNamespace(sft_chunkpipe_mode=False)
    model_config = SimpleNamespace(sft_chunkpipe_mode=False, foundation=foundation)

    _sync_chunkpipe_model_config(model_config, True)
    assert model_config.sft_chunkpipe_mode is True
    assert foundation.sft_chunkpipe_mode is True

    _sync_chunkpipe_model_config(model_config, False)
    assert model_config.sft_chunkpipe_mode is False
    assert foundation.sft_chunkpipe_mode is False


def test_chunkpipe_entrypoint_allows_llm_sft_only():
    args = SimpleNamespace(
        training_phase="sft",
        enable_chunkpipe=True,
        rerun_mode="disabled",
        eval_iters=0,
        calculate_per_token_loss=False,
        tensor_model_parallel_size=1,
        num_virtual_stages_per_pipeline_rank=1,
    )
    model_config = SimpleNamespace()

    _validate_chunkpipe_entrypoint(args, model_config, allow_chunkpipe=True)

    for overrides, config in (
        ({"training_phase": "pretrain"}, SimpleNamespace()),
        ({}, SimpleNamespace(image_encoder=object())),
        ({"rerun_mode": "validate_results"}, SimpleNamespace()),
        ({"eval_iters": 1}, SimpleNamespace()),
        ({"tensor_model_parallel_size": 2}, SimpleNamespace()),
    ):
        candidate = SimpleNamespace(**{**vars(args), **overrides})
        try:
            _validate_chunkpipe_entrypoint(
                candidate, config, allow_chunkpipe=True
            )
        except NotImplementedError:
            pass
        else:
            raise AssertionError(f"expected chunkpipe rejection for {overrides}")

    try:
        _validate_chunkpipe_entrypoint(
            args, model_config, allow_chunkpipe=False
        )
    except NotImplementedError:
        pass
    else:
        raise AssertionError("expected allow_chunkpipe=False to reject")


def test_fixed_sft_lengths_are_preserved():
    args = normalize_training_args(
        SimpleNamespace(
            training_phase="sft",
            sft=False,
            enable_chunkpipe=False,
            sft_chunkpipe_mode=True,
            sft_loss_log_mode="token-weighted",
            variable_seq_lengths=True,
            use_fixed_seq_lengths=True,
            apply_rope_fusion=True,
            tp_comm_overlap=False,
            num_experts=None,
            moe_token_dispatcher_type="alltoall",
        )
    )

    assert args.variable_seq_lengths is False
    assert args.apply_rope_fusion is True


def test_custom_pipeline_recompute_is_not_inherited_by_vlm_submodels():
    args = SimpleNamespace(custom_pipeline_recompute_layers="12,16")

    _drop_custom_pipeline_recompute_layers("image_encoder", args)

    assert args.custom_pipeline_recompute_layers is None


def test_clip_root_matches_legacy_trainer():
    sft_args = SimpleNamespace(training_phase="sft")
    assert _requires_legacy_scalar_grad_norm(sft_args, SimpleNamespace()) is True
    assert (
        _requires_legacy_scalar_grad_norm(
            sft_args, SimpleNamespace(image_encoder=object())
        )
        is True
    )
    assert (
        _requires_legacy_scalar_grad_norm(
            sft_args, SimpleNamespace(model_backend="mimo")
        )
        is True
    )
    assert (
        _requires_legacy_scalar_grad_norm(
            SimpleNamespace(training_phase="pretrain"), SimpleNamespace()
        )
        is False
    )
    assert (
        _requires_legacy_scalar_grad_norm(
            SimpleNamespace(training_phase="pretrain"),
            SimpleNamespace(model_type="llava_ov_1_5"),
        )
        is True
    )
    assert (
        _requires_legacy_scalar_grad_norm(
            SimpleNamespace(training_phase="pretrain"),
            SimpleNamespace(model_backend="mimo"),
        )
        is False
    )
    assert (
        _requires_legacy_scalar_grad_norm(
            SimpleNamespace(training_phase="pretrain"),
            SimpleNamespace(image_encoder=SimpleNamespace(model_type="qwen2_5_vl")),
        )
        is True
    )
    assert (
        _requires_legacy_scalar_grad_norm(
            SimpleNamespace(training_phase="pretrain"),
            SimpleNamespace(image_encoder=SimpleNamespace(model_type="rice_vit")),
        )
        is True
    )


def test_sft_clip_root_follows_packed_status(monkeypatch):
    calls = []
    monkeypatch.setattr(
        "megatron.core.optimizer.clip_grads.set_legacy_scalar_grad_norm",
        lambda enabled: calls.append(enabled),
    )
    monkeypatch.setattr(
        "loongforge.train.mcore_runtime.get_model_config",
        lambda: SimpleNamespace(image_encoder=object()),
    )

    _select_sft_clip_root(SimpleNamespace(training_phase="sft"), None)
    _select_sft_clip_root(SimpleNamespace(training_phase="sft"), object())
    _select_sft_clip_root(SimpleNamespace(training_phase="pretrain"), None)

    assert calls == [True, False]


def test_vlm_pretrain_uses_variable_seq_lengths():
    assert _is_vlm_config(SimpleNamespace(image_encoder=object())) is True
    assert _is_vlm_config(SimpleNamespace(model_backend="mimo")) is True
    assert _is_vlm_config(SimpleNamespace()) is False
    assert (
        _requires_variable_seq_lengths(
            SimpleNamespace(
                training_phase="pretrain",
                use_fixed_seq_lengths=False,
                tp_comm_overlap=False,
                num_experts=None,
                moe_token_dispatcher_type="alltoall",
            ),
            SimpleNamespace(model_type="llava_ov_1_5"),
        )
        is True
    )
    assert (
        _requires_variable_seq_lengths(
            SimpleNamespace(training_phase="pretrain"),
            SimpleNamespace(),
        )
        is False
    )
    assert (
        _requires_variable_seq_lengths(
            SimpleNamespace(training_phase="pretrain"),
            SimpleNamespace(model_backend="mimo"),
        )
        is False
    )
    assert (
        _requires_variable_seq_lengths(
            SimpleNamespace(training_phase="pretrain"),
            SimpleNamespace(model_backend="omni"),
        )
        is False
    )
    assert (
        _requires_variable_seq_lengths(
            SimpleNamespace(training_phase="pretrain"),
            SimpleNamespace(image_encoder=SimpleNamespace(model_type="qwen2_5_vl")),
        )
        is True
    )
    assert _is_llava_ov_config(SimpleNamespace(model_type="llava_ov_1_5")) is True
    assert (
        _is_llava_ov_config(
            SimpleNamespace(image_encoder=SimpleNamespace(model_type="rice_vit"))
        )
        is True
    )
    assert _is_llava_ov_config(SimpleNamespace(model_backend="mimo")) is False


def test_restore_variable_seq_lengths_updates_global_and_subconfigs(monkeypatch):
    runtime_args = SimpleNamespace(variable_seq_lengths=False)
    foundation = SimpleNamespace(variable_seq_lengths=False)
    image_encoder = SimpleNamespace(variable_seq_lengths=False)
    model_config = SimpleNamespace(
        variable_seq_lengths=False,
        foundation=foundation,
        image_encoder=image_encoder,
    )
    monkeypatch.setattr(
        "loongforge.train.mcore_runtime.get_args", lambda: runtime_args
    )

    _restore_variable_seq_lengths(model_config)

    assert runtime_args.variable_seq_lengths is True
    assert model_config.variable_seq_lengths is True
    assert foundation.variable_seq_lengths is True
    assert image_encoder.variable_seq_lengths is True

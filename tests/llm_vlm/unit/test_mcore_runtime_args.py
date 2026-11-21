from types import SimpleNamespace

from loongforge.train.mcore_runtime import (
    _is_vlm_config,
    _requires_legacy_scalar_grad_norm,
    _requires_variable_seq_lengths,
    normalize_training_args,
)
from loongforge.train.validators import _drop_custom_pipeline_recompute_layers


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
        is False
    )
    assert (
        _requires_legacy_scalar_grad_norm(
            sft_args, SimpleNamespace(model_backend="mimo")
        )
        is False
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
            SimpleNamespace(image_encoder=object()),
        )
        is True
    )
    assert (
        _requires_legacy_scalar_grad_norm(
            SimpleNamespace(training_phase="pretrain"),
            SimpleNamespace(model_backend="mimo"),
        )
        is True
    )


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
            SimpleNamespace(image_encoder=object()),
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

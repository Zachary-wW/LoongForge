from types import SimpleNamespace

from loongforge.train.mcore_runtime import normalize_training_args


def test_sft_training_phase_enables_mcore_sft_semantics():
    args = normalize_training_args(
        SimpleNamespace(training_phase="sft", sft=False, sft_loss_log_mode="token-weighted")
    )

    assert args.sft is True
    assert args.sft_loss_log_mode == "microbatch"


def test_pretrain_training_phase_does_not_enable_sft():
    args = normalize_training_args(SimpleNamespace(training_phase="pretrain", sft=False))

    assert args.sft is False

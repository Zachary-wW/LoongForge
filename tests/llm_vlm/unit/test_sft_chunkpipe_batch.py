from collections import deque
from types import SimpleNamespace

import torch

from loongforge.train.sft import sft_llm
from loongforge.train.sft import utils as sft_utils
from loongforge.data import sft_data_collator


def test_get_batch_consumes_sft_chunkpipe_metadata_and_mtp(monkeypatch):
    args = SimpleNamespace(
        enable_chunkpipe=True,
        sft_chunkpipe_mode=True,
        chunksize=4,
        mtp_num_layers=1,
    )
    config = SimpleNamespace(
        sft_chunkpipe_mode=True,
        chunkpipe_chunk_idx_in_group=0,
    )
    batch = {
        "tokens": torch.zeros(2, 4, dtype=torch.long),
        "labels": torch.zeros(2, 4, dtype=torch.long),
        "loss_mask": torch.ones(2, 4),
        "position_ids": torch.arange(4).expand(2, 4),
        "attention_mask": None,
        "packed_seq_params": None,
        "chunk_group_size": torch.tensor([4]),
        "composite_component_sizes": [2, 2],
        "group_total_tokens": torch.tensor([8]),
        "step_num_groups": torch.tensor([3]),
        "mtp_tokens": torch.zeros(2, 5, dtype=torch.long),
        "mtp_position_ids": torch.arange(5).expand(2, 5),
        "mtp_labels": torch.zeros(2, 5, dtype=torch.long),
        "mtp_loss_mask": torch.ones(2, 5),
    }

    monkeypatch.setattr(sft_llm, "get_batch_on_this_tp_rank", lambda _: batch)
    monkeypatch.setattr(sft_llm, "get_batch_on_this_cp_rank", lambda value: value)
    monkeypatch.setattr(sft_llm, "get_args", lambda: args)
    monkeypatch.setattr(sft_llm, "get_model_config", lambda: config)

    output = sft_llm.get_batch(None)

    assert args.chunkpipe_current_chunk_group_size == 4
    assert config.chunkpipe_current_group_size == 4
    assert config.chunkpipe_component_sizes == [2, 2]
    assert config.chunkpipe_composite_group_size == 4
    assert args.chunkpipe_group_total_tokens == 8
    assert args.chunkpipe_step_num_groups == 3
    assert config.chunkpipe_step_num_groups == 3
    assert output[0].shape == (2, 4)
    assert output[6]["tokens"].shape == (2, 5)
    assert output[6]["position_ids"].shape == (2, 5)


def test_bind_chunkpipe_queue_iter_binds_queues_before_each_batch(monkeypatch):
    args = SimpleNamespace()
    step_g_queue = deque([3])
    composite_queue = deque([[2, 2]])
    monkeypatch.setattr(sft_utils, "get_args", lambda: args)

    bound = sft_utils._bind_chunkpipe_queue_iter(
        iter([{"chunk": 0}, {"chunk": 1}]),
        step_g_queue,
        composite_queue,
    )

    assert "chunkpipe_step_g_queue" not in vars(args)
    assert next(bound) == {"chunk": 0}
    assert args.chunkpipe_step_g_queue is step_g_queue
    assert args.chunkpipe_composite_queue is composite_queue
    assert next(bound) == {"chunk": 1}


def test_build_sft_data_collator_sets_chunkpipe_contract(monkeypatch):
    class FakeAutoTokenizer:
        def hf_tokenizer(self):
            return object()

    class FakeCollator:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    args = SimpleNamespace(
        tensor_model_parallel_size=2,
        sequence_parallel=True,
        context_parallel_size=1,
        fp8=False,
        enable_chunkpipe=True,
        sft_chunkpipe_mode=True,
        chunksize=4,
        seq_length=16,
        variable_seq_lengths=False,
        mtp_num_layers=1,
        tokenizer_type="hf",
    )
    monkeypatch.setattr(sft_utils, "get_args", lambda: args)
    monkeypatch.setattr(sft_utils, "get_tokenizer", FakeAutoTokenizer)
    monkeypatch.setattr(sft_utils, "AutoTokenizerFromHF", FakeAutoTokenizer)

    collator = sft_utils.build_sft_data_collator(FakeCollator)

    assert collator.kwargs["pad_to_multiple_of"] == 4
    assert collator.kwargs["max_length"] == 4
    assert collator.kwargs["chunkpipe_base_length"] == 4
    assert collator.kwargs["chunkpipe_mtp_num_layers"] == 1


def test_sft_collator_splits_and_rejoins_bridge_window(monkeypatch):
    class FakeCollator:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, features, return_tensors=None):
            return {
                key: torch.tensor([feature[key] for feature in features])
                for key in ("input_ids", "labels", "attention_mask", "loss_mask")
            }

    monkeypatch.setattr(sft_data_collator, "DataCollatorForSeq2Seq", FakeCollator)
    collator = sft_data_collator.DataCollatorForSupervisedDataset(
        tokenizer=object(),
        chunkpipe_base_length=4,
        chunkpipe_mtp_num_layers=1,
        padding=None,
    )
    features = [
        {
            "input_ids": [10, 11, 12, 13, 90],
            "labels": [11, 12, 13, 14, 91],
            "attention_mask": [1, 1, 1, 1, 1],
            "loss_mask": [1, 1, 1, 1, 1],
            "chunk_group_size": 4,
            "group_total_tokens": 8,
        },
        {
            "input_ids": [20, 21, 22, 23, 92],
            "labels": [21, 22, 23, 24, 93],
            "attention_mask": [1, 1, 1, 1, 1],
            "loss_mask": [1, 1, 1, 1, 1],
            "chunk_group_size": 4,
            "group_total_tokens": 8,
        },
    ]

    result = collator(features)

    assert result["input_ids"].tolist() == [[10, 11, 12, 13, 90], [20, 21, 22, 23, 92]]
    assert result["labels"].tolist() == [[11, 12, 13, 14, 91], [21, 22, 23, 24, 93]]
    assert result["attention_mask"].shape == (2, 5)
    assert result["loss_mask"].shape == (2, 5)
    assert result["chunk_group_size"].tolist() == [4, 4]
    assert result["group_total_tokens"].tolist() == [8, 8]

from types import SimpleNamespace

import torch

from loongforge.train.sft import sft_llm


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

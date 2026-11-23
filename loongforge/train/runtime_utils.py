# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0

"""Small runtime diagnostics that do not depend on a legacy training loop."""

import os

import torch
from megatron.core import mpu
from megatron.core.rerun_state_machine import ChunkDataIterator


_PRINTED_MODEL_INPUT_EXAMPLE = False

_SAMPLE_DUMP_ANSI_COLOR = {
    "T": "\033[92m",
    "C": "\033[93m",
    "P": "\033[90m",
}
_SAMPLE_DUMP_ANSI_RESET = "\033[0m"


def dump_model_input_example_once(
    tokens, labels, attn_mask, cu_lengths=None, packed_seq_params=None
):
    """Dump the exact tensors flowing into ``model(...)`` once per rank-0 process."""
    global _PRINTED_MODEL_INPUT_EXAMPLE
    if _PRINTED_MODEL_INPUT_EXAMPLE:
        return

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        if torch.distributed.get_rank() != 0:
            return
    else:
        if int(os.environ.get("RANK", "0")) != 0:
            return

    try:
        if mpu.get_tensor_model_parallel_rank() != 0:
            return
        if mpu.get_context_parallel_rank() != 0:
            return
    except Exception:
        pass

    _PRINTED_MODEL_INPUT_EXAMPLE = True

    from loongforge.utils.constants import IGNORE_INDEX
    from loongforge.utils.global_vars import get_tokenizer as get_loongforge_tokenizer

    tokenizer = get_loongforge_tokenizer()

    def to_cpu_list(x):
        if torch.is_tensor(x):
            return x.detach().cpu().tolist()
        if x is None:
            return None
        return list(x)

    def decode(ids):
        if not ids:
            return ""
        try:
            return tokenizer.detokenize(ids, skip_special_tokens=False)
        except TypeError:
            return tokenizer.detokenize(ids)

    tok = tokens[0] if tokens.dim() == 2 else tokens
    lab = labels[0] if labels.dim() == 2 else labels

    cu = cu_lengths
    if torch.is_tensor(cu):
        cu = cu[0] if cu.dim() == 2 else cu
    cu_list = to_cpu_list(cu) or []

    tok_list = to_cpu_list(tok)
    lab_list = to_cpu_list(lab)

    am = attn_mask
    am_list = None
    if torch.is_tensor(am):
        if am.dim() == 4:
            am_view = am[0, 0].diagonal()
        elif am.dim() == 2:
            am_view = am[0]
        elif am.dim() == 1:
            am_view = am
        else:
            am_view = None
        if am_view is not None:
            am_list = am_view.detach().cpu().to(torch.bool).tolist()

    seq_len = len(tok_list)
    trainable_total = (
        sum(1 for value in lab_list[:-1] if value != IGNORE_INDEX)
        if lab_list
        else 0
    )

    def role_at(index):
        if index > 0 and lab_list[index - 1] != IGNORE_INDEX:
            return "T"
        if am_list is not None and index < len(am_list) and am_list[index]:
            return "P"
        return "C"

    header = [
        "===== model-input example (forward_step entry, same tensor as model(...)) =====",
        f"tokens.shape={tuple(tokens.shape)} labels.shape={tuple(labels.shape)} "
        f"attn_mask.shape={tuple(attn_mask.shape) if torch.is_tensor(attn_mask) else None}",
        f"local_seq_len={seq_len} trainable_tokens={trainable_total}",
        f"cu_lengths={cu_list}",
        f"packed_seq_params={packed_seq_params}",
        f"tp_rank={mpu.get_tensor_model_parallel_rank()} "
        f"cp_rank={mpu.get_context_parallel_rank()}/{mpu.get_context_parallel_world_size()} "
        f"pp_rank={mpu.get_pipeline_model_parallel_rank()}",
        f"[mask_legend] {_SAMPLE_DUMP_ANSI_COLOR['T']}T=trainable target{_SAMPLE_DUMP_ANSI_RESET} "
        f"{_SAMPLE_DUMP_ANSI_COLOR['C']}C=context attended{_SAMPLE_DUMP_ANSI_RESET} "
        f"{_SAMPLE_DUMP_ANSI_COLOR['P']}P=padded/masked{_SAMPLE_DUMP_ANSI_RESET}",
    ]

    trainable_ids = [value for value in lab_list if value != IGNORE_INDEX]

    colored_segments = []
    if seq_len:
        run_start = 0
        run_role = role_at(0)
        for index in range(1, seq_len):
            role = role_at(index)
            if role != run_role:
                colored_segments.append(
                    f"{_SAMPLE_DUMP_ANSI_COLOR[run_role]}"
                    f"{decode(tok_list[run_start:index])}"
                    f"{_SAMPLE_DUMP_ANSI_RESET}"
                )
                run_start = index
                run_role = role
        colored_segments.append(
            f"{_SAMPLE_DUMP_ANSI_COLOR[run_role]}"
            f"{decode(tok_list[run_start:seq_len])}"
            f"{_SAMPLE_DUMP_ANSI_RESET}"
        )

    body = [
        "[input_ids]",
        str(tok_list),
        "[decoded_input | colored by role]",
        "".join(colored_segments),
        "[decoded_trainable_labels]",
        decode(trainable_ids),
    ]

    print(
        "\n".join(header + body + ["===== end model-input example ====="]),
        flush=True,
    )


def get_next_batch_on_this_tp_rank(data_iterator):
    """Fetch and broadcast the next ChunkPipe chunk for MTP on the last stage."""
    from loongforge.utils import get_args

    args = get_args()

    def _broadcast(item):
        if item is not None:
            torch.distributed.broadcast(
                item,
                mpu.get_tensor_model_parallel_src_rank(),
                group=mpu.get_tensor_model_parallel_group(),
            )

    if mpu.get_tensor_model_parallel_rank() == 0:
        is_none = not isinstance(data_iterator, ChunkDataIterator)
        if not is_none:
            data = data_iterator.get_next_chunk()
            is_none = data is None

        item_exist = torch.tensor(1 if is_none else 0, dtype=torch.int32, device="cuda")
        _broadcast(item_exist)
        if is_none:
            return None

        if args.enable_chunkpipe and getattr(args, "variable_seq_lengths", False):
            seq_len_t = torch.tensor(
                [data["tokens"].size(1)], dtype=torch.int64, device="cuda"
            )
            _broadcast(seq_len_t)

        fake_mask = torch.tensor(1, dtype=torch.int32, device="cuda")
        batch = {
            "tokens": data["tokens"].cuda(non_blocking=True),
            "labels": data["labels"].cuda(non_blocking=True),
            "loss_mask": data["loss_mask"].cuda(non_blocking=True),
            "attention_mask": fake_mask,
            "position_ids": data["position_ids"].cuda(non_blocking=True),
        }

        if args.pipeline_model_parallel_size == 1:
            for key in ("tokens", "labels", "loss_mask", "attention_mask", "position_ids"):
                _broadcast(batch[key])
        elif mpu.is_pipeline_first_stage():
            for key in ("tokens", "attention_mask", "position_ids"):
                _broadcast(batch[key])
        elif mpu.is_pipeline_last_stage():
            if (getattr(args, "mtp_num_layers", 0) or 0) > 0 and (
                mpu.get_tensor_model_parallel_world_size() > 1
            ):
                _broadcast(batch["tokens"])
                _broadcast(batch["position_ids"])
            for key in ("labels", "loss_mask", "attention_mask"):
                _broadcast(batch[key])

        return batch

    item_exist = torch.tensor(0, dtype=torch.int32, device="cuda")
    _broadcast(item_exist)
    if item_exist.item() == 1:
        return None

    tmp_seq_length = args.seq_length
    if args.enable_chunkpipe:
        tmp_seq_length = args.chunksize
        if getattr(args, "variable_seq_lengths", False):
            seq_len_t = torch.empty((1,), dtype=torch.int64, device="cuda")
            _broadcast(seq_len_t)
            tmp_seq_length = int(seq_len_t.item())

    tokens = torch.empty(
        (args.micro_batch_size, tmp_seq_length),
        dtype=torch.int64,
        device=torch.cuda.current_device(),
    )
    labels = torch.empty(
        (args.micro_batch_size, tmp_seq_length),
        dtype=torch.int64,
        device=torch.cuda.current_device(),
    )
    loss_mask = torch.empty(
        (args.micro_batch_size, tmp_seq_length),
        dtype=torch.float32,
        device=torch.cuda.current_device(),
    )
    attention_mask = torch.tensor(0, dtype=torch.int32, device="cuda")
    position_ids = torch.empty(
        (args.micro_batch_size, tmp_seq_length),
        dtype=torch.int64,
        device=torch.cuda.current_device(),
    )

    if args.pipeline_model_parallel_size == 1:
        for tensor in (tokens, labels, loss_mask, attention_mask, position_ids):
            _broadcast(tensor)
    elif mpu.is_pipeline_first_stage():
        labels = None
        loss_mask = None
        for tensor in (tokens, attention_mask, position_ids):
            _broadcast(tensor)
    elif mpu.is_pipeline_last_stage():
        if (getattr(args, "mtp_num_layers", 0) or 0) > 0 and (
            mpu.get_tensor_model_parallel_world_size() > 1
        ):
            _broadcast(tokens)
            _broadcast(position_ids)
        else:
            tokens = None
            position_ids = None
        for tensor in (labels, loss_mask, attention_mask):
            _broadcast(tensor)

    return {
        "tokens": tokens,
        "labels": labels,
        "loss_mask": loss_mask,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
    }

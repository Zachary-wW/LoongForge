#!/usr/bin/env python3
"""Standalone TRANSPLANT dataloader checksum (Rule 1 input-alignment tool).

Twin of dataloader_checksum_base.py. Build the REAL transplant dataloader via
the project's build_dataloader(), then REBUILD a SEQUENTIAL loader over the same
dataset+collate so batch s == idx {s*bs .. s*bs+bs-1}, matching base. Print the
same per-key sha1 [TRANS_CKSUM] lines and diff against [BASE_CKSUM].

CRITICAL trap this guards against: at world_size=1 many frameworks' build_sampler
returns None and the loader silently defaults to shuffle=True -> random order.
We therefore force a SequentialSampler here so the comparison is deterministic.

REUSE: reference template from the Motus port. Edit the LF repo root, the sys.argv
launch flags to match your trainer's parser, the build_dataloader import, and the
batch attribute/key names printed in [TRANS_CKSUM] to match your project.
"""
import os, sys, hashlib

os.environ.setdefault("PARITY_DATA_SEED", "42")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

# ===== EDIT-ME: transplant repo root =====
LF = "/path/to/TRANSPLANT"
os.environ["LOONGFORGE_PATH"] = LF
sys.path.insert(0, LF)

DS_CFG = os.path.join(LF, "examples/embodied/motus/zero1.json")  # EDIT-ME (only if your parser requires it)
N = int(os.environ.get("PARITY_N", "6"))

# ===== EDIT-ME: the launch flags your trainer parser expects =====
sys.argv = [
    "parity_batch_trans.py",
    "--model-name", "motus",
    "--dataset-format", "lerobot_datasets",
    "--dataset-strategy", "motus",
    "--dataset-path", "/path/to/data",
    "--video-backend", "torchcodec",
    "--num-workers", "0",
    "--trainer-type", "MotusTrainer",
    "--train-iters", "1000000",
    "--per-device-batch-size", "8",
    "--gradient-accumulation-steps", "1",
    "--seed", "42",
    "--output-dir", "/tmp/parity_out",
    "--lr-base", "5.0e-5",
    "--lr-decay-style", "lambda_linear",
    "--optimizer", "AdamW",
    "--distributed-strategy", "ddp",
    "--dtype", "bfloat16",
    # NOTE: this probe only builds the DATALOADER, so the engine choice is
    # irrelevant here — keep whatever the parser will accept with the fewest
    # required extras. Drop `--deepspeed-config` unless your parser mandates it
    # for the strategy you name above.
    "--log-interval", "1",
]

import torch
from loongforge.embodied.train.parser import parse_train_args          # EDIT-ME
from loongforge.embodied.data.dataloader import build_dataloader       # EDIT-ME
from loongforge.embodied.distributed import DistributedContext         # EDIT-ME


def sha(t):
    if t is None:
        return "none"
    if isinstance(t, torch.Tensor):
        b = t.detach().float().cpu().contiguous().numpy().tobytes()
        return f"{tuple(t.shape)}:{hashlib.sha1(b).hexdigest()[:12]}"
    if isinstance(t, (list, tuple)):
        return "[" + ",".join(sha(x) for x in t) + "]"
    if isinstance(t, dict):
        return "{" + ",".join(f"{k}={sha(v)}" for k, v in sorted(t.items())) + "}"
    return type(t).__name__


def main():
    training_args, model_cfg, data_cfg = parse_train_args()
    ctx = DistributedContext()
    ctx.init()
    dl = build_dataloader(model_cfg, data_cfg, training_args, ctx)
    # Force SEQUENTIAL order (see docstring: ws=1 build_sampler->None->shuffle=True).
    from torch.utils.data import DataLoader, SequentialSampler
    ds = dl.dataset
    collate = dl.collate_fn
    seq = DataLoader(
        ds,
        batch_size=training_args.per_device_batch_size,
        sampler=SequentialSampler(ds),
        shuffle=False,
        num_workers=0,
        collate_fn=collate,
        drop_last=True,
    )
    print(f"[TRANS] bs={training_args.per_device_batch_size} len={len(ds)}", flush=True)

    it = iter(seq)
    for s in range(N):
        b = next(it)
        if s == 0 and getattr(b, "vlm_inputs", None) is not None:
            ids = b.vlm_inputs["input_ids"][0].tolist()
            import json
            open("/tmp/trans_ids.json", "w").write(json.dumps(ids))
            print(f"[TRANS_IDS] len={len(ids)} {ids}", flush=True)
        print(
            f"[TRANS_CKSUM] step={s} "
            f"first_frame={sha(getattr(b, 'first_frame', None))} "
            f"video_frames={sha(getattr(b, 'video_frames', None))} "
            f"state={sha(getattr(b, 'initial_state', None))} "
            f"actions={sha(getattr(b, 'action_sequence', None))} "
            f"lang={sha(getattr(b, 'language_embedding', None))} "
            f"vlm={sha(getattr(b, 'vlm_inputs', None))}",
            flush=True,
        )


if __name__ == "__main__":
    main()

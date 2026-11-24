#!/usr/bin/env python
"""Dump the REAL project sampler/batch_sampler index order (Rule 1 tool, CPU-only).

Where sampler_order_probe.py compares two *synthetic* samplers over range(N),
this builds the transplant's ACTUAL registered sampler on top of the REAL
dataset (with a faked distributed ctx: ws=8, rank=0) and iterates the
BatchSampler to print the exact per-step index lists rank0's dataloader would
feed to __getitem__ -- with NO video decode / preprocessor / model, so it's
fast and isolates order from content.

Use it to (a) confirm the parity-mode no-shuffle path is actually taken
(sampler is not None), and (b) compare the printed batches against base's real
order. A `sampler is None` warning means the loader would silently shuffle and
parity is broken.

REUSE: reference from the Motus port. Edit the repo root, the sys.argv launch
flags for your parser, and the three project imports (parse_train_args,
_build_dataset, build_sampler). The faked-ctx + BatchSampler-iteration pattern
is general.
"""
import os
import sys
import itertools

os.environ["PARITY_DATA_SEED"] = "42"
os.environ["PARITY_FM_SEED"] = "42"

# ===== EDIT-ME: transplant repo root =====
LOONGFORGE = "/path/to/TRANSPLANT"
sys.path.insert(0, LOONGFORGE)

# ===== EDIT-ME: launch flags your parser expects =====
sys.argv = [
    "train.py",
    "--model-name", "motus",
    "--dataset-format", "lerobot_datasets",
    "--dataset-strategy", "motus",
    "--dataset-path", "/path/to/data",
    "--video-backend", "torchcodec",
    "--num-workers", "0",
    "--dataloader-seed-workers",
    "--trainer-type", "MotusTrainer",
    "--train-iters", "1000",
    "--per-device-batch-size", "8",
    "--gradient-accumulation-steps", "1",
    "--seed", "42",
    "--output-dir", "/tmp/idx_probe_out",
    "--optimizer", "AdamW",
    "--distributed-strategy", "ddp",
    "--dtype", "bfloat16",
]

from torch.utils.data import BatchSampler, IterableDataset
from loongforge.embodied.train.parser import parse_train_args               # EDIT-ME
from loongforge.embodied.data.dataloader import _build_dataset              # EDIT-ME
from loongforge.embodied.data.datasets.sampler_builder import build_sampler # EDIT-ME
from loongforge.embodied.distributed import DistributedContext              # EDIT-ME

training_args, model_cfg, data_cfg = parse_train_args()
print("model_type =", model_cfg.model_type)
print("batch_drop_last =", training_args.batch_drop_last,
      " distributed_sampler_mode =", training_args.distributed_sampler_mode,
      " seed_workers =", training_args.dataloader_seed_workers)

ds = _build_dataset(model_cfg, data_cfg, training_args, training_args.dataset_format)
print("dataset:", type(ds).__name__, "len =", len(ds),
      "isIterable =", isinstance(ds, IterableDataset))

# Fake an 8-GPU rank0 ctx WITHOUT initializing real distributed comm.
ctx = DistributedContext()
ctx.rank = 0
ctx.world_size = 8
ctx._initialized = True

_parity_shuffle = os.environ.get("PARITY_DATA_SEED") is None
sampler = build_sampler(
    model_cfg.model_type or "dummy",
    dataset=ds,
    training_args=training_args,
    ctx=ctx,
    batch_size=training_args.per_device_batch_size,
    seed=training_args.seed,
    shuffle=_parity_shuffle,
)
print("sampler:", type(sampler).__name__ if sampler is not None else None)

if sampler is None:
    print("!! sampler is None -> DataLoader would shuffle (PARITY broken)")
else:
    bs = BatchSampler(sampler, batch_size=training_args.per_device_batch_size,
                      drop_last=training_args.batch_drop_last)
    print("REAL rank0 batch_sampler first 12 batches:")
    for i, b in enumerate(itertools.islice(bs, 12)):
        print(f"  B{i:2d}: {list(b)}")

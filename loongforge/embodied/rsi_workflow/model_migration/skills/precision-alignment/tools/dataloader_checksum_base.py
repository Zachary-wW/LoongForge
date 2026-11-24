#!/usr/bin/env python3
"""Standalone BASE dataloader checksum (Rule 1 input-alignment tool).

Bypass the model entirely: iterate the REAL base dataloader single-process,
no-shuffle, num_workers=0, with the parity data seed on, and print a per-key
sha1 for the first N batches. Run the TRANSPLANT twin (dataloader_checksum_trans
.py) the same way and diff the [*_CKSUM] lines step-by-step. Identical hashes
prove the two sides feed the model the same input tensors -> divergence, if any,
is forward/backward, not data.

Why this design:
  - world_size=1 + SequentialSampler + no shuffle => batch s == dataset idx
    {s*bs .. s*bs+bs-1}, deterministic and directly comparable across projects.
  - PARITY_DATA_SEED makes every in-__getitem__ random draw a pure function of
    idx, so content is reproducible.
  - sha() recurses into dict/list and stringifies (shape:hash) so nested batch
    structs (e.g. vlm_inputs) are compared field-by-field.

REUSE: reference template from the Motus port. Edit BASE path, the dataset
constructor import (create_dataset/collate_fn), the config path, and the batch
key names printed in the [BASE_CKSUM] line to match your project.
"""
import os, sys, hashlib, argparse

os.environ.setdefault("PARITY_DATA_SEED", "42")
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

# ===== EDIT-ME: base repo root + dataset API =====
BASE = "/path/to/BASE"
sys.path.insert(0, BASE)

import torch
from omegaconf import OmegaConf
from data.dataset import create_dataset, collate_fn   # EDIT-ME: base's dataset API
from torch.utils.data import DataLoader


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=os.path.join(BASE, "configs/lerobot.yaml"))
    ap.add_argument("--n", type=int, default=6)
    args = ap.parse_args()

    cfg = OmegaConf.load(args.config)
    ds = create_dataset(cfg, val=False)
    print(f"[BASE] dataset len={len(ds)} bs={cfg.training.batch_size}", flush=True)

    dl = DataLoader(
        ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        sampler=None,
        num_workers=0,
        pin_memory=False,
        collate_fn=collate_fn,
        drop_last=True,
    )

    it = iter(dl)
    for s in range(args.n):
        b = next(it)
        # EDIT-ME: dump the tokenizer ids on step 0 if you have a VLM/text path
        if s == 0 and b.get("vlm_inputs") is not None:
            ids = b["vlm_inputs"]["input_ids"][0].tolist()
            import json
            open("/tmp/base_ids.json", "w").write(json.dumps(ids))
            print(f"[BASE_IDS] len={len(ids)} {ids}", flush=True)
        # EDIT-ME: list your batch keys
        print(
            f"[BASE_CKSUM] step={s} "
            f"first_frame={sha(b.get('first_frame'))} "
            f"video_frames={sha(b.get('video_frames'))} "
            f"state={sha(b.get('initial_state'))} "
            f"actions={sha(b.get('action_sequence'))} "
            f"lang={sha(b.get('language_embedding'))} "
            f"vlm={sha(b.get('vlm_inputs'))}",
            flush=True,
        )


if __name__ == "__main__":
    main()

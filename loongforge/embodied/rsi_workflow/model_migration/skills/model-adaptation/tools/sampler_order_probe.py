#!/usr/bin/env python3
"""Sampler ORDER probe: base vs transplant DP sampler, rank0 (Rule 1 tool).

Isolates *sampler order* from dataset content. Builds each side's actual sampler
over range(N) (a plain index list, no dataset/collate/decode) and prints the
first K batches rank0 sees. This answers "does base step g feed the same global
indices as transplant step g?" without touching the model or the data.

Sweeps a few N values so drop_last / padding tail effects surface if any. A
`DIFF` on any step means the two samplers disagree on order -> data will diverge
regardless of how perfect the forward is.

REUSE: reference from the Motus port (base = torch DistributedSampler,
transplant = torchdata StatefulDistributedSampler). Swap the two sampler
constructors below for whatever your two projects use, and match shuffle /
drop_last / seed to the real launch config.
"""
import itertools
from torch.utils.data import DistributedSampler
from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

WS = 8
RANK = 0
BS = 8
K = 6  # batches to show


def batches(sampler, k, bs):
    it = iter(sampler)
    out = []
    for _ in range(k):
        b = list(itertools.islice(it, bs))
        if len(b) < bs:
            break
        out.append(b)
    return out


for N in (4000, 4003, 4007):  # sweep so drop_last tail effects show if any
    ds = list(range(N))
    # ===== EDIT-ME: BASE sampler =====
    base = DistributedSampler(ds, num_replicas=WS, rank=RANK, shuffle=False)
    # ===== EDIT-ME: TRANSPLANT sampler =====
    trans = StatefulDistributedSampler(
        ds, num_replicas=WS, rank=RANK, shuffle=False, seed=42, drop_last=True
    )
    bb = batches(base, K, BS)
    tb = batches(trans, K, BS)
    print(f"\n===== N={N}  base.num_samples={base.num_samples} "
          f"trans.num_samples={trans.num_samples} =====")
    for s in range(min(len(bb), len(tb))):
        same = "OK " if bb[s] == tb[s] else "DIFF"
        print(f"  step{s} {same} base={bb[s]}  trans={tb[s]}")

#!/usr/bin/env python
"""Double-shard detector: DistributedSampler + accelerate.prepare (Rule 1 tool).

THE classic distributed-sharding trap. If base builds a
DataLoader(sampler=DistributedSampler(rank,ws)) and then passes it THROUGH
accelerate.prepare()/prepare_data_loader(), accelerate layers a BatchSamplerShard
on top -> the loader is sharded TWICE. rank0 then consumes only every ws-th
batch: [0..8), [ws*bs..ws*bs+bs), ... i.e. base step g == transplant step g*ws,
and ~(ws-1)/ws of the data is silently dropped.

This script prints the raw (no-accelerate) first batches vs the
accelerate-wrapped first batches for rank0 so you can see the round-robin skip
directly. If the transplant does NOT double-shard (raw StatefulDataLoader), its
per-step indices will be the CONTIGUOUS stride the raw loader shows -> base and
transplant are off by a factor of ws unless you account for it.

REUSE: reference from the Motus port. Edit N/BS/WS; the pattern is framework-
agnostic (any accelerate.prepare on an already-DistributedSampler'd loader).
"""
import itertools
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from accelerate.data_loader import prepare_data_loader

N, BS, WS = 85000, 8, 8


class Dummy(Dataset):
    def __len__(self):
        return N

    def __getitem__(self, i):
        return int(i)


ds = Dummy()
samp = DistributedSampler(ds, num_replicas=WS, rank=0, shuffle=False)
dl = DataLoader(ds, batch_size=BS, sampler=samp, drop_last=True, collate_fn=list)

print("raw DataLoader (no accelerate) first 3 batches:")
for i, b in enumerate(itertools.islice(dl, 3)):
    print(f"  B{i}: {b}")

prep = prepare_data_loader(dl, num_processes=WS, process_index=0,
                           put_on_device=False, split_batches=False)
print("\naccelerate.prepare_data_loader(ws=8, rank=0) first 3 batches:")
for i, b in enumerate(itertools.islice(prep, 3)):
    print(f"  B{i}: {b}")

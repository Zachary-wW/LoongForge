# Rule 1 — Dataset & Sampling Alignment

**Goal:** the transplant dataloader emits, for a given step, the *same samples*
(content) in the *same order* as base. This is the #1 source of "loss diverges
at step 2" false alarms: if the forward/backward is perfect but the data differs,
loss still diverges.

## What to align (checklist)

1. **Episode/frame index bounds.** How does each `__getitem__` map a flat index →
   (episode, frame)? Watch for:
   - version skew in the dataset lib (keys renamed, e.g. v3.0 uses
     `dataset_from_index`/`dataset_to_index`).
   - absolute vs relative flat offsets (base manual-decodes rows directly →
     needs *relative* cumulative offsets over the *sorted* selected episodes;
     a native `__getitem__` that reads a stored absolute `item["index"]` does not).
   - real per-episode frame counts vs using an episode-ID as a frame index
     (a classic base bug — decide: replicate or fix).
2. **Media/video decode.** Confirm the frame actually decoded is the intended
   one. Concatenated-mp4 datasets need a per-stream `from_timestamp` shift;
   omitting it silently reads the wrong episode's pixels (verify empirically:
   `frame(ep5,0)` must differ from `frame(ep0,0)`).
3. **Transforms / normalization.** resize/pad, multi-cam stitch, action
   geometry, normalization stats. These are usually verbatim ports — confirm
   the stats file is byte-identical and the transform order matches.
4. **Anchor / condition-frame sampling RNG.** The exact RNG object, seed
   formula, and draw order. base `random.Random(seed+idx)` picking an episode
   then a condition frame must match the transplant sampler draw-for-draw.
5. **Text/prompt sources.** A subtle one: which fields feed the tokenizer/VLM.
   base may leave the instruction empty (image-only) while transplant fills the
   task string → different token sequences → perturbs the forward. Decide the
   alignment direction with the user.

## Tools / techniques

- **Standalone dataloader diff** (best tool). Bypass the model entirely; build
  each side's dataloader with `world_size=1, num_workers=0`, force a
  `SequentialSampler` (some frameworks default `shuffle=True` when
  `world_size==1`!), iterate idx 0..N, and `sha1` every batch key. Compare
  key-by-key across the first ~6 steps. This isolates data from all model math.
- **Anchor/index probes.** Temporarily dump `raw idx + selected global
  condition-frame index` from both `__getitem__`s; diff per step. Remove after.
- **Distributed shard math.** Verify `DistributedSampler` slicing is identical
  (strided `idx[rank::ws]`). BEWARE wrappers that *double-shard* (e.g. HF
  accelerate's `prepare(dataloader)` layers a `BatchSamplerShard` on top of an
  existing `DistributedSampler` → each rank consumes only every `ws`-th batch,
  i.e. base step g ≡ transplant step g·ws, and ~(ws-1)/ws of data is dropped).
- **drop_last geometry.** Match sampler `drop_last` (base `DistributedSampler`
  default `drop_last=False` pads the tail → each rank sees `ceil(N/ws)`), even
  if the DataLoader still drops the last partial batch.
- **Worker seeding.** Match `worker_init_fn` (e.g. `seed = base_seed + worker_id`)
  so per-worker RNG streams line up.

## Reference tools (`tools/`)

- **`sampler_order_probe.py`** — builds each side's sampler over `range(N)`
  (no dataset/decode) and prints the first K rank0 batches, sweeping a few N so
  drop_last/padding tails surface. Fastest way to prove "base step g feeds the
  same indices as transplant step g". `DIFF` on any step ⇒ order mismatch.
- **`real_sampler_idx_dump.py`** — builds the transplant's *real* registered
  sampler on the *real* dataset with a faked ws=8/rank0 ctx and iterates the
  `BatchSampler` (index lists only). Confirms the no-shuffle parity path is
  taken (`sampler is not None`) and dumps the actual per-step indices.
- **`double_shard_check.py`** — prints raw vs `accelerate.prepare_data_loader`
  batches to expose the double-shard trap (base step g ≡ transplant step g·ws).

All three are project-specific reference templates — edit the sampler
constructors / repo paths / launch flags. Content-level checksums (proving the
*tensors*, not just the indices, match) live in the `precision-alignment`
skill's `tools/` (`dataloader_checksum_*.py`).

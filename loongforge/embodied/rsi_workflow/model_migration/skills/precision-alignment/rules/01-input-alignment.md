# Rule 1 — Training Input Alignment

**Goal:** prove that at every training step k, BASE and TRANSPLANT feed the
model the *identical* input tensors (data batch AND per-step stochastic draws).
Until this is bit-identical, any loss/grad comparison is meaningless — a
perfect forward on different inputs still diverges.

This is the precision-side counterpart to model-adaptation Rule 1: there you
made the dataloader emit the same samples; here you *prove* it, and you also
lock down the per-step RNG that the model itself consumes (noise, timesteps).

## What to align & prove (checklist)

1. **Data anchor as a pure function of index.** Gate a parity mode (e.g.
   `PARITY_DATA_SEED=42`) that (a) forces `shuffle=False` / a
   `SequentialSampler`, and (b) makes every stochastic choice inside
   `__getitem__` (anchor episode, condition frame) a deterministic function of
   the flat idx — `random.Random(seed + idx)` — so rank r step g always sees
   the same content regardless of timing or worker count.
2. **Per-step model-side RNG (flow-matching / diffusion noise).** The model
   draws its own randomness each step: timestep indices and gaussian noise.
   These must be seeded identically and drawn in the *same order*. Gate e.g.
   `PARITY_FM_SEED=42` → `torch.manual_seed(seed + global_step)` immediately
   before the draw block, and match the exact draw sequence:
   `video_timestep = randint(...)` → `video_noise = randn(...)` →
   `action_timestep = randint(...)` → `action_noise = randn(...)`. A reordered
   or merged draw desyncs the whole stream even with the same seed.
3. **Text / prompt source.** Which string feeds the tokenizer/VLM. base may
   pass an empty instruction (image-only) while transplant fills the task text
   → different token ids → different forward. Decide the direction with the
   user and lock it (in Motus: `text_instr=""` to match base's image-only).
4. **Worker seeding.** `worker_init_fn` must produce matching per-worker RNG
   streams (`seed = base_seed + worker_id`), or set `num_workers=0` for the
   parity run to remove the variable entirely.
5. **Distributed shard identity.** Under DP, rank0 must consume the same flat
   indices on both sides (strided `idx[rank::ws]` → rank0 = 0,8,16,…). Beware
   wrappers that double-sward (accelerate `prepare(dataloader)` on top of a
   `DistributedSampler`).

## Tools / techniques

- **Standalone dataloader checksum (the decisive tool).** Bypass the model.
  Build each side's loader with `world_size=1, num_workers=0`, force a
  `SequentialSampler`, iterate idx 0..N, and `sha1` every batch tensor
  key-by-key. Compare across the first ~6 steps. Identical hashes ⇒ inputs are
  proven equal and you may move to Rule 2. This isolates data from all model
  math and from the per-step FM noise.
- **FM/noise draw probe.** With `PARITY_FM_SEED` on, dump the seed used and a
  checksum (`.cpu()`/`.tolist()`) of `video_timestep/video_noise/action_*` on
  both sides for steps 1..6; diff. This proves the *model-consumed* RNG matches
  (the dataloader checksum does not cover it). Remove/gate the probe after.
- **Anchor/index probe.** Temporarily dump `raw idx + selected condition-frame
  global index` from each `__getitem__`; diff per step to prove the anchor RNG
  agrees draw-for-draw.
- **Offline latent cache (when a VAE sits in the input path).** If inputs
  include VAE latents, an offline precompute (batch-preserving, resume-safe,
  bit-identical regardless of shard count) removes VAE nondeterminism from the
  per-step comparison and makes the input tensor a fixed artifact.

## Reference tools (`tools/`)

- **`dataloader_checksum_base.py`** — standalone BASE loader, single-process /
  no-shuffle / `num_workers=0` / `PARITY_DATA_SEED`, prints per-key sha1 for the
  first N batches (`[BASE_CKSUM]`).
- **`dataloader_checksum_trans.py`** — the TRANSPLANT twin. Note it *forces* a
  `SequentialSampler` because many frameworks default `shuffle=True` at
  `world_size==1`. Diff `[TRANS_CKSUM]` vs `[BASE_CKSUM]` step-by-step.

Both are project-specific reference templates (edit the repo paths, dataset API
imports, and batch key names). Sampler-order probes (proving *which indices* map
to each step, independent of content) live in the `model-adaptation` skill's
`tools/` — use those first, then these to confirm content.

## Acceptance

- Standalone dataloader sha1 is identical key-by-key for the first ~6 steps.
- With the parity FM seed on, the per-step noise/timestep checksums match.
- ⇒ Inputs are proven identical. Any remaining divergence is forward or
  backward → hand to Rule 2.

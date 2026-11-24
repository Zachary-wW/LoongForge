# Rule 2 — Forward / Loss Alignment

**Goal:** with inputs proven identical (Rule 1), prove the forward pass is
numerically equivalent by showing the **step-1 loss is bit-exact** (or in-band
where nondeterminism is unavoidable) on every loss channel. Step-1 loss is the
cleanest forward signal: it depends only on the initial weights + the input,
with no accumulated optimizer history yet.

## What to align & prove (checklist)

1. **Loss-logging semantics (fix the yardstick FIRST).** The most common false
   alarm: base logs **rank0-local** loss while transplant logs an **N-rank
   all-reduce mean**. The mean has ~1/√ws the per-step variance, so numbers
   look "different" even when the math is identical. Add a flag
   (`--loss-log-rank 0`) so both log rank0-local *before* you compare anything.
2. **Per-channel comparison.** A multi-modal model has several loss channels
   (e.g. video / action / und). Compare each channel separately — an aggregate
   total can hide a diverging channel behind a matching one. In Motus the
   action channel was insensitive to the step-0 bug while video diverged;
   watching only the total would have masked it.
3. **fp32 loss reduction.** Ensure both sides cast to `.float()` before the
   MSE/mean reduction. A bf16 reduction has a different accumulate order and
   will produce a small but real step-1 delta.
4. **Live-code equivalence.** The forward you're comparing must be the *live*
   path. Confirm via per-file diff that every hunk is reformat/import/logging
   or provably-dead code, not a live numeric change (see model-adaptation
   Rule 2 for the classification method).

## Tools / techniques

- **Prove the weights before you audit ops** (`tools/weight_probe_snippet.py`,
  `tools/diff_weight_probe.py`). `WPROBE=1` dumps every param and buffer as
  `name shape dtype sha1` at forward entry; the differ reports, in priority
  order, name-set → shape → dtype → value. Fix a name-set difference first — a
  missing module invalidates every later grad-norm or activation comparison. The
  `dtype` column is also the cheapest way to test the parameter-storage-dtype
  hypothesis (bf16-direct params vs fp32 params + autocast), and the buffer
  entries catch a rope table computed in a different precision.
- **Then localize the first divergent op** (`tools/activation_probe_snippet.py`).
  `VPROBE2=1` hashes named fp32 intermediates on rank0. Bisect, don't
  instrument everything: loss inputs first (a matching target with a differing
  pred means the divergence is upstream of the loss, so stop auditing loss
  precision), then per-layer with every modality hashed at the same point, then
  intra-block on the one layer that first differs. `VPROBE2_DIR` also saves the
  tensors so you can measure whether the delta is 1 ULP or a real bug.
- **Standalone fixed-input forward.** Feed the identical batch + identical RNG
  state through both models and compare the loss scalar per channel. Step-1
  loss bit-exact ⇒ the live forward is equivalent.
- **FM_PROBE / batch-checksum blocks.** Gated (`PARITY_FM_SEED`) probes that
  checksum the sampled noise/timesteps and the batch tensors right before the
  forward, printed on both sides. These confirm Rule 1 held all the way into
  the forward entry, so a step-1 loss delta can't be blamed on inputs.
  ⚠️ These blocks do `.cpu()`/`.tolist()` host reads — they are parity-only and
  must stay gated off in perf runs (they serialize the eager preprocess).
- **Hunt the small step-1 delta — do NOT call it noise.** A ~0.3% step-1 delta
  with fixed seeds is a real divergence seed. Candidate causes, in order of
  frequency: an extra/missing token in the VLM/und sequence (prompt text
  source), rope/norm precision (fp64 rope, norm eps 1e-5 vs 1e-6), a bf16
  accumulate-order difference, a live attention-backend mismatch. Bisect by
  channel to localize which sub-forward carries the delta.
- **Measure the run-to-run band before declaring bit-exact.** Multi-rank bf16
  NCCL all-reduce is non-deterministic across runs (no seed on the reduction
  order). Run base 2–3 times and measure its own video-loss spread (in Motus,
  ~0.008 wide). If the transplant lands inside base's band, that channel is
  aligned — don't chase bits that base itself doesn't reproduce.

## Common ghosts (measurement, not math)

- "Transplant loss is noisier per step" → it's logging the all-reduce mean;
  switch to `--loss-log-rank 0`.
- "Step-1 matches but step-2 diverges" → this is NOT a forward bug. The forward
  is proven by the step-1 match. The divergence is in the step-1 *update* →
  go straight to Rule 3 (in Motus this was the step-0 LR quirk).

## Acceptance

- With `--loss-log-rank 0` and aligned inputs, step-1 forward loss is
  bit-exact (or within base's measured run-to-run band) for **every** loss
  channel.
- The live-code diff contains only reformat/import/logging/dead-code hunks.
- ⇒ The forward is equivalent. Divergence at step k≥2 is now a backward /
  optimizer question → hand to Rule 3.

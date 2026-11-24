# Rule 3 — Backward / Optimizer Alignment

**Goal:** with inputs (Rule 1) and the forward/loss (Rule 2) proven equivalent,
prove the *update* is equivalent — gradients, grad-norm, LR schedule, and the
optimizer step — so that step-2 weights (hence step-2 loss) match and the runs
stay together for all k. This is where "loss diverges at step 2" is almost
always born.

## The archetypal trap: step-0 LR ≈ 0 (fix this FIRST)

**Symptom:** step-1 loss/grad bit-exact, step-2 loss diverges (one channel far
more than others), *logged* LR identical on both sides.

**Cause:** torch's `LambdaLR.__init__` immediately multiplies
`optimizer.param_group['lr']` by `schedule(0)` (≈ `lambda_f_start`, often
~1e-6 or ~0). So the transplant's **first** optimizer step moves the weights by
≈0, while a base scheduler that does *not* touch `optimizer.lr` in `__init__`
takes a **full-LR** first step. The two step-2 weight sets differ by a factor
of ~10^5 in the first update → step-2 loss diverges. The *logged* LR is the
scheduler value (identical), so this hides unless you inspect the **actual
`optimizer.param_group['lr']` at step 0**.

**Fix (gated):** expose `--lr-step0-unscaled` (default off). In the
`lambda_linear` branch, reset each group's `lr` back to its `initial_lr`
(cancel the `schedule(0)` factor) for step 0 only; the first
`lr_scheduler.step()` after the step-0 update restores `base_lr*schedule(1)`,
so k≥1 is unchanged. This makes the transplant's step-0 update match base's.

> This single flag resolved a divergence that masqueraded for days as a
> compile / accelerate-wrapper / ZeRO-reduce bug. When step-1 is exact but
> step-2 diverges, **inspect the real step-0 LR before suspecting the engine.**

## What to align & prove (checklist)

1. **Per-parameter gradient checksum (step 1).** Before any optimizer step,
   checksum each parameter's grad on both sides. Under ZeRO use the framework's
   full-grad accessor (`safe_get_full_grad`) so you compare the *un-sharded*
   grad. Step-1 grads bit-exact ⇒ backward math is equivalent; a mismatch
   localizes to a specific module's params.
2. **Grad-norm per channel (GRAD_PROBE).** Register `Tensor.register_hook`
   probes to capture **pre-reduce** grad norms per loss channel (video/action/
   und). These fire independent of the wrapper (DDP/DeepSpeed/accelerate), so
   they're directly comparable across engines. Compare the first ~6 steps.
   Pre- vs post-reduce is another apples-to-apples trap — pick one and use it
   on both sides.
3. **Gradient clipping.** Match exactly. If base does NOT clip (no
   `gradient_clipping` in ds config, or grads already consumed in-graph), set
   the transplant clip to 0 (`--clip-grad 0.0`) — norm-only, no scaling. A
   stray clip silently rescales grads and diverges the update.
4. **Scheduler per-step LR.** Port the exact schedule formula (warmup, decay
   style, cycle length, f_max/f_min/f_start, base_lr). Verify the actual
   `optimizer.param_group['lr']` (not just the logged value) agrees per step
   for all k≥1, and — with the step-0 fix — for k=0 too.
5. **Optimizer + param groups.** Same optimizer class, betas, eps,
   weight_decay, and *which* params get weight decay (bias/norm carve-out vs
   all). One vs two param groups is fine iff the per-group LRs are numerically
   identical. Frozen params excluded from groups and from ZeRO shards.
6. **Gradient accumulation & loss scaling.** Match GA steps and where the loss
   is divided by GA (client vs engine-owned). With GA=1 this is a no-op.
7. **ZeRO / ds-config byte-diff.** Dump the *effective* ds config on both sides
   (batch sizes auto-injected: `train_micro_batch_size_per_gpu`,
   `gradient_accumulation_steps`, `train_batch_size = micro·GA·ws`) and diff
   field-by-field. Confirm `gradient_clipping` presence/absence matches. Verify
   ZeRO collectives actually fire — a silent no-reduce masquerades as
   divergence.

## Tools / techniques

- **GRAD_PROBE grad-norm table.** Log `phase=… fm_step=N video_gnorm=…
  action_gnorm=… und_gnorm=…` from rank0 for the first ~6 steps on both sides;
  tabulate side-by-side. This is the primary backward signal and cleanly shows
  which channel and which step first diverges.
- **safe_get_full_grad per-param dump.** For a precise localization, checksum
  every parameter's full (un-sharded) grad at step 1 and diff.
- **Effective-config dump script.** Construct both engines' configs and diff.
- **Isolate the wrapper as a single variable.** If you suspect the distributed
  carrier (accelerate vs direct deepspeed.initialize), add a gated path that
  swaps *only* the wrapper (same optimizer, same config, same compile) so the
  comparison is clean. (In Motus this was `ACCEL_COMPILE=1` — it ultimately
  ruled the wrapper out; the real cause was the step-0 LR above. Lesson: rule
  out the cheap measurement/schedule causes before building a wrapper harness.)

## Reference tools (`tools/`)

- **`dump_effective_ds_config.py`** — constructs BOTH sides' *effective* ds
  config (auto-injecting `train_micro_batch_size_per_gpu` /
  `gradient_accumulation_steps` / `train_batch_size = micro*GA*ws` the way each
  side does at runtime) and prints a field-by-field diff with `>>>` markers.
  Catches silent drift like `gradient_clipping` present on one side only.
  Edit the two json paths + the WORLD/MICRO/GAS geometry.
- **`grad_probe_snippet.py`** — the GRAD_PROBE paste-in snippet: three trainer
  methods that register `Tensor.register_hook` per parameter and print the
  per-submodule LOCAL (pre-reduce) grad norm plus a hook-coverage count `n=`.
  Not a standalone script — a gated hook has to live in the trainer it measures.
  Edit the `(bucket label, attribute name)` tuple to your submodules.
  🔑 Compare `n=` before the norms: identical counts prove the trainable set and
  hook coverage match, without which a "matching" norm can be a coincidence over
  two different parameter subsets. For per-parameter checksums under ZeRO use the
  framework's full-grad accessor (`safe_get_full_grad`), not `p.grad` — see the
  snippet's docstring for why `p.grad` is a stale flat-buffer view.

## Acceptance

- Step-1 per-parameter grads bit-exact (or per-channel grad-norm in-band).
- Grad clipping matched; per-step `optimizer.lr` agrees for all k (incl. k=0
  with `--lr-step0-unscaled`).
- With all of the above, step-2 loss matches base and the runs track together
  step-over-step for the first several steps → full parity is established.

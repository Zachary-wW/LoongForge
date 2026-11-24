# Rule 3 — Trainer Alignment

**Goal:** the transplant training loop is behaviorally equivalent to base:
same model wrapping, same optimizer + param groups, same scheduler, same
grad-clip / grad-accum, same distributed engine, same loss-logging semantics.

## What to align (checklist)

1. **Model wrapping / distributed carrier.** Identify who wraps the model and
   how. Common trap: both sides *think* they use DDP but actually a DeepSpeed
   engine owns everything. Match:
   - base: often `accelerate` `Accelerator(deepspeed_plugin=DeepSpeedPlugin(
     hf_ds_config=...))` + `accelerator.prepare(model, opt, sched)`.
   - transplant: often a direct `deepspeed.initialize(..., dist_init_required=
     False)`. The two build the *same* engine from the *same* config; the only
     difference is "who constructs it". Confirm no extra torch-DDP wrap sneaks in.
2. **Optimizer + param groups.** Same optimizer class, betas, eps, weight_decay,
   and *which params get weight decay* (bias/norm carve-out or "all"). One vs
   two param groups is fine **iff** the LRs are identical (numerically
   equivalent). Frozen params must be excluded from groups and from ZeRO shards.
3. **Scheduler.** Match warmup, decay style, cycle length, f_max/f_min/f_start,
   base_lr. Port the exact schedule formula; verify LR agrees per-step for k≥1.
   - **Step-0 LR quirk (critical):** a `LambdaLR.__init__` immediately multiplies
     `optimizer.lr` by `schedule(0)` (≈ f_start ≈ 0), so the *first* optimizer
     step uses ≈0 LR, while a base scheduler that does not touch `optimizer.lr`
     takes a full-LR first step. This makes step-2 weights (hence loss) diverge
     even when everything else matches. Expose a flag (e.g. `--lr-step0-unscaled`)
     to reset group LRs to `initial_lr` for step 0. See precision-alignment Rule 3.
4. **Grad clipping.** Match exactly. If base does NOT clip (e.g. in-graph reduce
   already consumed grads, or no `gradient_clipping` in the ds config), set the
   transplant clip to 0 (norm-only, no scaling).
5. **Gradient accumulation & loss scaling.** Match GA steps and where loss is
   divided by GA (client vs engine-owned). With GA=1 this is a no-op.
6. **DeepSpeed/DeepCompile config.** Diff the ds config byte-for-byte. Dump the
   *effective* config on both sides (batch sizes are auto-injected:
   `train_micro_batch_size_per_gpu`, `gradient_accumulation_steps`,
   `train_batch_size = micro·GA·ws`). Confirm `gradient_clipping` presence/absence
   matches. Log-only fields (e.g. `steps_per_print`) are numerically irrelevant.
7. **Loss-logging semantics (apples-to-apples).** A frequent false alarm: base
   logs **rank0-local** loss while transplant logs an **8-rank all-reduce mean**.
   The mean has ~1/√ws the per-step variance, so per-step numbers look
   "different" even when correct. Add a flag (e.g. `--loss-log-rank 0`) so both
   log rank0-local before comparing.

## Never compare loss across engines (and check param dtype before hunting ops)

Different engines *can* hold trainable parameters in different dtypes during the
forward, and that alone changes the forward result even with identical weights on
disk and identical inputs:

- **bf16-direct** (e.g. accelerate DDP + a bf16 ZeRO optimizer with no fp32
  master): the parameters themselves are bf16, so every GEMM sees a bf16-rounded
  weight.
- **fp32 master + autocast** (e.g. DeepSpeed ZeRO with `bf16.enabled`): the
  parameters stay fp32 and autocast casts them per-op, so rounding happens at a
  different point and reductions may differ.

So when a step-1 **forward** mismatch survives identical inputs (batch checksum)
and identical weights (weight-dump md5), **dump the actual `p.dtype` histogram of
the trainable params at forward time on both sides first**. It is a two-line probe
and it is much cheaper than auditing operators or env vars.

⚠️ But *verify* it rather than assuming it — in the Motus port this hypothesis was
attractive and turned out to be **false**, and reading the code settled it with no
GPU run at all: the model builder casts every expert to bf16 itself, the DDP-path
wrapper (`MultiDtypeZeroOptimizer`) only *buckets* params by their existing dtype
and never casts, and DeepSpeed's `module.bfloat16()` only touched the two modules
the builder had deliberately put in fp32 — which held bf16-exact values, so the
downcast was lossless. That is exactly why `action_loss` was bit-identical across
engines. Autocast was not the differentiator either: `accelerate` sets
`native_amp=False` under DeepSpeed and `True` under DDP+bf16, but it patches only
`model.forward`, and this trainer calls `model.training_step(...)` on the
unwrapped module — so **neither** path autocasted.

The durable lesson is the *first* sentence, not the mechanism: **a cross-engine
number is not a parity datapoint.** Pick BASE's canonical engine, put both sides
on it, and compare like-for-like. In this port BASE's own step-1 `video_loss` read
1.1615263223648071 on its DDP path and 1.162245750427246 on its DeepSpeed path,
and that 6.2e-4 was used as evidence for the dtype story — but those two runs'
inputs were never checksummed against each other, so the delta is better read as
"two runs differing in more than one variable" than as a measured engine effect.
Never let an unverified cross-engine delta become the premise of an investigation.

Consequences:

1. **Never compare loss across engines.** Pick BASE's canonical engine and
   compare like-for-like.
2. On a surviving step-1 forward mismatch, dump `p.dtype` on both sides — the
   `precision-alignment` skill's `tools/weight_probe_snippet.py` already prints a
   `dtype` column per parameter, so this is free once WPROBE is in. But treat it
   as a *hypothesis to kill*, and prefer killing it by reading the model
   builder / wrapper code (free) over another 8-GPU run.
3. Env/allocator differences are a still weaker prior — measured directly,
   `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` left step-1 bit-identical
   on this 30-layer bf16 stack.

## The DeepSpeed engine ownership contract (7 items)

If BASE's canonical engine is DeepSpeed, the transplant's generic
wrap/backward/step plumbing almost certainly assumes it owns the optimizer. It
does not. Every item below caused a real, silent divergence in the Motus port —
copy them wholesale when adding a `deepspeed` strategy:

1. **Return the CLIENT optimizer, not `engine.optimizer`.** `deepspeed.initialize`
   returns a `DeepSpeedZeroOptimizer` that has no `.defaults`, which a scheduler
   reading `optimizer.defaults["lr"]` needs. ZeRO steps the client AdamW in-place
   through the live `param_groups`, so LR writes to the client object still apply.
2. **Loss divisor is 1 under DeepSpeed.** `engine.backward()` applies
   `_scale_loss_by_gas` itself; dividing by `grad_accum` again divides twice.
   (Watch for the mirror bug in BASE: if base does
   `accelerator.backward(loss / gas)` under DeepSpeed it *is* dividing twice —
   only equivalent at GAS=1. Match base's *intent*, and note the discrepancy.)
3. **`engine.backward(loss)` + `engine.step()` every micro-step.** That is the
   contract; the engine no-ops the update on non-boundary micro-steps. Don't gate
   `step()` on `is_last`.
4. **Never call `optimizer.zero_grad()`.** `engine.step()` clears at the
   boundary; zeroing a ZeRO flat partition buffer corrupts the
   `contiguous_gradients` bookkeeping.
5. **Never clip by hand.** The engine clips inside `step()` per the config's
   `gradient_clipping`. Host-side `p.grad` is a view into the flat partition
   buffer, so a manual clip reads the wrong thing. Read the pre-clip global norm
   after the fact via `engine.get_global_grad_norm()`.
   → If the ds json has **no** `gradient_clipping` key, base does **not clip**;
   the transplant must be run with its clip disabled (`--clip-grad 0`) for parity.
6. **Skip host-side NaN-gradient cleanup** for the same buffer-aliasing reason.
7. **Forward on the UNWRAPPED module** when the entry point isn't `forward`
   (e.g. `raw.training_step(...)`). The engine has nothing to intercept on a
   non-`forward` method, and the reducer's per-parameter autograd hooks fire
   regardless.

**Validate the divisor empirically**: run GAS=1 vs GAS=2 at the same global batch
and compare step-1 grad norm. A 2× divisor error shows up as a ~2× or ~0.5×
grad norm; a correct divisor gives a ratio near 1 (Motus: 21.786 vs 21.420).

Structural note: adding a DeepSpeed strategy touches training_args → wrap →
optimizer build → backward → step → validators → example launcher, because
`deepspeed.initialize` **inverts** the framework's usual wrap-then-build-optimizer
order (it needs the optimizer to build the engine).

## Tools / techniques

- **`git diff` / manual diff the launch scripts and ds json.** Most trainer
  drift is in the launch flags, not the code.
- **Effective-config dump script.** Construct both engines' configs and diff
  field-by-field (`/tmp/dump_ds_cfg.py`-style).
- **Gate parity-only trainer behavior behind env flags** (default inert) so the
  production trainer path is byte-unchanged. Keep custom-model trainers isolated
  from generic engine paths so a stray `--deepspeed-config` can't half-engage them.

## Reference tools

- The effective-config dump/diff script lives in the `precision-alignment`
  skill's `tools/dump_effective_ds_config.py` — it constructs both sides'
  runtime ds config (batch fields auto-injected) and diffs field-by-field. Use
  it here to verify the trainer builds the same engine config on both sides.

## Acceptance

- With aligned data + model, and matched LR/clip/logging, step-1 loss is
  bit-exact and per-step LR agrees for all k≥1. Remaining divergence is now a
  pure numerics question → hand to `precision-alignment`.

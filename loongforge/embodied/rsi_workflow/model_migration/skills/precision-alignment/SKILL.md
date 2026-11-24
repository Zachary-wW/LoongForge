---
name: precision-alignment
argument-hint: "<BASE_PATH: reference run, read-only> <TRANSPLANT_PATH: ported run> — in that order"
description: >-
  Playbook for proving bit-level / band-level numerical parity between a
  reference training run (BASE) and a ported one (TRANSPLANT), step by step.
  Use after `model-adaptation` has made the two functionally equivalent, when
  you must show the transplant produces the same loss and gradients as base
  and diagnose any per-step divergence. Covers three rules: (1) training input
  alignment, (2) forward / loss alignment, (3) backward alignment (gradients,
  grad-norm, LR schedule + optimizer update). Trigger phrases: "精度对齐",
  "loss 对齐", "梯度对齐", "prove parity", "loss diverges at step N".
---

# Precision Alignment (BASE ≈ TRANSPLANT)

## Inputs (paths)

This skill compares two runs: **BASE** (the reference run / repo — read-only
ground truth) and **TRANSPLANT** (the ported run being proven).

Resolve them in this priority order:

1. **From the invocation arguments**, read in order — first path = BASE, second
   path = TRANSPLANT. They are rendered on the next line:

   > args: `$ARGUMENTS`

   ⚠️ Parse that line yourself. Positional placeholders (a dollar sign followed
   by a digit) are **not** reliable here and this file deliberately contains
   none: in a real invocation they were substituted 0-indexed, so the
   dollar-one slot rendered the *second* path and dollar-two rendered empty,
   silently **inverting** BASE and TRANSPLANT. They are spelled out in words
   above because a skill file that writes them literally gets its own warning
   substituted away too.
2. **If the args line is empty or carries only one path** — it renders empty
   whenever the skill is triggered by description match instead of an explicit
   invocation — take the paths from the user's message, or infer them from the
   current working directory and recent context (a `bak_*/<model>` reference tree
   vs an in-house trainer tree). If `model-adaptation` already ran in this
   session, reuse the exact roots *and landing sub-directory* it resolved rather
   than re-deriving them. Confirm anything you inferred with the user.
3. **Verify before running anything.** A swapped mapping here does not just
   mis-label a number — it makes you write probe code, logs and checkpoints into
   the read-only reference tree. `ls` both roots (BASE = single-model research
   repo; TRANSPLANT = multi-model framework with registries/`trainers/`), and
   confirm which side you are allowed to edit.
4. **Resolve the two output locations too.** Parity comparison needs each side's
   log / tensorboard dir, and BASE's must be redirected under `/tmp` so its tree
   stays pristine. A run whose outputs you cannot locate is a run you cannot
   compare — settle this before launching, not after.

Echo the resolved mapping back before starting:
`BASE = <path> (read-only, logs -> /tmp/...)  |  TRANSPLANT = <path> (logs -> ...)`
Every "base:" / "transplant:" reference in the rules below is relative to these
two roots.

A structured method for *proving* numerical parity between a reference run
(**BASE**) and a port (**TRANSPLANT**), and for localizing the exact step and
tensor where they first diverge. Assumes the two are already functionally
aligned by the `model-adaptation` skill (same batches, same module math, same
loop). This skill turns "looks about right" into "step-1 is bit-exact and
per-step loss/grad/LR agree".

## Golden rules

1. **One variable at a time.** Every parity experiment changes exactly one
   thing between the base run and the trans run. If two things differ, the
   result is uninterpretable — you will chase ghosts.
2. **Gate all parity behavior behind default-off flags.** Every alignment hack
   (fixed seeds, LR step-0 reset, rank0-local logging, grad probes) lives
   behind an env var or CLI flag that is inert by default, so the production
   path is byte-unchanged.
3. **Compare apples to apples.** The #1 false alarm is a *measurement*
   mismatch, not a *math* mismatch (rank0-local vs all-reduce-mean loss,
   scheduler-LR vs optimizer-LR, pre- vs post-reduce grad). Fix the yardstick
   before blaming the model.
4. **Bit-exact where determinism exists; band where it doesn't.** A fixed-seed
   single-rank forward should be bit-exact. A multi-rank bf16 forward/backward
   (NCCL all-reduce, and especially flash-attn's bf16 backward, which uses
   non-deterministic atomic adds) is *not* reproducible run-to-run.
   **Measure BOTH sides' run-to-run spread, ≥3 runs each, before judging.**
   - n=2 gives a misleadingly narrow band. In the Motus DeepSpeed port base's
     step-2 video loss looked like a 3.0e-5 band from 2 runs and the transplant
     sat 6e-5 outside it — "clear divergence". Four transplant runs showed its
     own band was 1.0e-4 wide and **fully contained base's band**: pure noise.
   - **Size the band per-quantity, not globally.** Raw pre-reduce grad norms
     jitter ~1 order of magnitude more than losses (Motus: loss ~1e-5 relative,
     `video_gnorm` ~2e-4 relative *within one side*). A 3e-4 grad-norm delta can
     be in-band while a 3e-4 loss delta is a real bug.
   - Only the reference side being run twice is the classic mistake: you end up
     comparing a wide unknown distribution against a narrow sample.
5. **Don't dismiss a small delta as noise.** A 0.3% step-1 delta with fixed
   seeds is a real seed of divergence (an extra token, a norm eps, a rope
   cast) — hunt it. "Noise" is only noise once you've measured the band.
6. **Read full-precision numbers, not the stdout log.** Reference repos often
   print 4 decimals (`Video: 0.3775`), which cannot distinguish a 3e-5 band from
   a 2e-4 divergence. Pull fp32 values out of the reporting backend — for
   tensorboard use `tools/read_tb.py`. Also note that an fp32-logged scalar
   agreeing "bit-exactly" only proves ~6e-8 relative agreement; it does not
   prove the bf16 activations were identical.
7. **Verify the probe's own coverage count matches.** Any per-parameter probe
   should report how many tensors it accumulated (`n=825/292/241`). Identical
   `n` on both sides proves the trainable parameter set and the hook coverage
   are the same — without it, a "matching" grad norm can be a coincidence over
   different parameter subsets.

## The three rules

Apply in order — each depends on the previous being clean.

- **Rule 1 — Training input alignment** (`rules/01-input-alignment.md`)
  Prove both sides feed the model the *same* tensors at step k: data anchor
  seed, per-step flow-matching / noise RNG, VLM prompt source, worker seeding,
  standalone dataloader checksum.

- **Rule 2 — Forward / loss alignment** (`rules/02-forward-loss-alignment.md`)
  Prove the forward is equivalent: step-1 loss bit-exact per channel,
  rank0-local logging, per-file live-code diff, hunting the step-1 delta,
  FM/batch checksums.

- **Rule 3 — Backward / optimizer alignment** (`rules/03-backward-optimizer-alignment.md`)
  Prove the update is equivalent: per-parameter grad checksum, grad-norm
  (video/action/und channels), grad clipping, the step-0 LR quirk, per-step
  scheduler LR match, optimizer + ZeRO + ds-config alignment.

## Workflow

1. Lock inputs (Rule 1) until a standalone dataloader diff is byte-identical
   for the first ~6 steps.
2. Lock the forward (Rule 2) until step-1 loss is bit-exact (or in-band) on
   every loss channel.
3. Lock the backward (Rule 3): step-1 grads bit-exact, then walk k=2,3,…
   until per-step loss/grad/LR agree.
4. Record every classified diff and every resolved divergence to memory so you
   never re-litigate a settled question.

## Tools

`tools/` holds reference implementations from a real port — see
`tools/README.md`. Runnable scripts: the standalone dataloader checksums
(Rule 1), the WPROBE dump differ, the effective ds-config diff (Rule 3), the
fp32 tensorboard reader, and the GPU reaper. Paste-in snippets (a gated probe
must live inside the model/trainer it measures): `weight_probe_snippet.py`
(WPROBE — weights + param storage dtype), `activation_probe_snippet.py`
(VPROBE2 — locate the first divergent op), `grad_probe_snippet.py` (GRAD_PROBE —
pre-reduce local grad norms). Most are project-specific templates with `EDIT-ME`
markers.

On a step-1 mismatch use them in this order: batch checksum → `WPROBE` +
`diff_weight_probe.py` → `VPROBE2` bisect → `GRAD_PROBE`. An activation bisect
run over unproven inputs or unproven weights only re-discovers that the
inputs/weights differ.

## Ops pitfalls that will eat your day

- **A "finished" run can keep holding every GPU.** DeepSpeed + accelerate runs
  routinely print the last step, save, and then hang in teardown (`dist` already
  torn down, GC re-enters `print_rank_0` → `AssertionError: DeepSpeed backend not
  set`). The next run then OOMs with a bogus "tried to allocate 18 MiB" while
  another process holds 70 GB. **Standard reaping step between every pair of
  runs**: `tools/reap_gpus.sh "<pattern matching the run's command line>"`,
  which SIGKILLs the matches and then *waits until `nvidia-smi` actually drops
  below 100 MiB* before returning 0. Budget this into every A/B — and do not
  hand-roll it with `pkill -f "$PAT"`, because the killer's own argv contains
  `$PAT`, so it kills itself and reports success while the real run keeps the
  GPUs.
- **`nvidia-smi` PIDs are HOST PIDs, invisible inside a container.** `ps -p` will
  exit 1 and `/proc/<pid>/cmdline` will not exist. Match by command line with
  `pgrep -fa "train/train.py"` and kill the container-local PID instead.
- **Redirect the reference run's outputs.** Point `checkpoint_dir` / log dir at
  `/tmp/...` so BASE's tree stays pristine (golden rule: BASE is read-only).
- **Shell cwd does not persist** across tool calls in some setups — use absolute
  paths in launchers, never a relative `torchrun train/train.py` that depends on
  an earlier `cd`.
- **Never fabricate a baseline.** If a regression baseline file is missing,
  collect it from a real run. A hand-written "expected" number turns every
  future comparison into a comparison against your own guess.
- **Prefer an existing framework flag over inventing an env-var hack.** Check
  whether the transplant already exposes the knob (e.g. `--clip-grad 0`,
  `--loss-log-rank`, `--no-sampler-shuffle`) before adding a new gate.

## The cardinal debugging pattern: "loss diverges at step 2"

This is the archetypal symptom. The decisive move is to **bisect by step and
by quantity**:

- step-1 loss bit-exact + step-2 loss diverges ⇒ the divergence is in the
  step-1 *update* (grad, LR, or optimizer), NOT the forward. Go to Rule 3.
- step-1 loss already differs ⇒ inputs or forward. Go to Rule 1 then 2.

In the Motus port, "step-2 video loss 1.29 vs 0.37" looked like a compile /
wrapper bug for days; the real cause was a **step-0 LR of ~0** in the
transplant's `LambdaLR` (Rule 3). Bisecting step-1(exact)→step-2(diverge)
pointed straight at the update, not the forward — trust the bisection.

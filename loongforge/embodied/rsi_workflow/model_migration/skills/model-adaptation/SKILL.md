---
name: model-adaptation
argument-hint: "<BASE_PATH: reference repo, read-only> <TRANSPLANT_PATH: target framework> — in that order"
description: >-
  Playbook for porting a model's training stack from a reference (BASE)
  codebase into a target training framework (TRANSPLANT). Use when adapting /
  transplanting a model into another training system (e.g. porting a research
  repo into an in-house trainer) and you need the ported code to be
  functionally equivalent to the source. Covers three rules: (1) dataset +
  sampling alignment, (2) network / architecture (组网) alignment, (3) trainer
  alignment. Pair with the `precision-alignment` skill to then prove numerical
  parity. Trigger phrases: "适配模型", "移植模型", "port model to <framework>",
  "align transplant with base".
---

# Model Adaptation (BASE → TRANSPLANT)

## Inputs (paths)

This skill operates on two roots: **BASE** (the reference / open-source repo —
read-only ground truth you must NOT edit) and **TRANSPLANT** (the target
training framework the model is being ported into).

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
   current working directory and recent context (an obvious `bak_*/<model>`
   reference tree vs an in-house trainer tree). Then ask the user to confirm
   anything you inferred, and treat their answer as authoritative over the args.
3. **Verify the assignment mechanically before touching anything.** Argument
   order alone is not enough, because a swapped mapping means you will edit the
   read-only ground truth. `ls` both roots: BASE should look like a single-model
   research repo (`train/train.py`, `configs/*.yaml`, the model code near the
   top level); TRANSPLANT should look like a multi-model framework (registries,
   `trainers/`, `examples/`, many unrelated models). If the roles look swapped,
   stop and confirm with the user.
4. **Also resolve the landing sub-directory, not just the framework root.** The
   framework root is rarely where the port goes — the caller usually means
   something like `<TRANSPLANT>/<area>/<domain>/`. "Port into the framework" and
   "port into `<framework>/loongforge/embodied`" produce very different file
   trees, so ask for it if it was not given.

Echo the resolved mapping back before starting so it is unambiguous:
`BASE = <path> (read-only)  |  TRANSPLANT = <path>  |  landing dir = <path>`
Every "base:" / "transplant:" reference in the rules below is relative to these
two roots.


A structured method for porting a model's *training* stack out of a reference
repo (**BASE**, the ground-truth reference you must NOT edit) into a target
framework (**TRANSPLANT**). The goal of this skill is **functional
equivalence**: the transplant produces the same batches, the same network math,
and runs the same optimizer/loop as base. Bit-level numerical parity is proven
separately with the `precision-alignment` skill.

## Golden rules

1. **BASE is read-only ground truth.** Never edit base to make things match
   unless the user explicitly approves (e.g. fixing a real base bug). If you
   must edit base for a parity experiment, gate it behind an env flag that is
   inert by default.
2. **Diff before you assume.** Before claiming two implementations differ,
   run a real diff (`sed 's/[[:space:]]*$//'` to kill trailing whitespace) or
   a standalone reproduction. Most apparent differences are dead code, import
   paths, or reformatting.
3. **Separate live code from dead code.** A file can have 76 diff lines that
   are all in a branch that never executes. Trace the actual runtime path first.
4. **One variable at a time.** When something diverges, change exactly one
   thing between runs so the cause is unambiguous.
5. **Gate every alignment hack.** Parity-only behavior lives behind a default-off
   env var or CLI flag so production behavior is untouched.
6. **Identify BASE's *canonical* engine from its own launch scripts and the
   pristine upstream — not the fork's default code path.** A research fork's
   `train.py` default is often a local experiment (DDP + `torch.compile`,
   DeepCompile, …) while `scripts/train.sh` / upstream README show the real
   engine (e.g. `USE_DEEPSPEED=1` + `configs/zero1.json`). Porting against the
   wrong engine wastes the entire alignment effort — confirm with the user before
   committing to one.
7. **Check what the historical reference port already achieved** before chasing
   a divergence. If a prior parity-proven port of the same model exists, diff
   against it and classify every hunk as dead-code / probe / live-op. Most
   "differences" resolve to the first two.

## Rule 0 — Framework contract & registration (from-scratch ports)

Before any numerics work, land the model in the target framework's own
abstractions, or you will fight the framework instead of the math:

- **Enumerate the registries** the framework expects an entry in (model, trainer,
  dataset, optimizer, config schema) and register there rather than bolting on a
  parallel path. Run a **registration smoke test** (`build_model(cfg)` /
  `build_dataloader(...)` / trainer construction) before any GPU run.
- **Probe the framework's contracts**: what does `_wrap_model_for_training`
  guarantee? Who owns `zero_grad` / clipping / `step()`? What signature must
  `forward` have, and which kwargs get silently dropped if unaccepted? Silently
  dropped kwargs are the dangerous case (e.g. an `iteration` that seeds the
  deterministic noise RNG).
- **Enumerate launcher env deltas** between BASE's script and the transplant's
  example script: `PYTORCH_CUDA_ALLOC_CONF`, `CUBLAS_WORKSPACE_CONFIG`,
  `DISABLE_ADDMM_CUDA_LT`, `CUDA_DEVICE_MAX_CONNECTIONS`, `TORCHINDUCTOR_*`,
  NCCL vars. These can change numerics and are invisible in a code diff, so list
  them — but **rank them below dtype/precision differences, and verify by single
  variable before believing one.** Measured counter-example: adding
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to one side of a 30-layer
  bf16 video-diffusion tower left step-1 loss **bit-identical** (8.723114 /
  1.162246 / 7.560868 with and without). The "different allocator → different
  tensor alignment → different cuBLASLt kernel → 1-ULP drift" story is plausible
  and is a real mechanism in principle, but it did NOT fire here. A step-1
  forward mismatch that survives identical inputs and identical weights should
  send you first to **the dtype the parameters are actually held in during the
  forward** (bf16-direct params vs fp32 params + autocast) — but settle that by
  *reading the model builder / wrapper code*, which is free, before spending
  another 8-GPU run. In this port both engines turned out to hold bf16 params and
  neither autocasted, so that hypothesis died too — see
  `rules/03-trainer-alignment.md`. The rule that survived both dead hypotheses:
  **a cross-engine loss number is not a parity datapoint at all.**

- **Note engine capability gaps early.** If BASE's engine has no strategy in the
  transplant, adding it is a cross-cutting change (see the DeepSpeed contract in
  `rules/03-trainer-alignment.md`), not a one-file edit.

## The three rules

Apply in order. Each has a detailed playbook in `rules/`.

- **Rule 1 — Dataset & sampling alignment** (`rules/01-dataset-sampling-alignment.md`)
  Make the transplant emit the same samples in the same order: dataset
  indexing/bounds, video/media decode, transforms/normalization, anchor
  sampling RNG, distributed sharding, worker seeding, drop_last geometry.

- **Rule 2 — Network / architecture (组网) alignment** (`rules/02-network-architecture-alignment.md`)
  Make the module tree and every live forward op numerically equivalent:
  precision of critical ops (rope, norm, loss), attention backend, config
  eps/hyperparams, frozen-module definitions.

- **Rule 3 — Trainer alignment** (`rules/03-trainer-alignment.md`)
  Make the training loop equivalent: model wrapping (DDP/DeepSpeed/FSDP),
  optimizer construction + param groups, scheduler, grad clipping, grad
  accumulation, loss logging semantics, distributed engine construction.

## Workflow

1. Map both trees: list BASE and TRANSPLANT files for data / model / trainer.
2. Do Rule 0 (registration + framework contracts + engine choice) first if this
   is a from-scratch port.
3. Work Rule 1 → 2 → 3. After each, record findings to memory.
4. Once functionally aligned, hand off to `precision-alignment` to prove parity
   step-by-step.

## Tools

`tools/` holds runnable reference implementations (from a real port) — see
`tools/README.md`. For Rule 1: the sampler-order / real-sampler-index /
double-shard probes (CPU-only templates with `EDIT-ME` markers). For Rule 2/3:
`codenorm.py` + `diff_live_code.sh`, which normalize away formatting so a
per-file diff shows only real code changes (project-independent — run this
*before* claiming two implementations differ).

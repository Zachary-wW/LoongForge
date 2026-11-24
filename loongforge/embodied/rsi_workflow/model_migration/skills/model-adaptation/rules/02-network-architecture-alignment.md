# Rule 2 — Network / Architecture (组网) Alignment

**Goal:** the module tree and every *live* forward operation in the transplant
are numerically equivalent to base. Structural equivalence is not enough —
op-level precision and the actually-executed code path must match.

## What to align (checklist)

1. **Module tree / construction.** Diff the `__init__` build order and submodule
   wiring. Reformatting and import-path changes (`from wan...` vs `from .wan...`)
   are fine; a different layer count / dim / order is not.
2. **Precision of numerically-sensitive ops.** These commonly diverge on port:
   - **RoPE** in fp64 (cast `cos`/`sin` to `.double()` where base does).
   - **Loss reductions** in fp32 (`.float()` before MSE/mean).
   - **Norm eps** values (e.g. `1e-5` vs `1e-6`) — a silent config drift.
3. **Attention backend.** The single biggest hidden trap. Determine which
   attention function *actually runs* on the training path and match it exactly
   (e.g. `flash_attn.flash_attn_func` dense vs `sdpa` with a forced backend vs
   varlen FA). Many attention files carry 2-3 implementations where only one is
   live — align the live one, ignore the dead ones (but confirm they're dead).
4. **Config / hyperparams.** eps, dropout, window sizes, expert ratios,
   `batch_size` assumptions baked into shapes.
5. **Frozen modules.** Which params are frozen (`requires_grad=False`), whether
   frozen submodules run under `no_grad`, and whether they're in `train()` vs
   `.eval()` mode. Mismatched train/eval on a frozen module with dropout would
   break parity (verify dropout=0 or eval matches).
6. **Shape/geometry derivation.** Prefer deriving grid/patch sizes from actual
   tensor shapes at runtime rather than a build-time `config.batch_size`, so the
   ported model survives batch sizes ≠ the config default.

## Tools / techniques

- **Full per-file diff of the live path — normalized, not raw**
  (`tools/diff_live_code.sh`, wrapping `tools/codenorm.py`). A raw diff of two
  ports is mostly reformatting: line reflow, added type hints on unchanged
  signatures, docstrings, comments. `codenorm.py` reduces each file to code-only
  tokens (comments/docstrings/blank lines dropped, statements re-emitted one per
  line) so the diff shows only REAL code changes; a 76-line raw diff routinely
  normalizes to zero and the wrapper prints `IDENTICAL after normalization`.
  Then classify every surviving hunk as: (a) reformat/import/logging — ignore;
  (b) dead code — confirm unreachable on the runtime path; (c) probe/gated
  parity code — ignore if default-inert; (d) live numeric difference — fix. Only
  (d) matters. Write the classification to memory so you don't re-litigate it.
- **Trace the runtime path.** Set breakpoints / read the branch conditions
  (e.g. a `use_sdpa` flag set by an `enable_static_path()` method) to prove
  which attention/forward branch executes. A 76-line attention diff can be
  100% dead code.
- **Standalone forward on a fixed input.** Once data is aligned (Rule 1), feed
  an identical batch + identical RNG state through both models and compare the
  loss scalar. Step-1 loss bit-exact ⇒ the live forward is equivalent. A small
  (e.g. 0.3%) step-1 delta is a real seed of divergence — hunt it (candidates:
  rope/norm precision, an extra token in the und/VLM sequence, a bf16 accumulate
  order). Do NOT dismiss it as "noise".

## Acceptance

- Live-code diff = only reformat/import/logging/dead-code hunks.
- With aligned data + RNG, step-1 forward loss is bit-exact across BASE and
  TRANSPLANT for every loss channel.

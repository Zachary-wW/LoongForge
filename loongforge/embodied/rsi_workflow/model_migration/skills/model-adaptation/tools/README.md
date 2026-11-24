# model-adaptation tools

Reference implementations of the tools used during the BASE→TRANSPLANT port.
Some are project-specific templates (from the Motus port) with `EDIT-ME` markers
for repo paths, launch flags, and sampler constructors; the rest are
project-independent. All are CPU-only and touch no model/GPU, so they run fast
for quick isolation.

| tool | rule | what it proves | EDIT-ME? |
|---|---|---|---|
| `sampler_order_probe.py` | Rule 1 | base vs transplant sampler emit the same rank0 index order (synthetic `range(N)`, sweeps drop_last tails) | yes |
| `real_sampler_idx_dump.py` | Rule 1 | the transplant's *real* registered sampler on the *real* dataset takes the no-shuffle parity path and dumps actual per-step indices | yes |
| `double_shard_check.py` | Rule 1 | exposes the `accelerate.prepare` double-shard trap (base step g ≡ transplant step g·ws) | yes |
| `codenorm.py` | Rule 2/3 | normalizes a `.py` to code-only tokens (drops comments/docstrings/blank lines/reflow) so a diff shows only REAL code changes | no |
| `diff_live_code.sh` | Rule 2/3 | wrapper: normalizes a BASE file and its TRANSPLANT counterpart and diffs them; prints "IDENTICAL after normalization" when the whole diff was formatting | no |

Use `diff_live_code.sh` **before** claiming two implementations differ (golden
rule 2). A 76-line raw diff routinely normalizes to zero. Every hunk that
survives must then be classified as dead-code / probe / live-op — only live-op
matters.

Content-level batch checksums (proving the *tensors* match, not just the
indices) live in the `precision-alignment` skill's `tools/`, as do the weight /
activation / gradient probes and the effective ds-config diff used in Rule 3.

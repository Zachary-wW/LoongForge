# precision-alignment tools

Reference implementations of the tools used to *prove* numerical parity between
BASE and TRANSPLANT. Some are project-specific templates (from the Motus port)
with `EDIT-ME` markers for repo paths, dataset APIs, launch flags, batch key
names, and submodule names; the rest are project-independent.

Three of them are **paste-in snippets, not runnable scripts**: a gated probe has
to live inside the model / trainer it measures, so the file ships the code plus
the placement strategy and the traps. Add the same snippet to BOTH sides, run
one step each, and diff the printed lines.

| tool | kind | rule | what it proves | EDIT-ME? |
|---|---|---|---|---|
| `dataloader_checksum_base.py` | script | Rule 1 | per-key sha1 of the first N BASE batches (single-proc, no-shuffle, `PARITY_DATA_SEED`) | yes |
| `dataloader_checksum_trans.py` | script | Rule 1 | the TRANSPLANT twin (forces `SequentialSampler` — ws=1 often defaults `shuffle=True`); diff `[TRANS_CKSUM]` vs `[BASE_CKSUM]` step-by-step | yes |
| `weight_probe_snippet.py` | snippet | Rule 2 | `WPROBE=1` dumps every param/buffer as `name shape dtype sha1` at forward time — identical weights, and the `dtype` column doubles as the parameter-storage-dtype check | yes |
| `diff_weight_probe.py` | script | Rule 2 | diffs two WPROBE dumps in priority order name-set → shape → dtype → value; groups value mismatches by subtree; exit 1 on any mismatch | no |
| `activation_probe_snippet.py` | snippet | Rule 2 | `VPROBE2=1` layer-by-layer fp32 activation hashes — localizes the FIRST divergent op ("L4 video differs, L4 action/und identical") | yes |
| `grad_probe_snippet.py` | snippet | Rule 3 | `GRAD_PROBE=1` per-submodule LOCAL (pre-reduce) grad norm via autograd hooks, with the `n=` coverage count that proves the trainable set matches | yes |
| `dump_effective_ds_config.py` | script | Rule 3 | constructs BOTH sides' runtime ds config (batch fields auto-injected) and diffs field-by-field; catches silent `gradient_clipping`/ZeRO drift | yes |
| `read_tb.py` | script | Rule 2/3 | dumps **fp32** scalars from a tensorboard event file — mandatory when the reference only prints 4 decimals to stdout | no |
| `reap_gpus.sh` | script | ops | kills a finished-but-still-holding run and waits for GPU memory to actually drop — run between EVERY pair of A/B runs, or the next run dies of a nonsense OOM while a zombie holds 70 GB | no |

Order of use on a step-1 mismatch: batch checksum (Rule 1) → `WPROBE` +
`diff_weight_probe.py` → `VPROBE2` bisect → `GRAD_PROBE`. Never skip ahead: an
activation bisect over different inputs or different weights only tells you the
inputs/weights differ.

Sampler-order probes (proving *which indices* map to each step, before content)
and the normalized live-code diff live in the `model-adaptation` skill's
`tools/`.

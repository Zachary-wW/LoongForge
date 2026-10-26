# Megatron Core v0.19.2 Migration Evidence

This directory contains reviewed evidence for moving LoongForge from its
historical Megatron Core base to the Loong-Megatron `v0.19.2` integration
branch.

## Artifacts

- [Copy inventory](LOONGFORGE_MEGATRON_COPY_INVENTORY.md): reviewed source
  provenance, upgrade scope and confidence levels.
- [Inventory CSV](inventory.csv): one row per reviewed file.
- [Inventory JSON](inventory.json): machine-readable form of the same data.
- [Similarity results](similarity_results.csv): structural comparison against
  Megatron Core v0.10, v0.12-v0.15 and v0.19.2.
- [Test matrix](TEST_MATRIX.md): shared test levels, cases and representative
  configurations.
- [Issue map](ISSUE_MAP.md): existing fork Issues and their Copy ID coverage.

## Boundaries

- The inventory distinguishes confirmed source-derived files from model-side
  candidates that still require provenance review.
- Similarity scores locate likely historical source snapshots; they do not
  prove that every line originated upstream.
- This directory does not change LoongForge runtime code or the Loong-Megatron
  submodule.
- Runtime validation starts only after a fetchable v0.19.2 integration head is
  frozen.

## Reproduction

The scripts accept repository paths through CLI arguments and do not embed
developer-specific absolute paths:

```bash
python tools/migration/megatron_copy_audit/scan_similarity.py \
  --loongforge-root . \
  --reference-root /path/to/megatron-version-checkouts \
  --output-dir /tmp/lf-copy-audit

python tools/migration/megatron_copy_audit/build_inventory.py \
  --loongforge-root . \
  --similarity /tmp/lf-copy-audit/similarity_results.json \
  --output-dir /tmp/lf-copy-audit

python tools/migration/megatron_copy_audit/render_inventory.py \
  --inventory /tmp/lf-copy-audit/inventory.csv \
  --output /tmp/lf-copy-audit/LOONGFORGE_MEGATRON_COPY_INVENTORY.md
```

The reference directory must contain extracted Megatron-LM trees named
`core_v0.10.0`, `core_v0.12.0`, `core_v0.13.0`, `core_v0.14.0`,
`core_v0.15.0` and `core_v0.19.2`.

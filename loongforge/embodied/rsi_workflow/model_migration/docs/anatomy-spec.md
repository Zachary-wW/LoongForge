# Frozen AnatomySpec and VLA action evidence

`core/contracts/anatomy_spec.py` defines immutable stdlib-only `AnatomySpec`,
`ActionSpec` and `ViewSpec`, re-exported by Stage 2's `contract.py`.
The spec schema is **2**, independently of the still-loadable v1 F1/F2 schemas.
`pi05` is a preset, not a universal requirement. Unknown fields and mutable
constructor lists are rejected. `to_dict()` emits every field, including
concrete view policies; serialized specs cannot silently omit requirements.

## Per-view and action policy

Each view declares `role`, `required`, `enabled`, positive CHW `shape` (excluding
batch), `dtype`, boolean `mask`, `placeholder` and `content_rule`.
Required views cannot be disabled. Disabled optional views must be absent;
enabled optional views must be present with their declared shape/dtype/mask.
Masked-out views require zero placeholders; their digest includes batch, shape
and dtype byte width. `role_samples_sha256` is an explicitly synthetic digest
of role/sample/shape/dtype metadata. `sha256` checks digest syntax and relies
on independent backend replay for actual content verification.

Generate complete custom YAML without accidentally omitting default fields:

```python
import yaml
from core.contracts.anatomy_spec import AnatomySpec, ActionSpec, ViewSpec

spec = AnatomySpec(
    spec_id="mono-ar", required_camera_roles=("front",),
    optional_camera_roles=("wrist",), required_rng_consumers=(),
    views=(
        ViewSpec("front", shape=(3, 16, 20), dtype="uint8"),
        ViewSpec("wrist", required=False, enabled=True,
                 shape=(3, 8, 12), dtype="float16"),
    ),
    action=ActionSpec(
        representation="discrete_tokens", head="autoregressive",
        vocabulary_size=128, label_shift=1,
        tokenizer_id="cpu-fixture-v1", codebook_id="cpu-fixture-codebook",
        codebook_sha256="c" * 64, bos_token_id=0, eos_token_id=1, pad_token_id=2,
        token_layout=("batch", "horizon", "dimension"),
    ),
)
with open("spec.yaml", "w") as stream:
    yaml.safe_dump(spec.to_dict(), stream)
```

These tokenizer/codebook names and hash describe a **CPU fixture**, not a real
tokenizer artifact. Continuous flow uses `continuous/flow_matching` and forbids
discrete-only metadata. Discrete policies require positive `label_shift`,
tokenizer/codebook identifiers and non-placeholder SHA, distinct in-vocabulary
BOS/EOS/PAD IDs, and batch-first BHD or BDH layout.

## Self-contained spec release and external trust

Before either subprocess runs, capture seals `prep/specs/frozen/`; repeated
capture cannot change it. Capture and freeze copy the **complete** release
into the working and final Anatomy packages:

```text
analysis/anatomy_spec.yaml
analysis/spec_lock.json                  # package_dir: evidence/spec_freeze
evidence/spec_freeze/anatomy_spec.yaml
evidence/spec_freeze/artifact_manifest.json
```

The final lock uses only this fixed package-relative reference. The Anatomy
package can be moved and the original preparation deleted: spec verification
and fresh replay no longer depend on the old prep directory/absolute launch path.
The separately verified Stage 1 package and pinned assets remain prerequisites;
this is a self-contained **spec release**, not duplication of every Intake asset.

`launch_receipt.command` is normalized metadata, never an executable trust root.
It records `python3 -m core.stages.f2_anatomy.adapters.fake`, the fixed relative
`--spec-file evidence/spec_freeze/anatomy_spec.yaml`, and
`--output evidence/attempt-1/capture.json` (or `attempt-2`). The checker rejects
absolute/escaping/wrong output paths and noncanonical interpreter names.
Actual capture and independent replay always launch the current controlled
`sys.executable` with their real output paths, without resolving the recorded
name through `PATH`. Replay receipts use temporary-output basenames.
This permits Anatomy verification/replay in a different venv with the same
pinned validator/dependencies and separately valid Intake/asset prerequisites.
For portable Intake startup contracts, use a bare controlled `python3` name;
an explicit Stage 1 interpreter path remains subject to its existing strict
local-interpreter policy and is not made portable by this Anatomy change.

**The lock is not a trust root.** Retain `spec_manifest_sha256` from trusted
capture/freeze control-plane output before consuming the package (the controller
also emits it in release metadata). Supply it independently alongside trusted
outer Anatomy and Intake manifest digests. Do not read an expected digest from
an untrusted package's own lock/case/handoff/receipt. Missing trust, absolute
lock paths, inner tampering and a re-sealed inner release inconsistent with the
external anchor are rejected—even if the outer package is also re-sealed.

The external spec manifest SHA is checked against the nested manifest, case,
both captures, behavior contract, observation plan, step facts, pre-publish
checks and handoff. The spec binds its canonical policy hash, Intake manifest,
source commit and (for v2 Intake) independently verified report source reference.

```bash
python -m core.stages.f2_anatomy.cli capture \
  --intake-dir <sealed-intake> --intake-manifest-sha256 <intake-sha> \
  --prep-dir <prep> --spec <spec.yaml>
python -m core.stages.f2_anatomy.cli freeze --prep-dir <prep>
# Retain <spec-sha> from trusted capture/freeze output, not from the package lock.
python -m core.stages.f2_anatomy.cli replay \
  --package-dir <sealed-anatomy> --manifest-sha256 <anatomy-sha> \
  --spec-manifest-sha256 <spec-sha>
python -m validate.anatomy_check \
  --package-dir <sealed-anatomy> --manifest-sha256 <anatomy-sha> \
  --spec-manifest-sha256 <spec-sha> --expected-commit <source-commit> \
  --intake-dir <sealed-intake> --intake-manifest-sha256 <intake-sha>
```

Python callers use `CheckContext.parameters["spec_manifest_sha256"]` or
`validate_directory(expected_spec_manifest_sha256=...)`;
`replay_sealed_anatomy` requires `spec_manifest_sha256`.
For v2 Intake, capture automatically uses the accepted typed report spec and
pinned `source_ref`/analysis basis; conflicting caller specs are rejected.
For v1 Intake, the caller chooses/approves the initial spec or `pi05` preset.
`capture --spec` accepts strict bare policy YAML and the schema-2 wrappers
produced at `analysis/anatomy_spec.yaml` by Intake or Anatomy. CLI, adapter and
validator use the same loader; unknown fields/versions, duplicate YAML keys
and an incorrect supplied policy digest are rejected. Loading a wrapper
does not approve its metadata or replace sealed-package/source verification.

## Independent checks, not just matching attempts

Each route is independently checked against the frozen policy:

- Views: presence/absence, shape/dtype, slot/mask, placeholders and digest rule.
- Actions: exact semantics, logical BHD shape, finite values or vocabulary bounds,
  strict boolean masks and nonempty supervision.
- Discrete layout: declared coordinates and flattening against a canonical BHD
  snapshot; labels follow non-wrapping shift **within each sample**, PAD-filling
  tails. BOS occurs once at the start, EOS once, no specials/PAD in payload,
  only PAD after EOS, and masks agree with token/label semantics.

Matching wrong shapes, unshifted labels, layout errors or illegal special tokens
therefore fail. A fully coherent forged canonical snapshot remains untrusted:
fresh independent backend replay checks actual content, rather than treating
self-reported snapshots as truth.

Observation boundaries and negative injections are spec-derived: no enabled
views means no camera boundary; no declared step RNG consumers means no RNG
boundary. Optional switches are frozen, not inferred from observed absence.
The v1 behavior schema retains required assertion IDs; empty camera/RNG
assertions describe empty-state invariants, not active observation boundaries.
Initial weights, ranks, sample order, executed forwards, real optimizer updates,
finite endpoints, process isolation and independent replay cannot be disabled.

## Validator and evidence boundaries

New receipts use `validate.intake_check` / `validate.anatomy_check`,
**validator_version="2"**, with updated dependency closures and source SHA.
v1 task/behavior schemas still load; newly generated v1-task releases use v2
receipts. Historical `stages.verify.*` v1 sealed packages require their actual
pinned v1 runtime/dependencies. No real fixed historical v1 sealed fixture was
replayed in this change; compatibility with those sealed packages is **not**
claimed. Earlier `validate.*` version-1 review bundles likewise need their pinned
runtime or regeneration, not relabelled receipts. Never rewrite an old receipt
ID/version/SHA to bypass validation.

Offline formats remain `cpu_mock/orchestration_only`: they do not prove real
autoregressive VLA execution, actual tokenizer/codebook parity, image semantics
or numerical parity. `OpenPIJaxAdapter` remains fail-closed on missing
assets/runtime/GPU. Real verification requires a real adapter and hardware replay.
Only `backend: jax` invokes OpenPI JAX preflight; it uses a temporary pinned
source checkout restored from the verified Intake bundles, never a remote
`source.repo` interpreted as a local path. Other non-fake backends (including
`pytorch`) are blocked directly with `UNSUPPORTED_BACKEND_REPLAY_REQUIRED`,
without JAX diagnostics. Even a ready JAX preflight cannot substitute for
independent real-hardware replay and remains blocked.
Schema-only F2 examples retain their separate schema identity/version and explicit
blocker; they are not accepted v2 validator receipts.

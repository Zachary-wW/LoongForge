# v2 Intake: imported Skill analysis, soft slots and provenance

The v1 `F1TaskPack` and pinned-tree discovery/startup-probe responsibilities are
unchanged. `load_intake_task_pack` dispatches `schema_version: 2` to
`IntakeTaskPackV2`; v1 does not accept new fields or silently upgrade.

## Explicit handoff_only boundary

`model-adaptation` is a Markdown Agent playbook, not a Python analysis API.
This milestone imports a completed report. It does **not** start a Skill
harness, run an Agent, execute imported Python or download assets.
The receipt always says `boundary: handoff_only`, `status: imported`,
`harness_executed: false`.

The caller supplies a v2 task with two additional fields:

1. `analysis_handoff`: `SkillHandoffRequest`, including request/source/target
   identity, Skill repo/commit, playbook path and content SHA, offline Skill
   bundle SHA, report SHA, catalog SHA and read/write scopes.
2. `slot_catalog`: a versioned `internal` target-layout declaration.

`report_sha256` hashes the exact `model_analysis.json` bytes. `catalog_sha256`
and the receipt's request SHA hash UTF-8 canonical JSON (`ensure_ascii=False`,
sorted keys, compact separators, no NaN). Read scopes are normalized relative
source files/directories. Write scopes are confined to `analysis/`.

```bash
python -m core.stages.f1_intake.cli init --request <task-v2.yaml> --prep-dir <prep>
python -m core.stages.f1_intake.cli analyze \
  --prep-dir <prep> --handoff-dir <import-directory> --probe none
python -m core.stages.f1_intake.cli freeze --prep-dir <prep>
```

For a remote `source.repo` or a different local checkout, pass
`analyze --repo-root <local-checkout>`. The resolved path is retained in
unsealed `prep_state.json`; all freeze-time live checks use that same checkout.
An unavailable/changed checkout blocks freezing, rather than silently falling
back to offline evidence. Independent validation of an already sealed package
can still restore its source bundles without the original checkout.
Re-running analysis in the same preparation clears `work/evidence/` before
importing the new handoff, including when the replacement is invalid, so old
paper/Skill evidence cannot leak into a later release.

The import directory contains:

```text
model_analysis.json
evidence/skill.bundle
evidence/papers/<provided-text-evidence>  # only when cited
```

The Skill bundle must advertise the requested commit. The importer and
independent Checker restore it offline and check the playbook's bytes against
that commit. Repo names are request metadata, not a claim of online identity
authentication. Partial/failed reports and missing/unreproducible inputs block
publication instead of being labelled complete.

## Three engineering groups without hardcoded slot rules

The caller declares each of six extension points once, with a side (`data`,
`model`, `train`), registry path, allowed target files and reusable files:

`model_registry`, `data_and_transforms`, `config_map`, `trainer_builder`,
`eval_adapter`, `regression_runner`.

The catalog must cover all three sides. It does not impose KU's S0–S7 IDs,
segment ordering or a fixed assignment of extension points to groups.
Use the actual target framework layout, such as the paths declared by
`core/profiles/loongforge_embodied.yaml`.

Each arbitrary `slot_id` identifies an applicable mapping with pinned source
evidence and target files, or `not_applicable` with a nonempty reason and no
claimed source/target hits. Target files must be declared in the catalog.
`ScaffoldPlan.from_intake` independently accepts the sealed v2 Intake and maps
these slots to `ExtensionPointBinding` entries. It does not implement the
remaining Stage 3 code generator or promise an external KU consumer format.

When `anatomy_package_dir` is supplied, provide the trusted outer
`anatomy_manifest_sha256` and external `spec_manifest_sha256` as well.
`AnatomyChecker` verifies the package, Intake/source binding and spec release;
the plan takes the accepted binding digest and records
`anatomy_reference_status="verified"`. Missing/wrong trust or rejected packages
raise an error. A bare Anatomy SHA is syntax-checked but explicitly marked
`anatomy_reference_status="unverified_external"`; it is not an accepted Stage 2
result and must not be used as one. This distinction is part of the plan structure.

## Training parameters

Each parameter records `name`, `default_value`, `effective_value`, source
locations, a selected source and, when needed, an override source.

- Code evidence binds `commit + path + blob_sha256 + symbol + line_start/end`.
  Verification reads the pinned Git blob and resolves the exact AST node.
  This evidence adapter extracts literal Python assignments, not dynamic
  runtime expressions or values guessed from a startup probe.
- Paper evidence binds document SHA, ID/version, page, section, excerpt and
  a JSON value literal located in the excerpt. The supplied UTF-8 text must
  contain those locators (page marker `page: N`). This is a textual evidence
  adapter, not a PDF/OCR reader or scientific claim verifier.
- Conflicting defaults require a selected source and explicit resolution.
  A changed effective value requires independently verifiable override
  evidence. Numeric/boolean types are not treated as interchangeable.

## Typed, source-bound Anatomy policy

`ModelAnalysisReport.anatomy_spec` is a typed immutable `AnatomySpec` (spec
schema 2), not an arbitrary dictionary. It includes concrete view policies and
complete action semantics, rejecting unknown/missing fields.
`anatomy_spec_source` is mandatory and contains:

- `source_ref`: pinned code location (`commit/path/blob_sha256/symbol/lines`);
- `analysis_basis`: nonempty explanation tying analyzed code to the policy;
- optional `approved_by`: audit metadata, **not authenticated approval**.

The importer and independent IntakeChecker resolve this reference against the
restored pinned Git blob and permitted read scopes; a valid report hash alone
cannot bypass source validation. This is a reproducible analysis reference,
not a machine proof that an analyst's interpretation is semantically correct.
The controller still owns initial policy approval.

The receipt binds canonical `anatomy_spec_sha256` and
`anatomy_spec_source_sha256`. `analysis/anatomy_spec.yaml` is a schema-2 wrapper
containing `spec`, `source_ref` (including analysis basis/approval metadata),
`source_commit` and `request_id`. The checker reconstructs the wrapper and
receipt from the independently verified report. Stage 2 carries the same source
reference into its nested spec freeze and checks it against accepted Intake.
Re-sealing a changed wrapper/receipt does not supply new approval.

## Publication and independent checks

v2 requires the following manifested files in addition to the v1 layout:

```text
analysis/model_analysis.json
analysis/slot_hits.json
analysis/training_parameters.yaml
analysis/slot_catalog.yaml
analysis/skill_handoff_receipt.json
analysis/anatomy_spec.yaml
evidence/skill.bundle
```

Referenced paper evidence is also copied into the manifest. IntakeChecker
restores the source independently, rechecks report/Skill/catalog identity,
resolves source symbols and parameter values, and reconstructs every derived
artifact and receipt. Their hashes enter the pre-publish verdict.
Re-sealing a tampered derived file does not make it acceptable.

Composite validator digests include the new schema/import helpers and their
dependencies, along with `validate/base.py` and
`core/orchestrator/judger.py`. Canonical source-relative paths participate in
the digest, so moving a checkout does not change it.

The new Intake/Anatomy validator identities explicitly use version **2** and
include the shared spec schema in their dependency closure.
v1 task files and behavior schemas remain loadable; newly generated v1-task
packages use new v2 receipts. Historical `stages.verify.*` version-1 sealed
packages require their actual pinned old validator runtime and dependencies.
This change does not implement or claim historical sealed-package replay
compatibility: no real fixed historical v1 package was tested.
Old review-time `validate.*` version-1 receipts likewise cannot be relabelled.
Never rewrite receipt IDs/versions/SHAs to evade fail-closed validation.

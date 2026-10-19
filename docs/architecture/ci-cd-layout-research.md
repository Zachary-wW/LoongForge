# CI/CD directory layout research

## Scope

This note compares the current LoongForge layout with the official
NVIDIA/Megatron-LM repository and proposes a migration target for CI/CD-only
code.

## What Megatron-LM does

The official Megatron-LM repository separates automation by execution system:

| Area | Location | Role |
|---|---|---|
| GitLab pipeline entrypoint | `.gitlab-ci.yml` | Pipeline rules, variables, stages, and job orchestration |
| GitLab pipeline fragments | `.gitlab/stages/` | Stage-specific job definitions |
| GitLab helper scripts | `.gitlab/scripts/` | Build and CI helper programs |
| GitHub Actions | `.github/workflows/` | GitHub workflow definitions |
| GitHub helper scripts | `.github/scripts/` | GitHub bot and workflow helper programs |
| General project tooling | `scripts/` | Tools that are useful beyond one CI provider |
| Image definitions | `docker/` | CI and release image definitions |
| Test code | `tests/` | Tests and test utilities, including CI-facing test helpers |

The repository also assigns `.gitlab/`, `.github/`, `.gitlab-ci.yml`, and
`docker/` to the CI owners in `.github/CODEOWNERS`, which makes the ownership
boundary explicit.

Primary sources:

- [Megatron-LM repository](https://github.com/NVIDIA/Megatron-LM)
- [Megatron-LM `.gitlab-ci.yml`](https://github.com/NVIDIA/Megatron-LM/blob/main/.gitlab-ci.yml)
- [Megatron-LM `.gitlab/`](https://github.com/NVIDIA/Megatron-LM/tree/main/.gitlab)
- [Megatron-LM `.github/workflows/`](https://github.com/NVIDIA/Megatron-LM/tree/main/.github/workflows)
- [Megatron-LM `.github/scripts/`](https://github.com/NVIDIA/Megatron-LM/tree/main/.github/scripts)
- [Megatron-LM `.github/CODEOWNERS`](https://github.com/NVIDIA/Megatron-LM/blob/main/.github/CODEOWNERS)

## Comparison with LoongForge

LoongForge already follows part of this shape:

- `.github/workflows/` contains workflow definitions.
- `.github/actions/` contains local GitHub Actions.
- `.github/scripts/` contains workflow, release, and self-hosted-runner helpers.
- `docker/` contains image definitions.
- `tests/` contains CI helper tests.

The outlier is the root `ci/` directory. Its files are all automation-facing:

- repository quality checks: `check_doc_links.py`, `sensitive_scan.py`,
  `sensitive_rules.py`
- CI artifact handling: `redact_ci_artifact.py`
- release CI evidence: `release_evidence.py`

They are invoked by GitHub workflows, pre-commit, self-hosted runner scripts,
skills, and CI tests. They are not imported by the LoongForge runtime package.

## Candidate layouts

### Option A: `.github/scripts/` — recommended

```text
.github/
├── actions/
├── scripts/
│   ├── check_doc_links.py
│   ├── redact_ci_artifact.py
│   ├── release_evidence.py
│   ├── sensitive_rules.py
│   ├── sensitive_scan.py
│   └── self_runner/
└── workflows/
```

This keeps all GitHub automation under `.github/`, follows the Megatron-LM
pattern of placing CI helpers beside the workflow system, and avoids adding
another `ci/` directory. The existing `self_runner/` subdirectory still
separates host and GPU runner orchestration from the other helper programs.

### Option B: `.github/ci/`

```text
.github/
├── ci/
├── scripts/
├── actions/
└── workflows/
```

This is readable and shallow, but it keeps the `ci` directory name and creates
two competing homes for executable automation: `.github/ci/` and
`.github/scripts/`. It becomes less clear whether a new helper belongs in
`ci/` or `scripts/`.

### Option C: `scripts/ci/`

```text
scripts/
└── ci/
```

This matches projects that support several CI providers and want reusable
automation outside GitHub. It is less aligned with the current LoongForge
design, where the scripts depend on GitHub Actions conventions, protected
runner variables, GitHub checks, and the trusted workflow checkout.

## Recommendation

Use **Option A: `.github/scripts/`**.

The migration should move the five root `ci/` files directly into
`.github/scripts/` as one atomic change and update every caller in the same
change:

1. `.github/workflows/doc-links.yml` and `sensitive.yml`
2. `.pre-commit-config.yaml`
3. `.github/scripts/self_runner/` shell scripts
4. `tests/test_ci_artifacts.py`, `tests/test_doc_links.py`,
   `tests/test_release_regression_harness.py`, and related CI helper tests
5. `skills/sensitive-scan/SKILL.md`
6. `.github/workflows/README.md`, script docstrings, and
   `.github/scripts/sensitive_rules.py` self-reference allowlists

The migration should remove the root `ci/` directory completely after all
references are updated. Do not put these files under `.github/workflows/`;
that directory should remain declarative workflow YAML. Do not put them under
`third_party/`; they are maintained project automation.

## Design constraints

- CI-only code lives under `.github/`.
- Workflow YAML stays under `.github/workflows/`.
- Reusable CI helper programs live directly under `.github/scripts/`.
- Self-hosted-runner and release shell orchestration remains under
  `.github/scripts/self_runner/`.
- General user-facing data and checkpoint tools remain under `tools/`.
- Every move updates local invocation examples, tests, pre-commit hooks, and
  skill documentation in the same commit.
- The migration should preserve executable behavior and test import paths; it
  should not change the scanner rules or release semantics.

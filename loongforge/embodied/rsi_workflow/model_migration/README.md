# LoongForge Embodied Model Adaptation RSI Framework

A 6-stage deterministic orchestration and verification framework (`intake` → `anatomy` → `scaffold` → `adapt` → `verify` → `deliver`, plus outer-loop `rsi`) for adapting open-source embodied/VLA models into [`baidu-baige/LoongForge`](https://github.com/baidu-baige/LoongForge).

## Architecture & Implementation Status

| Package / Stage | Responsibility | Status |
| --- | --- | --- |
| [`core/`](core/README.md) | Strict Pydantic v2 contracts (`intake`, `anatomy`, `stage`, `verdict`, `handoff`), JSON Schema export, and immutable artifact sealing (`publish`/`verify`) | Implemented |
| `core/orchestrator/` | Stdlib + PyYAML DAG orchestration kernel (`graph`, `scheduler` with atomic checkpoint/resume/replan, `invalidation`, `isolation`, deterministic `judger`) | Implemented |
| `core/stages/f1_intake/` | Stage 1: Git commit/tree/recursive submodule lock, offline `git bundle` archive/restore, pinned-tree AST discovery, read-only asset triage (`WAITING_ASSET`/`blocked`), isolated startup probe, atomic freeze & CLI | Implemented (CPU + `openpi` dry-run verified) |
| `core/stages/f2_anatomy/` | Stage 2: Verified intake handoff, dual-subprocess single-step capture & replay, 4-tier weight snapshot diff, behavior contract & observation plan, `FakeStepAdapter` + `OpenPIJaxAdapter` skeleton | Implemented (offline CPU fake closed-loop; GPU smoke pending assets/hardware) |
| `validate/` | Independent checkers (`IntakeChecker`, `AnatomyChecker` implemented; `Construct`/`Parity`/`TrainResume`/`Eval` placeholders); orchestration lives in `core/stages/f5_verify/` | Stage 1 & 2 implemented |
| `core/stages/{f3_scaffold,f4_adapt,f6_deliver}/` & `core/rsi/` | Contracts and protocol placeholders targeting LoongForge's 6 extension points, 3 embodied Skills, and outer-loop RSI | Interfaces & placeholders ready |
| `core/profiles/loongforge_embodied.yaml` | Declarative 6-stage DAG, LoongForge 6 extension-point mapping, 3 core embodied Skills, and 5 stateless deterministic tools | Configured & validated |

## Quick Verification

```bash
python3 -m pytest -q
ruff check .
python3 -m compileall -q core validate tests
```

CLI entrypoints:

```bash
python3 -m core.stages.f1_intake.cli {init,analyze,freeze} --help
python3 -m validate.intake_check --help
python3 -m core.stages.f2_anatomy.cli {capture,freeze,replay} --help
python3 -m validate.anatomy_check --help
```

## Versioned analysis and model-specific behavior

- [Frozen AnatomySpec and action evidence](docs/anatomy-spec.md): pre-capture family policy, continuous/discrete action evidence, immutable bindings and independent dual replay.
- [v2 Intake Skill Handoff](docs/intake-v2-handoff.md): imported playbook reports (`handoff_only`, no harness invocation), configurable Data/Model/Train soft slots, training-parameter provenance and `ScaffoldPlan` mapping. v1 task packs remain supported.

Intake/Anatomy validators use version **2**. Anatomy spec releases are embedded
and relocatable; verification/replay requires caller-held `spec_manifest_sha256`.
v1 schemas remain loadable, but historical `stages.verify.*` v1 sealed packages
require their pinned old runtime (historical sealed replay compatibility is not
claimed). Scaffold records verified packages versus unverified external SHA refs.

Top-level packages are `core/`, `validate/`, `skills/`, `examples/`, `docs/` and `tests/`. There are no legacy forwarding directories. `core/__init__.py` is empty; importing the orchestration kernel requires only stdlib + PyYAML, not Pydantic.

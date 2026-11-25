# LoongForge Embodied Model Adaptation RSI Framework

A 6-stage deterministic orchestration and verification framework (`intake` → `anatomy` → `scaffold` → `adapt` → `verify` → `deliver`, plus outer-loop `rsi`) for adapting open-source embodied/VLA models into [`baidu-baige/LoongForge`](https://github.com/baidu-baige/LoongForge).

## Architecture & Implementation Status

| Package / Stage | Responsibility | Status |
| --- | --- | --- |
| [`core/`](core/README.md) | Strict Pydantic v2 contracts (`intake`, `anatomy`, `stage`, `verdict`, `handoff`), JSON Schema export, and immutable artifact sealing (`publish`/`verify`) | Implemented |
| `engine/` | Stdlib + PyYAML DAG orchestration kernel (`graph`, `scheduler` with atomic checkpoint/resume/replan, `invalidation`, `isolation`, deterministic `judger`) | Implemented |
| `stages/intake/` | Stage 1: Git commit/tree/recursive submodule lock, offline `git bundle` archive/restore, pinned-tree AST discovery, read-only asset triage (`WAITING_ASSET`/`blocked`), isolated startup probe, atomic freeze & CLI | Implemented (CPU + `openpi` dry-run verified) |
| `stages/anatomy/` | Stage 2: Verified intake handoff, dual-subprocess single-step capture & replay, 4-tier weight snapshot diff, behavior contract & observation plan, `FakeStepAdapter` + `OpenPIJaxAdapter` skeleton | Implemented (offline CPU fake closed-loop; GPU smoke pending assets/hardware) |
| `stages/verify/` | Independent checkers (`IntakeChecker`, `AnatomyChecker` implemented; `Construct`/`Parity`/`TrainResume`/`Eval` placeholders) + fail-closed `VerifyStageRunner` | Stage 1 & 2 implemented |
| `stages/{scaffold,adapt,deliver}/` & `rsi/` | Contracts and protocol placeholders targeting LoongForge's 6 extension points, 3 embodied Skills, and outer-loop RSI | Interfaces & placeholders ready |
| `profiles/loongforge_embodied.yaml` | Declarative 6-stage DAG, LoongForge 6 extension-point mapping, 3 core embodied Skills, and 5 stateless deterministic tools | Configured & validated |

## Quick Verification

```bash
python3 -m pytest -q
ruff check .
python3 -m compileall -q core engine stages rsi tests
```

CLI entrypoints:

```bash
python3 -m stages.intake.cli {init,analyze,freeze} --help
python3 -m stages.verify.intake_check --help
python3 -m stages.anatomy.cli {capture,freeze,replay} --help
python3 -m stages.verify.anatomy_check --help
```

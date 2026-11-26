# Core

Shared strict contracts and immutable artifact sealing primitives for the 6-stage embodied adaptation pipeline.

## Contents

- `contracts/`: strict Pydantic v2 models (`intake`, `anatomy`, `stage`, `verdict`, `handoff`, plus backwards-compatible `f1`/`f2` exports), YAML unique-key loader, and `PipelineProfile` validator.
- `contracts/schema.py`: JSON Schema exporter (`artifact-manifest`, `f1-task-pack`, `f2-behavior-contract`, `handoff-result`, `stage-spec`, `gate-result`, `gate-decision`).
- `contracts/errors.py`: structured parse and validation error types.
- `artifacts.py`: staging → atomic release `publish()` and full manifest/digest/symlink `verify()`.

Run CPU tests with:

```bash
python3 -m pytest -q
```

## Canonical package layout

```text
core/
  __init__.py         # empty: no transitive Pydantic imports for orchestrator
  artifacts.py
  contracts/         # v1 contracts + versioned v2 Intake handoff
  orchestrator/      # stdlib + PyYAML graph/scheduler/judger
  profiles/
  rsi/
  stages/
    base.py
    f1_intake/
    f2_anatomy/
    f3_scaffold/
    f4_adapt/
    f5_verify/       # orchestration only; concrete checkers in ../validate/
    f6_deliver/
```

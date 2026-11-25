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

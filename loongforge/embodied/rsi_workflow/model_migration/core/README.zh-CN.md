# Core

6 阶段具身模型适配流水线的公共严格契约与不可变包封存基础库。

## 模块说明

- `contracts/`：严格 Pydantic v2 模型（`intake` / `anatomy` / `stage` / `verdict` / `handoff`，保留 `f1`/`f2` 兼容导出）、防重复键 YAML 解析器与 `PipelineProfile` 校验器。
- `contracts/schema.py`：JSON Schema 导出（`artifact-manifest`、`f1-task-pack`、`f2-behavior-contract`、`handoff-result`、`stage-spec`、`gate-result`、`gate-decision`）。
- `contracts/errors.py`：结构化解析与校验错误类型。
- `artifacts.py`：`.staging` → 原子发布 `publish()` 与全量 manifest/摘要/symlink 拦截 `verify()`。

运行 CPU 测试：

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

# LoongForge 具身模型适配 RSI 框架

面向 [`baidu-baige/LoongForge`](https://github.com/baidu-baige/LoongForge) 的统一 6 阶段具身/VLA 模型适配编排与验证框架（`intake` → `anatomy` → `scaffold` → `adapt` → `verify` → `deliver`，外加跨任务 `rsi` 外循环）。

## 架构与当前落地状态

| 模块 / 阶段 | 职责 | 状态 |
| --- | --- | --- |
| [`core/`](core/README.zh-CN.md) | 严格 Pydantic v2 契约（`intake` / `anatomy` / `stage` / `verdict` / `handoff`）、JSON Schema 导出及不可变包原子封存（`publish`/`verify`） | 已完成 |
| `engine/` | 纯标准库 + PyYAML 轻量 DAG 编排内核（`graph`、原子 checkpoint 断点续跑/重规划 `scheduler`、传递失效 `invalidation`、`isolation`、确定性 `judger`） | 已完成 |
| `stages/intake/` | Stage 1：Git commit/tree/递归 submodule 锁、离线 `git bundle` 打包恢复、固定树 AST 入口发现、只读资产分流（`WAITING_ASSET`/`blocked`）、只读隔离 probe、原子封存与 CLI | 已完成（CPU 测试 + 真实 `openpi` 演练通过） |
| `stages/anatomy/` | Stage 2：强制复验 Stage 1 封存包、双独立子进程单步采集与重放、4 级权重快照对比、行为契约与观测计划、`FakeStepAdapter` 与 `OpenPIJaxAdapter` 骨架 | 离线闭环已完成（真机 GPU 待资产就绪） |
| `stages/verify/` | 独立校验器（`IntakeChecker` / `AnatomyChecker` 已实装；`Construct`/`Parity`/`TrainResume`/`Eval` 占位）与 fail-closed `VerifyStageRunner` | Stage 1 & 2 已实装 |
| `stages/{scaffold,adapt,deliver}/` 与 `rsi/` | 对齐 LoongForge 6 大扩展点、3 个具身 Skill 与外循环 RSI 的契约与接口占位 | 接口占位就绪 |
| `profiles/loongforge_embodied.yaml` | 声明式 6 阶段默认 DAG、LoongForge 6 大扩展点映射、3 个具身 Skill 与 5 个无状态确定性工具清单 | 已配置并校验 |

## 快速验证

```bash
python3 -m pytest -q
ruff check .
python3 -m compileall -q core engine stages rsi tests
```

CLI 入口：

```bash
python3 -m stages.intake.cli {init,analyze,freeze} --help
python3 -m stages.verify.intake_check --help
python3 -m stages.anatomy.cli {capture,freeze,replay} --help
python3 -m stages.verify.anatomy_check --help
```

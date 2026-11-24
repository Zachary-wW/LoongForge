# 模型迁移

将开源模型迁移到 LoongForge 的完整工作区：在保证逻辑一致、精度对齐的前提下，把开源模型的格式、结构与 LoongForge 的模型约定对齐，使其可以在 LoongForge 框架内直接训练、评估与发布。

迁移工作以 RSI（Robust Self-Improvement / Recurrent Self-Improvement）智能体流程为核心组织：由 `core` 中的调度与执行逻辑驱动，复用 `skills` 中的技能完成具体操作，产出物通过 `validate` 的检查后归档为 `examples` 中的可复现任务，全部过程记录在 `docs` 中。

## 目录结构

| 目录 | 说明 |
| --- | --- |
| [`core/`](core/README.zh-CN.md) | 核心代码。完成迁移工作的主要部分，包含沙盒、模板、调度控制等涉及 RSI 的主要实现 |
| [`docs/`](docs/README.zh-CN.md) | 说明文档。包含各功能的使用说明和样例 |
| [`examples/`](examples/README.zh-CN.md) | 参考任务。已经跑通、可作为模板复用的迁移任务 |
| [`skills/`](skills/README.zh-CN.md) | 技能集。迁移过程中主要使用的 skill 整理于此，由 `core` 调用 |
| [`validate/`](validate/README.zh-CN.md) | 校验模块。检查最终成果的准确性，包括逻辑一致性、精度对齐等，确保成果准确可用 |

## 使用流程

1. 在 `examples/` 中选择或新建一个迁移任务作为目标。
2. `core/` 依据任务描述，调度 `skills/` 中的技能完成模型迁移与格式转换。
3. 迁移产物提交到 `validate/` 进行逻辑与精度校验。
4. 通过校验后，将流程沉淀为 `examples/` 中的可复现任务，并把用法写入 `docs/`。

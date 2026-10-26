#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Render the reviewed Markdown inventory from inventory.csv."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

INTRO = """# LoongForge 中源自 Megatron-LM 的代码盘点

> 审计日期：2026-10-26
> LoongForge 基线：`a8268b7973e108fb61676a2cb967961dccb9851f`（`master`）
> 对比版本：NVIDIA Megatron-LM `core_v0.10.0`、`v0.12.0`、`v0.13.0`、`v0.14.0`、`v0.15.0`、`v0.19.2`
> 机器清单：[inventory.csv](inventory.csv)
> 相似度明细：[similarity_results.csv](similarity_results.csv)

## 结论

这轮共整理出 **__TOTAL__ 个需要纳入 review 的文件**：

- **__CONFIRMED__ 个 A 类：已确认从 Megatron-LM 派生。** 其中 47 个文件头明确写着 `Modified from Megatron-LM`，另 4 个由此前人工映射、import/vendor 审计确认。
- **__B_COUNT__ 个 B 类：模型侧强候选。** 文件名、构造方式或核心方法明显沿用 Megatron。
- **__C_COUNT__ 个 C 类：扩展 review 候选。** 包括数据构建、tokenizer、Diffusion、VLM adapter 等；代码可能经过大量改写，但目标 Megatron API 变化后仍需逐文件确认。

此前文档只覆盖了训练入口和几个共享基类，遗漏最多的部分是：

- 各模型的 `*_model.py`、`*_attention.py`、`*_transformer_layer.py`、`*_layer_spec.py`。
- `model_chunk_schedule_plan.py`、`fine_grained_callables.py`。
- MLA、local attention/norm、DSA/CSA vendor 代码。
- VLM 的 ViT layer spec、ViT attention/layer 和 RoPE 实现。

与旧清单相比：

- 旧表可精确对应的文件只有 10 个。
- 新清单在 A 类中另有 __NEW_CONFIRMED__ 个明确来源文件，主要集中在模型的 layer spec、experts/router、RoPE、VLM、model chunk schedule 和 fine-grained callables。
- 旧表里的 `xxx_attention.py`、`xxx_model.py` 被展开为逐模型的具体路径；其中无法确认整文件复制的放入 B 类，而不是直接当成已知上移代码。

另一个重要发现是：**不能把全部文件都简单标成基于 v0.15.0。** 结构相似度显示，代码来源混合了多个旧版本：

- `minimax/attention.py` 最接近 v0.10。
- `qwen3_next/qwen3_next_model.py` 最接近 v0.12。
- `validators.py` 最接近 v0.12。
- `qwen3_next/gated_attention.py` 最接近 v0.13。
- `model_chunk_schedule_plan.py`、`fine_grained_callables.py` 与 v0.14/v0.15 高度接近。
- checkpointing、GPTModel、TransformerBlock 和大部分训练入口更接近 v0.15。

因此后续升级不能只做一次 v0.15 → v0.19.2 diff，逐个文件确认其实际来源版本、LoongForge 增量以及 upstream 是否已有等价实现。

## 审计方法

1. 扫描 LoongForge `loongforge/` 下 680 个可解析的 Python 文件。
2. 将文件转换为去注释、去格式化差异的 AST 归一化文本，分别使用 6 行片段和 12-token 片段进行匹配。
3. 分别与 Megatron v0.10、v0.12、v0.13、v0.14、v0.15、v0.19.2 比较。
4. 同时检查文件头声明、此前 `feature_registry.json` 的 import/vendor 分类和人工映射。
5. 将“明确声明或人工确认”与“仅结构相似”的模型侧候选分开，避免把通用模型实现误判为复制。

相似度是定位工具，不是法律或架构意义上的来源证明。A 类可直接进入升级 review；B/C 类必须先人工确认是整文件复制、片段复制，还是仅依赖 Megatron API。

使用的 NVIDIA tag commit：

| Tag | Commit |
|---|---|
| `core_v0.10.0` | `7ee599a25ca15f042c1087176b73ebddb0566d02` |
| `core_v0.12.0` | `d580efc68a9f0dbf1945f834f6f6200cd01d3343` |
| `core_v0.13.0` | `c550cf6c41c31cd3ec72e05c25ea0c979f2b6631` |
| `core_v0.14.0` | `23e00ed0963c35382dfe8a5a94fb3cda4d21e133` |
| `core_v0.15.0` | `0d7e02bd26d3eb347ad8d1bef625002babfa6df7` |
| `core_v0.19.2` | `4b4acac9a1d28ea6829c8d4f566d75698a21249d` |

## 高置信度版本证据

下表使用 AST 归一化后的行序列相似度。数值只用于判断代码更接近哪个历史快照。

| LoongForge 文件 | 最接近版本 | 归一化相似度 | 主要信号 |
|---|---:|---:|---|
| `train/checkpointing.py` | v0.15 | 0.900 | 文件头、helper 名称和 checkpoint 流程一致 |
| `models/foundation/base/base_gpt_model.py` | v0.15 | 0.706 | GPTModel 全套主方法和流程一致 |
| `models/omni_models/model_chunk_schedule_plan.py` | v0.14 / v0.15 | 0.834 | 同文件职责和大量连续结构一致 |
| `models/omni_models/fine_grained_callables.py` | v0.14 / v0.15 | 0.628 | callable 分派与模型层包装流程一致 |
| `train/pretrain/pretrain_llm.py` | v0.15 | 0.490 | `pretrain_gpt.py` batch/loss 流程一致 |
| `train/training_utils.py` | v0.14 / v0.15 | 0.468 | training loop 主体一致 |
| `models/foundation/qwen3_next/gated_attention.py` | v0.13 | 0.381 | Attention 主结构接近 v0.13 |
| `models/encoder/vision_transformer_block.py` | v0.15 | 0.378 | TransformerBlock 主体一致 |
| `models/foundation/ernie4_5_vl/ernie_experts.py` | v0.15 | 0.355 | MLP/expert 构造与 forward 一致 |
| `models/foundation/minimax/attention.py` | v0.10 | 0.355 | Attention 主结构与早期版本更接近 |
| `models/foundation/language_transformer_block.py` | v0.15 | 0.345 | TransformerBlock 主体一致 |
| `train/sft/sft_llm.py` | v0.15 | 0.340 | SFT 入口沿用 `pretrain_gpt.py` |
| `train/validators.py` | v0.12 | 0.299 | 参数校验与 v0.12 最接近，但已有大量 LoongForge 增量 |
| `train/initialize.py` | v0.15 | 0.271 | 初始化主流程一致，LoongForge 增量为运行期扩展 |
| `models/common/.../multi_latent_attention.py` | v0.15 | 0.196 | MLA 主结构和接口沿用 upstream |
| `train/get_loss_func.py` | v0.15 | 0.207 | loss 组织方式来自 GPT/VLM 入口 |

## 升级优先级

### P0：运行时和共享基类

这些文件一旦 upstream API 变化，影响所有 LLM/VLM 训练入口。

- 训练运行时：checkpointing、training_utils、validators、initialize、pretrain/SFT/loss。
- 模型共享边界：BaseGPTModel、Language/Vision TransformerBlock、model chunk schedule、fine-grained callables、MLA、local attention/norm。

重点检查：

- Moved imports、函数参数和返回类型。
- `ProcessGroupCollection`、TP/CP/DP 显式传组。
- `config_overrides`、VPP dataloader、model chunk schedule。
- `TransformerModelChunkSchedulePlan` output processor。
- `tie_word_embeddings_state_dict` 的 TP/DP-CP group。
- checkpoint helper、async save、PEFT/LoRA state dict。

### P1：模型侧代码

- 每个模型的 `*_model.py`：优先比较 GPTModel 的 preprocess/postprocess、shared embedding/output、VPP 和 schedule 行为。
- 每个模型的 `*_attention.py`：优先比较 attention/MLA 的构造签名、inference context、RoPE、packed sequence 和 CP。
- 每个模型的 `*_transformer_layer.py`：重点检查 submodule 初始化、recompute、CUDA graph 和 residual/dropout 路径。
- 每个模型的 `*_layer_spec.py`：重点检查 `ModuleSpec`、TE/local linear、activation 和 MoE backend API。
- `experts.py` / `router.py`：重点检查 grouped GEMM、TP/EP group 和 token dispatcher。

### P1：VLM/ViT

ViT 代码的风险不只在 `vision_transformer_block.py`。多个模型的 vision model 和 layer spec 也可能从旧 Megatron/NVLM 示例演化而来，但目前没有来源声明，因此放入 B 类候选。

### P2：DSA/CSA 和 vendored 算子

这些文件来自较新的实验分支，未必对应 v0.15，但仍需跟随目标 Megatron-LM 的模块布局和算子签名升级。不能仅因为文件较新就排除在同步清单外。

## A 类：已确认的 Megatron 派生文件

"""

FOOTER = """B/C 类文件没有明确的 `Modified from Megatron-LM` 头，也没有足够证据认定整文件复制。它们进入候选清单的原因是：

- 文件名和层职责对应 Megatron 的 Model / Attention / TransformerLayer / LayerSpec。
- 同一模型族中部分相邻文件已经明确声明来自 Megatron。
- 代码构造方式依赖 Megatron 的 ModuleSpec、TransformerConfig、并行组和模型层接口。
- 如果 Megatron 的模型构造 API 改变，这些文件仍然需要同步，只是不能直接按 upstream diff 替换。

Review 时应逐文件回答：

1. 是否复制了 upstream 的方法主体，还是只使用了公开 API？
2. LoongForge 增量是否能移到 wrapper/plugin，避免继续维护整份拷贝？
3. 目标 v0.19.2 是否已有等价能力可以删除本地实现？

## 暂不计入 A 类的相关文件

以下文件也需要跟踪，但不属于本轮“确认来自 Megatron 的实现代码”：

- `loongforge/embodied/optimizer/optimizer.py`：存在 vendored 迹象，但尚不能证明来源为 Megatron-LM。
- `loongforge/models/common/peft/*`：可能是 NVIDIA PEFT/NeMo 或社区实现，当前 v0.15 没有一一对应的 Megatron path。
- 大部分 `loongforge/embodied/*`：主要来自 NVIDIA Cosmos、GR00T、FastWAM、VJEPA2 等模型工程，不属于 Megatron-LM Core 文件升级。
- `Megatron-Energon` task encoder/flavor：需要跟随 Energon 版本，不能只用 Megatron-LM tag 判断。
- `loongforge/train.py`、`trainer_builder.py`：属于 LoongForge 自有轻量调度层，没有与 Megatron 文件一一对应的整文件拷贝；由 CLI/model-build 集成测试覆盖。
- `loongforge/data/sft_dataset.py`、`sft_data_collator.py`、`sft_format_utils.py`、`sft_supervised_utils.py`：主要跟随 HuggingFace Dataset / LLaMA-Factory，不属于 Megatron-LM 文件同步。
- `patches/TransformerEngine_v2.9/*`：属于 TE patch，不是 Megatron Python 源码拷贝。
- 只通过 import 使用 Megatron、没有复制实现主体的 145 个文件：继续由 API 调用扫描维护，不放进本清单。

## 建议的 Review 顺序

1. 先 review A 类中的 P0 训练入口和共享基类。
2. 再按模型族 review A 类的 model/attention/layer/expert/layer-spec。
3. 对 B 类逐文件判断“整文件复制 / 片段复制 / 仅 API 依赖”。
4. 对确认复制且 upstream 已有等价实现的文件，优先改为 import 或薄 wrapper。
5. 对必须保留的 LoongForge 分叉，在文件中明确 header、源 upstream SHA/version、分叉原因和同步 owner。

## 证据文件

- [inventory.csv](inventory.csv)：__TOTAL__ 个文件和置信度、推断 upstream 路径、升级关注点。
- [inventory.json](inventory.json)：同一清单的 JSON 版本。
- [similarity_results.csv](similarity_results.csv)：680 个 LoongForge Python 文件的行级结构相似度。
- [token_similarity_results.csv](token_similarity_results.csv)：token 级结构相似度，用于发现经过大规模改写的模型和数据代码。
- [ISSUE_MAP.md](ISSUE_MAP.md)：现有 Issue 与 Copy ID 的映射。
- [scan_similarity.py](../../../tools/migration/megatron_copy_audit/scan_similarity.py)：相似度扫描脚本。
- [scan_token_similarity.py](../../../tools/migration/megatron_copy_audit/scan_token_similarity.py)：token 级补充扫描脚本。
- [build_inventory.py](../../../tools/migration/megatron_copy_audit/build_inventory.py)：清单生成脚本。
"""


def table_for(rows: list[dict]) -> str:
    lines = [
        "| ID | 分组 | LoongForge 文件 | 推断 Megatron 来源 | 版本线索 | 升级关注点 | 证据 |",
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows:
        values = [
            row["id"],
            row["group"],
            f"`{row['loongforge_path']}`",
            row["probable_megatron_source"],
            row["best_old_version"],
            row["upgrade_hint"],
            row["evidence"],
        ]
        values = [value.replace("|", "\\|").replace("\n", "<br>") for value in values]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.inventory.open()))
    confirmed = [row for row in rows if row["confidence"] == "A-已确认"]
    candidates_b = [row for row in rows if row["confidence"] == "B-派生候选"]
    candidates_c = [row for row in rows if row["confidence"] == "C-扩展候选"]
    candidates = candidates_b + candidates_c

    by_group = defaultdict(list)
    for row in confirmed:
        by_group[row["group"]].append(row)

    intro = (
        INTRO.replace("__TOTAL__", str(len(rows)))
        .replace("__CONFIRMED__", str(len(confirmed)))
        .replace("__B_COUNT__", str(len(candidates_b)))
        .replace("__C_COUNT__", str(len(candidates_c)))
        .replace("__NEW_CONFIRMED__", str(max(0, len(confirmed) - 10)))
    )
    sections = [intro]
    for group in [
        "训练入口/运行时",
        "数据/Tokenizer",
        "共享模型/算子",
        "语言基础模型",
        "视觉/多模态模型",
        "Diffusion",
        "其他",
    ]:
        group_rows = by_group.get(group, [])
        if not group_rows:
            continue
        sections.append(f"### {group}（{len(group_rows)}）\n\n{table_for(group_rows)}\n\n")

    sections.append("## B 类：模型侧派生候选\n\n")
    sections.append(table_for(candidates))
    sections.append("\n\n")
    sections.append(FOOTER.replace("__TOTAL__", str(len(rows))))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(sections), encoding="utf-8")
    print(f"WROTE {args.output} ({len(confirmed)} A + {len(candidates_b)} B + {len(candidates_c)} C)")


if __name__ == "__main__":
    main()

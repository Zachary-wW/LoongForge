#!/usr/bin/env python3
# Copyright 2026 The LoongForge Authors.
# SPDX-License-Identifier: Apache-2.0
"""Build a provenance inventory for LoongForge code derived from Megatron-LM."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

RATIO_OVERRIDES = {
    "loongforge/train/checkpointing.py": ("v0.15", 0.900),
    "loongforge/train/training_utils.py": ("v0.14 / v0.15", 0.468),
    "loongforge/train/validators.py": ("v0.12", 0.299),
    "loongforge/train/initialize.py": ("v0.15", 0.271),
    "loongforge/train/pretrain/pretrain_llm.py": ("v0.15", 0.490),
    "loongforge/train/pretrain/pretrain_vlm.py": ("v0.14 / v0.15", 0.060),
    "loongforge/train/sft/sft_llm.py": ("v0.15", 0.340),
    "loongforge/train/get_loss_func.py": ("v0.15", 0.207),
    "loongforge/models/foundation/base/base_gpt_model.py": ("v0.15", 0.706),
    "loongforge/models/foundation/language_transformer_block.py": ("v0.15", 0.345),
    "loongforge/models/encoder/vision_transformer_block.py": ("v0.15", 0.378),
    "loongforge/models/common/experimental_attention_variant/multi_latent_attention.py": (
        "v0.15",
        0.196,
    ),
    "loongforge/models/foundation/qwen3_next/gated_attention.py": ("v0.13", 0.381),
    "loongforge/models/foundation/minimax/attention.py": ("v0.10", 0.355),
    "loongforge/models/foundation/ernie4_5_vl/ernie_experts.py": ("v0.15", 0.355),
    "loongforge/models/foundation/ernie4_5_vl/ernie_router.py": ("v0.14 / v0.15", 0.260),
    "loongforge/models/foundation/ernie4_5_vl/ernie_transformer_layer.py": ("v0.15", 0.317),
    "loongforge/models/omni_models/model_chunk_schedule_plan.py": (
        "v0.14 / v0.15",
        0.834,
    ),
    "loongforge/models/omni_models/fine_grained_callables.py": (
        "v0.14 / v0.15",
        0.628,
    ),
}
EXTRA_CANDIDATES = {
    "loongforge/data/blended_hf_dataset_builder.py",
    "loongforge/data/blended_hf_dataset_config.py",
    "loongforge/data/encoder_strided_sampler.py",
    "loongforge/data/multimodal/base/task_encoder.py",
    "loongforge/data/multimodal/dataloader_provider.py",
    "loongforge/data/multimodal/internvl/internvl_preprocess.py",
    "loongforge/data/multimodal/internvl/internvl_task_encoder.py",
    "loongforge/models/common/cuda_graph_base.py",
    "loongforge/models/common/experimental_attention_variant/sparse_mla_bwd.py",
    "loongforge/models/diffusion/qwen_image/qwen_image_layer.py",
    "loongforge/models/diffusion/qwen_image/qwen_image_layer_spec.py",
    "loongforge/models/diffusion/qwen_image/qwen_image_model.py",
    "loongforge/models/diffusion/qwen_image/qwen_image_modules.py",
    "loongforge/models/diffusion/qwen_image/qwen_image_provider.py",
    "loongforge/models/diffusion/qwen_image/qwen_image_rope.py",
    "loongforge/models/diffusion/qwen_image/qwen_image_transformer_block.py",
    "loongforge/models/diffusion/wan/communications.py",
    "loongforge/models/diffusion/wan/wan_attention.py",
    "loongforge/models/diffusion/wan/wan_layer.py",
    "loongforge/models/diffusion/wan/wan_layer_spec.py",
    "loongforge/models/diffusion/wan/wan_model.py",
    "loongforge/models/diffusion/wan/wan_provider.py",
    "loongforge/models/diffusion/wan/wan_transformer_block.py",
    "loongforge/models/diffusion/wan/wan_ulysses.py",
    "loongforge/models/diffusion/wan/wan_utils.py",
    "loongforge/models/dispatch.py",
    "loongforge/models/encoder/ernie4_5_vl_vision_models/ernie_encoder_spec.py",
    "loongforge/models/encoder/minicpm_v_4_6_vision_models/vision_model.py",
    "loongforge/models/encoder/moon_vision_models/patch_merger_adapter.py",
    "loongforge/models/encoder/qwen2_vl_vision_models/adapter.py",
    "loongforge/models/foundation/ernie4_5_vl/ernie_moe_layer.py",
    "loongforge/models/foundation/mimo/mimo_multi_token_prediction.py",
    "loongforge/models/foundation/qwen3_next/gated_deltanet.py",
    "loongforge/models/utils.py",
    "loongforge/tokenizer/tokenizer.py",
    "loongforge/train/megatron_trainer.py",
    "loongforge/train/parser.py",
    "loongforge/train/sft/sft_vlm.py",
    "loongforge/train/sft/utils.py",
}


def declared_source_files(root: Path) -> set[str]:
    files = set()
    for path in root.glob("loongforge/**/*.py"):
        head = "\n".join(path.read_text(encoding="utf-8", errors="ignore").splitlines()[:12])
        if "Modified from Megatron-LM" in head:
            files.add(path.relative_to(root).as_posix())
    files.update(
        {
            "loongforge/train/validators.py",
            "loongforge/models/common/experimental_attention_variant/dsa_fused_kernels.py",
            "loongforge/models/common/experimental_attention_variant/dsa_fused_utils.py",
            "loongforge/models/foundation/deepseek_v4/deepseek_v4_csa.py",
        }
    )
    return files


def candidate_model_files(root: Path, confirmed: set[str]) -> set[str]:
    patterns = [
        "loongforge/models/foundation/**/*_model.py",
        "loongforge/models/foundation/**/*_attention.py",
        "loongforge/models/foundation/**/*_layer_spec.py",
        "loongforge/models/foundation/**/*_transformer_layer.py",
        "loongforge/models/encoder/**/*_model.py",
        "loongforge/models/encoder/**/*_attention.py",
        "loongforge/models/encoder/**/*_layer_spec.py",
        "loongforge/models/encoder/**/*_transformer_layer.py",
    ]
    candidates = set()
    for pattern in patterns:
        candidates.update(path.relative_to(root).as_posix() for path in root.glob(pattern))
    return (candidates | EXTRA_CANDIDATES) - confirmed


def group_for(path: str) -> str:
    if path.startswith("loongforge/train/") or path.startswith("loongforge/data/dp_balance/"):
        return "训练入口/运行时"
    if path.startswith("loongforge/data/") or path.startswith("loongforge/tokenizer/"):
        return "数据/Tokenizer"
    if path.startswith("loongforge/models/omni_models/") or path.startswith("loongforge/models/common/"):
        return "共享模型/算子"
    if path.startswith("loongforge/models/encoder/"):
        return "视觉/多模态模型"
    if path.startswith("loongforge/models/foundation/"):
        return "语言基础模型"
    if path.startswith("loongforge/models/diffusion/"):
        return "Diffusion"
    return "其他"


def source_for(path: str) -> str:
    name = Path(path).name
    exact = {
        "loongforge/train/checkpointing.py": "megatron/training/checkpointing.py",
        "loongforge/train/training_utils.py": "megatron/training/training.py",
        "loongforge/train/validators.py": "megatron/training/arguments.py",
        "loongforge/train/initialize.py": "megatron/training/initialize.py",
        "loongforge/train/pretrain/pretrain_llm.py": "pretrain_gpt.py",
        "loongforge/train/pretrain/pretrain_vlm.py": "pretrain_vlm.py",
        "loongforge/train/sft/sft_llm.py": "pretrain_gpt.py",
        "loongforge/train/get_loss_func.py": "pretrain_gpt.py",
        "loongforge/data/dp_balance/rerun_iterator.py": "megatron/core/rerun_state_machine.py::RerunDataIterator",
        "loongforge/models/foundation/base/base_gpt_model.py": "megatron/core/models/gpt/gpt_model.py",
        "loongforge/models/foundation/language_transformer_block.py": "megatron/core/transformer/transformer_block.py",
        "loongforge/models/encoder/vision_transformer_block.py": "megatron/core/transformer/transformer_block.py",
        "loongforge/models/omni_models/model_chunk_schedule_plan.py": "megatron/core/models/common/model_chunk_schedule_plan.py",
        "loongforge/models/omni_models/fine_grained_callables.py": "megatron/core/models/gpt/fine_grained_callables.py",
        "loongforge/models/common/experimental_attention_variant/multi_latent_attention.py": "megatron/core/transformer/multi_latent_attention.py",
        "loongforge/models/common/local_layers/local_attention.py": "megatron/core/transformer/attention.py",
        "loongforge/models/common/local_layers/local_norm.py": "megatron/core/transformer/torch_norm.py + fused_layer_norm.py",
        "loongforge/models/common/experimental_attention_variant/dsa_fused.py": "Megatron DSA / fused sparse attention",
        "loongforge/models/common/experimental_attention_variant/dsa_fused_kernels.py": "Megatron DSA kernels",
        "loongforge/models/common/experimental_attention_variant/dsa_fused_utils.py": "Megatron DSA utilities",
        "loongforge/models/foundation/deepseek_v4/deepseek_v4_csa.py": "Megatron CSA/DSA experimental attention",
        "loongforge/models/foundation/llm_model_provider.py": "pretrain_gpt.py / model provider",
        "loongforge/models/encoder/qwen3_vl_vision_models/rope_utils.py": "megatron/core/models/common/embeddings/rope_utils.py",
        "loongforge/models/foundation/deepseek_v4/deepseek_v4_rope.py": "megatron/core/models/common/embeddings/rope_utils.py",
        "loongforge/models/foundation/ernie4_5_vl/ernie_experts.py": "megatron/core/transformer/moe/experts.py + mlp.py",
        "loongforge/models/foundation/ernie4_5_vl/ernie_router.py": "megatron/core/transformer/moe/router.py",
        "loongforge/models/foundation/ernie4_5_vl/ernie_transformer_layer.py": "megatron/core/transformer/transformer_layer.py",
        "loongforge/models/foundation/qwen3_next/gated_attention.py": "megatron/core/transformer/attention.py",
        "loongforge/models/foundation/minimax/attention.py": "megatron/core/transformer/attention.py",
        "loongforge/utils/utils.py": "megatron/training/utils.py + megatron/core/utils.py",
        "loongforge/data/multimodal/internvl/internvl_preprocess.py": "examples/multimodal/image_processing.py",
        "loongforge/data/multimodal/dataloader_provider.py": "examples/multimodal/dataloader_provider.py",
        "loongforge/data/multimodal/base/task_encoder.py": "examples/multimodal/dataset_helpers.py",
        "loongforge/data/blended_hf_dataset_builder.py": "megatron/core/datasets/blended_megatron_dataset_builder.py",
        "loongforge/data/blended_hf_dataset_config.py": "megatron/core/datasets/blended_megatron_dataset_config.py",
        "loongforge/data/encoder_strided_sampler.py": "megatron/legacy/data/data_samplers.py",
        "loongforge/train/megatron_trainer.py": "megatron/training/training.py",
        "loongforge/train/parser.py": "megatron/training/arguments.py",
        "loongforge/train/sft/sft_vlm.py": "pretrain_vlm.py",
        "loongforge/train/sft/utils.py": "pretrain_gpt.py + Megatron dataloader utilities",
        "loongforge/tokenizer/tokenizer.py": "megatron/training/tokenizer/tokenizer.py",
        "loongforge/models/utils.py": "megatron/core/transformer/spec_utils.py",
        "loongforge/models/dispatch.py": "Megatron multimodal model dispatch",
        "loongforge/models/common/cuda_graph_base.py": "megatron/core/full_cuda_graph.py",
        "loongforge/models/common/experimental_attention_variant/sparse_mla_bwd.py": "megatron/core/transformer/experimental_attention_variant/ops/tilelang_sparse_mla_bwd.py",
        "loongforge/models/foundation/mimo/mimo_multi_token_prediction.py": "megatron/core/transformer/multi_token_prediction.py",
        "loongforge/models/foundation/qwen3_next/gated_deltanet.py": "megatron/core/transformer/attention.py + SSM layers",
        "loongforge/models/foundation/ernie4_5_vl/ernie_moe_layer.py": "megatron/core/transformer/moe/moe_layer.py",
        "loongforge/models/encoder/ernie4_5_vl_vision_models/ernie_encoder_spec.py": "megatron/core/models/vision/vit_layer_specs.py",
        "loongforge/models/encoder/qwen2_vl_vision_models/adapter.py": "Megatron multimodal vision adapter examples",
        "loongforge/models/encoder/moon_vision_models/patch_merger_adapter.py": "Megatron multimodal vision adapter examples",
        "loongforge/models/diffusion/wan/wan_transformer_block.py": "megatron/core/transformer/transformer_block.py",
        "loongforge/models/diffusion/qwen_image/qwen_image_transformer_block.py": "megatron/core/transformer/transformer_block.py",
        "loongforge/models/diffusion/wan/wan_attention.py": "megatron/core/transformer/attention.py",
        "loongforge/models/diffusion/wan/wan_layer.py": "megatron/core/transformer/transformer_layer.py + mlp.py",
        "loongforge/models/diffusion/wan/wan_layer_spec.py": "GPT/ViT layer specs",
        "loongforge/models/diffusion/wan/wan_model.py": "megatron/core/models/gpt/gpt_model.py",
        "loongforge/models/diffusion/wan/wan_provider.py": "pretrain/model provider",
        "loongforge/models/diffusion/wan/wan_ulysses.py": "Megatron context-parallel attention utilities",
        "loongforge/models/diffusion/wan/wan_utils.py": "megatron/training + core utils",
        "loongforge/models/diffusion/wan/communications.py": "Megatron collective communication wrappers",
        "loongforge/models/diffusion/qwen_image/qwen_image_layer.py": "megatron/core/transformer/transformer_layer.py",
        "loongforge/models/diffusion/qwen_image/qwen_image_layer_spec.py": "GPT/ViT layer specs",
        "loongforge/models/diffusion/qwen_image/qwen_image_model.py": "megatron/core/models/gpt/gpt_model.py",
        "loongforge/models/diffusion/qwen_image/qwen_image_modules.py": "Megatron transformer modules",
        "loongforge/models/diffusion/qwen_image/qwen_image_provider.py": "pretrain/model provider",
        "loongforge/models/diffusion/qwen_image/qwen_image_rope.py": "megatron/core/models/common/embeddings/rope_utils.py",
    }
    if path in exact:
        return exact[path]
    if name.endswith("_layer_spec.py"):
        if path.startswith("loongforge/models/encoder/"):
            return "megatron/core/models/vision/vit_layer_specs.py"
        if "mimo" in path:
            return "megatron/core/models/mimo + gpt_layer_specs.py"
        return "megatron/core/models/gpt/gpt_layer_specs.py + moe_module_specs.py"
    if name.endswith("_attention.py"):
        return "megatron/core/transformer/attention.py / multi_latent_attention.py"
    if name.endswith("_transformer_layer.py"):
        return "megatron/core/transformer/transformer_layer.py"
    if name.endswith("_model.py"):
        if path.startswith("loongforge/models/encoder/"):
            return "Megatron vision model / historical NVLM example"
        return "megatron/core/models/gpt/gpt_model.py"
    return "待逐文件追溯"


def change_hint(path: str) -> str:
    name = Path(path).name
    if "checkpointing" in name:
        return "checkpoint tracker/helper 位置、async save、PEFT/LoRA state dict、checkpoint format"
    if "training_utils" in name:
        return "moved imports、optimizer config_overrides、VPP iterator、RL/modelopt/蒸馏删除逻辑"
    if "validators" in name:
        return "删除参数、Gloo 名称、pipeline/recompute 校验和 VLM adapter 覆盖"
    if "initialize" in name:
        return "async worker rank、MoE aux-loss scaler 位置、process-group 初始化"
    if "pretrain_llm" in name or "sft_llm" in name or "get_loss_func" in name:
        return "batch helper、MTP placement/signature、dataset builder、loss function"
    if "pretrain_vlm" in name:
        return "VLM batch/loss、dataset builder、MTP 和 packed-seq 接口"
    if "base_gpt_model" in name or name.endswith("_model.py"):
        return "GPTModel 构造、forward、preprocess/postprocess、shared embedding/output、VPP schedule"
    if "transformer_block" in name:
        return "TransformerBlock 构造、forward、CUDA graph、pipeline layout、model chunk schedule"
    if "attention" in name or "mla" in name:
        return "attention/MLA 构造、inference context、RoPE、packed sequence、CP group"
    if "experts" in name or "router" in name or "moe" in name:
        return "MoE expert/router、grouped GEMM、TP/EP group、aux loss"
    if "transformer_layer" in name:
        return "TransformerLayer 构造、submodules、recompute、CUDA graph"
    if "layer_spec" in name:
        return "ModuleSpec、TE/local linear、activation、MoE backend API"
    if "rope" in name:
        return "RoPE helper、CP group、packed sequence、MLA rotary mode"
    if "rerun_iterator" in name:
        return "RerunDataIterator 基类协议与 replay 状态"
    if "dsa" in name or "csa" in name:
        return "DSA/CSA kernel 导出位置、CP layout、packed sequence、算子签名"
    if "local_norm" in name:
        return "LayerNorm/RMSNorm fallback、TE/Apex 支持和 module API"
    return "按目标 v0.19.2 API 逐函数核对"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--loongforge-root", type=Path, required=True)
    parser.add_argument("--similarity", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    root = args.loongforge_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    confirmed = declared_source_files(root)
    candidates = candidate_model_files(root, confirmed)
    similarity = {row["loongforge_path"]: row for row in json.loads(args.similarity.read_text(encoding="utf-8"))}

    records = []
    for index, path in enumerate(sorted(confirmed | candidates), 1):
        is_confirmed = path in confirmed
        sim = similarity.get(path, {})
        ratio_override = RATIO_OVERRIDES.get(path)
        evidence_parts = []
        if "Modified from Megatron-LM" in "\n".join(
            (root / path).read_text(encoding="utf-8", errors="ignore").splitlines()[:12]
        ):
            evidence_parts.append("文件头声明 Modified from Megatron-LM")
        if not is_confirmed:
            evidence_parts.append("模型侧派生候选：同名职责/同构方法，但未声明源码")
        elif path in {
            "loongforge/train/validators.py",
            "loongforge/models/common/experimental_attention_variant/dsa_fused_kernels.py",
            "loongforge/models/common/experimental_attention_variant/dsa_fused_utils.py",
            "loongforge/models/foundation/deepseek_v4/deepseek_v4_csa.py",
        }:
            evidence_parts.append("人工映射或 vendored/import 审计确认")
        if ratio_override:
            evidence_parts.append(f"对应 upstream 主文件归一化相似度 {ratio_override[1]:.3f}")
        elif sim.get("best_old_containment"):
            evidence_parts.append(
                f"旧版结构相似度 {sim['best_old_containment']:.3f}；"
                f"最接近 {sim['best_old_version']} {sim['best_old_source']}"
            )
        records.append(
            {
                "id": f"LF-COPY-{index:03d}",
                "confidence": (
                    "A-已确认" if is_confirmed else ("C-扩展候选" if path in EXTRA_CANDIDATES else "B-派生候选")
                ),
                "group": group_for(path),
                "loongforge_path": path,
                "probable_megatron_source": source_for(path),
                "best_old_version": (
                    ratio_override[0] if ratio_override else sim.get("best_old_version", "v0.15 或更早")
                ),
                "upgrade_hint": change_hint(path),
                "evidence": "；".join(evidence_parts),
            }
        )

    with (output_dir / "inventory.csv").open("w", newline="") as handle:
        fieldnames = list(records[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(records)
    (output_dir / "inventory.json").write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n")
    print(f"WROTE {output_dir / 'inventory.csv'} ({len(confirmed)} confirmed + {len(candidates)} candidates)")


if __name__ == "__main__":
    main()

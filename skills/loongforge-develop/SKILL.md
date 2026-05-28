---
name: loongforge-develop
description: Use when implementing features or refactoring LoongForge code. Performs pre-implementation impact analysis to prevent cross-module breakage. Triggers on 'implement', 'add model', 'refactor', 'new feature', 'change trainer', 'modify dispatch', 'add support'.
---

# Pre-Implementation Impact Analysis

Before writing code, identify what your change touches and what else must be co-modified.

## Impact Matrix

| Area Changed | Must Also Check/Update |
|---|---|
| `loongforge/utils/constants.py` | config_map.py entries, configs/models/ YAML `model_type` field, trainer_builder.py dispatch, examples/ launch scripts |
| `loongforge/utils/config_map.py` | constants.py family exists, configs/models/ YAML exists at declared path, examples/ script uses correct `--model-name` |
| `loongforge/models/foundation/<new>/` | foundation/__init__.py import, constants.py family, config_map.py entry, trainer in train/pretrain/ or train/sft/, YAML in configs/models/ |
| `loongforge/models/encoder/<new>/` | omni_models/ composition, VLM trainer (sft_vlm/pretrain_vlm), mm_plugin.py if new modality |
| `loongforge/models/omni_models/` | encoder/ counterpart exists, VLM family in constants.py, model_chunk_schedule_plan.py for PP |
| `loongforge/train/trainer_builder.py` | constants.py family classes used in dispatch, all trainer imports in train/__init__.py |
| `loongforge/train/training_utils.py` | All trainer paths (pretrain + sft), Megatron-LM API compatibility |
| `loongforge/models/dispatch.py` | All model forward paths (GPU + XPU), TransformerEngine patch compatibility |
| `tools/convert_checkpoint/key_mappings/` | Exact weight naming in model definition, TP/PP dimension splits in module_convertor/ |
| `loongforge/data/` | chat_template.py, sft collators, model-specific data handling |
| `configs/models/<family>/` | config_map.py path/name match, `model_type` matches constants.py |
| `third_party/Loong-Megatron` | NEVER change submodule pointer without explicit intent; verify patches/ still apply |
| `patches/TransformerEngine_*` | dispatch.py compatibility, setup_env.py tag reference |

## Safety Rules

1. **Family string consistency**: the same string must appear in constants.py, config_map.py, YAML `model_type`, and example scripts.
2. **No accidental submodule changes**: never commit a `third_party/Loong-Megatron` pointer change unless intentional.
3. **SPDX headers required**: all new `.py/.sh/.cu/.cpp/.h` files (outside `third_party/`, `patches/`) need the Apache-2.0 header.
4. **No large files**: nothing > 1MB committed.
5. **VLM atomicity**: encoder + projector + decoder + omni_model_provider must be updated together.
6. **Protected files need extra scrutiny**: constants.py, config_map.py, trainer_builder.py, training_utils.py, dispatch.py.

## Pre-Implementation Checklist

1. Identify which row(s) in the Impact Matrix your change touches.
2. List all files that must be co-modified for consistency.
3. If adding a model: verify the family string is consistent across all layers.
4. If touching VLM: confirm encoder + projector + decoder are all addressed.
5. If touching checkpoint conversion: verify key_mappings match model weight attribute names exactly.
6. If touching dispatch.py: verify both GPU and XPU paths are handled.

## Routing

| Situation | Skill |
|---|---|
| Implementation complete, ready to self-review | `/loongforge-review` |
| Ready to submit PR | `/submit-pr` |
| Training fails after your change | `/loongforge-debug` |
| Launching a training job to verify | `/loongforge-launch-training` |

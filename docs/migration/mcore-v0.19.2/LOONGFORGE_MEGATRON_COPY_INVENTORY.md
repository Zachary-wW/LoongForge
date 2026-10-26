# LoongForge 中源自 Megatron-LM 的代码盘点

> 审计日期：2026-10-26
> LoongForge 基线：`a8268b7973e108fb61676a2cb967961dccb9851f`（`master`）
> 对比版本：NVIDIA Megatron-LM `core_v0.10.0`、`v0.12.0`、`v0.13.0`、`v0.14.0`、`v0.15.0`、`v0.19.2`
> 机器清单：[inventory.csv](inventory.csv)
> 相似度明细：[similarity_results.csv](similarity_results.csv)

## 结论

这轮共整理出 **118 个需要纳入 review 的文件**：

- **51 个 A 类：已确认从 Megatron-LM 派生。** 其中 47 个文件头明确写着 `Modified from Megatron-LM`，另 4 个由此前人工映射、import/vendor 审计确认。
- **28 个 B 类：模型侧强候选。** 文件名、构造方式或核心方法明显沿用 Megatron。
- **39 个 C 类：扩展 review 候选。** 包括数据构建、tokenizer、Diffusion、VLM adapter 等；代码可能经过大量改写，但目标 Megatron API 变化后仍需逐文件确认。

此前文档只覆盖了训练入口和几个共享基类，遗漏最多的部分是：

- 各模型的 `*_model.py`、`*_attention.py`、`*_transformer_layer.py`、`*_layer_spec.py`。
- `model_chunk_schedule_plan.py`、`fine_grained_callables.py`。
- MLA、local attention/norm、DSA/CSA vendor 代码。
- VLM 的 ViT layer spec、ViT attention/layer 和 RoPE 实现。

与旧清单相比：

- 旧表可精确对应的文件只有 10 个。
- 新清单在 A 类中另有 41 个明确来源文件，主要集中在模型的 layer spec、experts/router、RoPE、VLM、model chunk schedule 和 fine-grained callables。
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

### 训练入口/运行时（9）

| ID | 分组 | LoongForge 文件 | 推断 Megatron 来源 | 版本线索 | 升级关注点 | 证据 |
|---|---|---|---|---|---|---|
| LF-COPY-003 | 训练入口/运行时 | `loongforge/data/dp_balance/rerun_iterator.py` | megatron/core/rerun_state_machine.py::RerunDataIterator |  | RerunDataIterator 基类协议与 replay 状态 | 文件头声明 Modified from Megatron-LM |
| LF-COPY-106 | 训练入口/运行时 | `loongforge/train/checkpointing.py` | megatron/training/checkpointing.py | v0.15 | checkpoint tracker/helper 位置、async save、PEFT/LoRA state dict、checkpoint format | 文件头声明 Modified from Megatron-LM；对应 upstream 主文件归一化相似度 0.900；已确认 1 个符号缺失；3 条调用签名疑点 |
| LF-COPY-107 | 训练入口/运行时 | `loongforge/train/get_loss_func.py` | pretrain_gpt.py | v0.15 | batch helper、MTP placement/signature、dataset builder、loss function | 文件头声明 Modified from Megatron-LM；对应 upstream 主文件归一化相似度 0.207 |
| LF-COPY-108 | 训练入口/运行时 | `loongforge/train/initialize.py` | megatron/training/initialize.py | v0.15 | async worker rank、MoE aux-loss scaler 位置、process-group 初始化 | 文件头声明 Modified from Megatron-LM；对应 upstream 主文件归一化相似度 0.271；已确认 1 个符号缺失；1 条调用签名疑点 |
| LF-COPY-111 | 训练入口/运行时 | `loongforge/train/pretrain/pretrain_llm.py` | pretrain_gpt.py | v0.15 | batch helper、MTP placement/signature、dataset builder、loss function | 文件头声明 Modified from Megatron-LM；对应 upstream 主文件归一化相似度 0.490；已确认 3 个符号缺失；4 条调用签名疑点 |
| LF-COPY-112 | 训练入口/运行时 | `loongforge/train/pretrain/pretrain_vlm.py` | pretrain_vlm.py | v0.14 / v0.15 | VLM batch/loss、dataset builder、MTP 和 packed-seq 接口 | 文件头声明 Modified from Megatron-LM；对应 upstream 主文件归一化相似度 0.060 |
| LF-COPY-113 | 训练入口/运行时 | `loongforge/train/sft/sft_llm.py` | pretrain_gpt.py | v0.15 | batch helper、MTP placement/signature、dataset builder、loss function | 文件头声明 Modified from Megatron-LM；对应 upstream 主文件归一化相似度 0.340 |
| LF-COPY-116 | 训练入口/运行时 | `loongforge/train/training_utils.py` | megatron/training/training.py | v0.14 / v0.15 | moved imports、optimizer config_overrides、VPP iterator、RL/modelopt/蒸馏删除逻辑 | 文件头声明 Modified from Megatron-LM；对应 upstream 主文件归一化相似度 0.468；已确认 2 个符号缺失；3 条调用签名疑点 |
| LF-COPY-117 | 训练入口/运行时 | `loongforge/train/validators.py` | megatron/training/arguments.py | v0.12 | 删除参数、Gloo 名称、pipeline/recompute 校验和 VLM adapter 覆盖 | 人工映射或 vendored/import 审计确认；对应 upstream 主文件归一化相似度 0.299 |

### 共享模型/算子（8）

| ID | 分组 | LoongForge 文件 | 推断 Megatron 来源 | 版本线索 | 升级关注点 | 证据 |
|---|---|---|---|---|---|---|
| LF-COPY-010 | 共享模型/算子 | `loongforge/models/common/experimental_attention_variant/dsa_fused.py` | Megatron DSA / fused sparse attention |  | DSA/CSA kernel 导出位置、CP layout、packed sequence、算子签名 | 文件头声明 Modified from Megatron-LM；已确认 1 个符号缺失 |
| LF-COPY-011 | 共享模型/算子 | `loongforge/models/common/experimental_attention_variant/dsa_fused_kernels.py` | Megatron DSA kernels | core_v0.14.0 | DSA/CSA kernel 导出位置、CP layout、packed sequence、算子签名 | 人工映射或 vendored/import 审计确认；旧版结构相似度 0.287；最接近 core_v0.14.0 megatron/core/fusions/fused_mla_yarn_rope_apply.py |
| LF-COPY-012 | 共享模型/算子 | `loongforge/models/common/experimental_attention_variant/dsa_fused_utils.py` | Megatron DSA utilities |  | DSA/CSA kernel 导出位置、CP layout、packed sequence、算子签名 | 人工映射或 vendored/import 审计确认 |
| LF-COPY-013 | 共享模型/算子 | `loongforge/models/common/experimental_attention_variant/multi_latent_attention.py` | megatron/core/transformer/multi_latent_attention.py | v0.15 | attention/MLA 构造、inference context、RoPE、packed sequence、CP group | 文件头声明 Modified from Megatron-LM；对应 upstream 主文件归一化相似度 0.196 |
| LF-COPY-015 | 共享模型/算子 | `loongforge/models/common/local_layers/local_attention.py` | megatron/core/transformer/attention.py | core_v0.10.0 | attention/MLA 构造、inference context、RoPE、packed sequence、CP group | 文件头声明 Modified from Megatron-LM；旧版结构相似度 0.014；最接近 core_v0.10.0 megatron/core/transformer/attention.py |
| LF-COPY-016 | 共享模型/算子 | `loongforge/models/common/local_layers/local_norm.py` | megatron/core/transformer/torch_norm.py + fused_layer_norm.py |  | LayerNorm/RMSNorm fallback、TE/Apex 支持和 module API | 文件头声明 Modified from Megatron-LM |
| LF-COPY-102 | 共享模型/算子 | `loongforge/models/omni_models/fine_grained_callables.py` | megatron/core/models/gpt/fine_grained_callables.py | v0.14 / v0.15 | 按目标 v0.19.2 API 逐函数核对 | 文件头声明 Modified from Megatron-LM；对应 upstream 主文件归一化相似度 0.628；已确认 1 个符号缺失 |
| LF-COPY-103 | 共享模型/算子 | `loongforge/models/omni_models/model_chunk_schedule_plan.py` | megatron/core/models/common/model_chunk_schedule_plan.py | v0.14 / v0.15 | 按目标 v0.19.2 API 逐函数核对 | 文件头声明 Modified from Megatron-LM；对应 upstream 主文件归一化相似度 0.834 |

### 语言基础模型（24）

| ID | 分组 | LoongForge 文件 | 推断 Megatron 来源 | 版本线索 | 升级关注点 | 证据 |
|---|---|---|---|---|---|---|
| LF-COPY-057 | 语言基础模型 | `loongforge/models/foundation/base/base_gpt_model.py` | megatron/core/models/gpt/gpt_model.py | v0.15 | GPTModel 构造、forward、preprocess/postprocess、shared embedding/output、VPP schedule | 文件头声明 Modified from Megatron-LM；对应 upstream 主文件归一化相似度 0.706；2 条调用签名疑点 |
| LF-COPY-058 | 语言基础模型 | `loongforge/models/foundation/deepseek/deepseek_layer_spec.py` | megatron/core/models/gpt/gpt_layer_specs.py + moe_module_specs.py |  | ModuleSpec、TE/local linear、activation、MoE backend API | 文件头声明 Modified from Megatron-LM |
| LF-COPY-060 | 语言基础模型 | `loongforge/models/foundation/deepseek_v4/deepseek_v4_attention.py` | megatron/core/transformer/attention.py / multi_latent_attention.py |  | attention/MLA 构造、inference context、RoPE、packed sequence、CP group | 文件头声明 Modified from Megatron-LM；已确认 4 个符号缺失 |
| LF-COPY-061 | 语言基础模型 | `loongforge/models/foundation/deepseek_v4/deepseek_v4_csa.py` | Megatron CSA/DSA experimental attention |  | DSA/CSA kernel 导出位置、CP layout、packed sequence、算子签名 | 人工映射或 vendored/import 审计确认；已确认 6 个符号缺失 |
| LF-COPY-064 | 语言基础模型 | `loongforge/models/foundation/deepseek_v4/deepseek_v4_rope.py` | megatron/core/models/common/embeddings/rope_utils.py |  | RoPE helper、CP group、packed sequence、MLA rotary mode | 文件头声明 Modified from Megatron-LM |
| LF-COPY-065 | 语言基础模型 | `loongforge/models/foundation/ernie4_5_vl/ernie4_5_vl_moe_model.py` | megatron/core/models/gpt/gpt_model.py |  | GPTModel 构造、forward、preprocess/postprocess、shared embedding/output、VPP schedule | 文件头声明 Modified from Megatron-LM |
| LF-COPY-066 | 语言基础模型 | `loongforge/models/foundation/ernie4_5_vl/ernie_decoder_layer_spec.py` | megatron/core/models/gpt/gpt_layer_specs.py + moe_module_specs.py |  | ModuleSpec、TE/local linear、activation、MoE backend API | 文件头声明 Modified from Megatron-LM |
| LF-COPY-067 | 语言基础模型 | `loongforge/models/foundation/ernie4_5_vl/ernie_experts.py` | megatron/core/transformer/moe/experts.py + mlp.py | v0.15 | MoE expert/router、grouped GEMM、TP/EP group、aux loss | 文件头声明 Modified from Megatron-LM；对应 upstream 主文件归一化相似度 0.355 |
| LF-COPY-069 | 语言基础模型 | `loongforge/models/foundation/ernie4_5_vl/ernie_router.py` | megatron/core/transformer/moe/router.py | v0.14 / v0.15 | MoE expert/router、grouped GEMM、TP/EP group、aux loss | 文件头声明 Modified from Megatron-LM；对应 upstream 主文件归一化相似度 0.260 |
| LF-COPY-070 | 语言基础模型 | `loongforge/models/foundation/ernie4_5_vl/ernie_transformer_layer.py` | megatron/core/transformer/transformer_layer.py | v0.15 | TransformerLayer 构造、submodules、recompute、CUDA graph | 文件头声明 Modified from Megatron-LM；对应 upstream 主文件归一化相似度 0.317；已确认 1 个符号缺失 |
| LF-COPY-076 | 语言基础模型 | `loongforge/models/foundation/internlm/internlm_layer_spec.py` | megatron/core/models/gpt/gpt_layer_specs.py + moe_module_specs.py |  | ModuleSpec、TE/local linear、activation、MoE backend API | 文件头声明 Modified from Megatron-LM |
| LF-COPY-082 | 语言基础模型 | `loongforge/models/foundation/language_transformer_block.py` | megatron/core/transformer/transformer_block.py | v0.15 | TransformerBlock 构造、forward、CUDA graph、pipeline layout、model chunk schedule | 文件头声明 Modified from Megatron-LM；对应 upstream 主文件归一化相似度 0.345 |
| LF-COPY-083 | 语言基础模型 | `loongforge/models/foundation/llama/llama_layer_spec.py` | megatron/core/models/gpt/gpt_layer_specs.py + moe_module_specs.py |  | ModuleSpec、TE/local linear、activation、MoE backend API | 文件头声明 Modified from Megatron-LM |
| LF-COPY-085 | 语言基础模型 | `loongforge/models/foundation/llm_model_provider.py` | pretrain_gpt.py / model provider |  | 按目标 v0.19.2 API 逐函数核对 | 文件头声明 Modified from Megatron-LM |
| LF-COPY-086 | 语言基础模型 | `loongforge/models/foundation/mimo/mimo_layer_spec.py` | megatron/core/models/mimo + gpt_layer_specs.py |  | ModuleSpec、TE/local linear、activation、MoE backend API | 文件头声明 Modified from Megatron-LM |
| LF-COPY-089 | 语言基础模型 | `loongforge/models/foundation/minimax/attention.py` | megatron/core/transformer/attention.py | v0.10 | attention/MLA 构造、inference context、RoPE、packed sequence、CP group | 文件头声明 Modified from Megatron-LM；对应 upstream 主文件归一化相似度 0.355 |
| LF-COPY-090 | 语言基础模型 | `loongforge/models/foundation/minimax/minimax_layer_spec.py` | megatron/core/models/gpt/gpt_layer_specs.py + moe_module_specs.py |  | ModuleSpec、TE/local linear、activation、MoE backend API | 文件头声明 Modified from Megatron-LM |
| LF-COPY-092 | 语言基础模型 | `loongforge/models/foundation/qwen2/qwen_layer_spec.py` | megatron/core/models/gpt/gpt_layer_specs.py + moe_module_specs.py |  | ModuleSpec、TE/local linear、activation、MoE backend API | 文件头声明 Modified from Megatron-LM |
| LF-COPY-093 | 语言基础模型 | `loongforge/models/foundation/qwen2/qwen_model.py` | megatron/core/models/gpt/gpt_model.py |  | GPTModel 构造、forward、preprocess/postprocess、shared embedding/output、VPP schedule | 文件头声明 Modified from Megatron-LM |
| LF-COPY-094 | 语言基础模型 | `loongforge/models/foundation/qwen3/qwen_layer_spec.py` | megatron/core/models/gpt/gpt_layer_specs.py + moe_module_specs.py |  | ModuleSpec、TE/local linear、activation、MoE backend API | 文件头声明 Modified from Megatron-LM |
| LF-COPY-095 | 语言基础模型 | `loongforge/models/foundation/qwen3/qwen_model.py` | megatron/core/models/gpt/gpt_model.py |  | GPTModel 构造、forward、preprocess/postprocess、shared embedding/output、VPP schedule | 文件头声明 Modified from Megatron-LM |
| LF-COPY-096 | 语言基础模型 | `loongforge/models/foundation/qwen3_5/qwen3_5_layer_spec.py` | megatron/core/models/gpt/gpt_layer_specs.py + moe_module_specs.py |  | ModuleSpec、TE/local linear、activation、MoE backend API | 文件头声明 Modified from Megatron-LM |
| LF-COPY-098 | 语言基础模型 | `loongforge/models/foundation/qwen3_next/gated_attention.py` | megatron/core/transformer/attention.py | v0.13 | attention/MLA 构造、inference context、RoPE、packed sequence、CP group | 文件头声明 Modified from Megatron-LM；对应 upstream 主文件归一化相似度 0.381 |
| LF-COPY-100 | 语言基础模型 | `loongforge/models/foundation/qwen3_next/qwen3_next_layer_spec.py` | megatron/core/models/gpt/gpt_layer_specs.py + moe_module_specs.py |  | ModuleSpec、TE/local linear、activation、MoE backend API | 文件头声明 Modified from Megatron-LM |

### 视觉/多模态模型（9）

| ID | 分组 | LoongForge 文件 | 推断 Megatron 来源 | 版本线索 | 升级关注点 | 证据 |
|---|---|---|---|---|---|---|
| LF-COPY-037 | 视觉/多模态模型 | `loongforge/models/encoder/internvl_vision_models/intern_vision_attention.py` | megatron/core/transformer/attention.py / multi_latent_attention.py | core_v0.12.0 | attention/MLA 构造、inference context、RoPE、packed sequence、CP group | 文件头声明 Modified from Megatron-LM；旧版结构相似度 0.363；最接近 core_v0.12.0 examples/multimodal/nvlm/internvit.py |
| LF-COPY-039 | 视觉/多模态模型 | `loongforge/models/encoder/internvl_vision_models/intern_vision_transformer_layer.py` | megatron/core/transformer/transformer_layer.py |  | TransformerLayer 构造、submodules、recompute、CUDA graph | 文件头声明 Modified from Megatron-LM |
| LF-COPY-040 | 视觉/多模态模型 | `loongforge/models/encoder/internvl_vision_models/internvl_layer_spec.py` | megatron/core/models/vision/vit_layer_specs.py |  | ModuleSpec、TE/local linear、activation、MoE backend API | 文件头声明 Modified from Megatron-LM |
| LF-COPY-041 | 视觉/多模态模型 | `loongforge/models/encoder/llavaov1_5_vision_models/llavaov_1_5_layer_spec.py` | megatron/core/models/vision/vit_layer_specs.py |  | ModuleSpec、TE/local linear、activation、MoE backend API | 文件头声明 Modified from Megatron-LM |
| LF-COPY-049 | 视觉/多模态模型 | `loongforge/models/encoder/qwen2_vl_vision_models/qwen2_vl_layer_spec.py` | megatron/core/models/vision/vit_layer_specs.py |  | ModuleSpec、TE/local linear、activation、MoE backend API | 文件头声明 Modified from Megatron-LM |
| LF-COPY-051 | 视觉/多模态模型 | `loongforge/models/encoder/qwen3_5_vision_models/qwen3_5_vision_layer_spec.py` | megatron/core/models/vision/vit_layer_specs.py |  | ModuleSpec、TE/local linear、activation、MoE backend API | 文件头声明 Modified from Megatron-LM |
| LF-COPY-053 | 视觉/多模态模型 | `loongforge/models/encoder/qwen3_vl_vision_models/qwen3_vl_layer_spec.py` | megatron/core/models/vision/vit_layer_specs.py |  | ModuleSpec、TE/local linear、activation、MoE backend API | 文件头声明 Modified from Megatron-LM |
| LF-COPY-054 | 视觉/多模态模型 | `loongforge/models/encoder/qwen3_vl_vision_models/rope_utils.py` | megatron/core/models/common/embeddings/rope_utils.py | core_v0.10.0 | RoPE helper、CP group、packed sequence、MLA rotary mode | 文件头声明 Modified from Megatron-LM；旧版结构相似度 0.412；最接近 core_v0.10.0 megatron/core/models/common/embeddings/rope_utils.py |
| LF-COPY-056 | 视觉/多模态模型 | `loongforge/models/encoder/vision_transformer_block.py` | megatron/core/transformer/transformer_block.py | v0.15 | TransformerBlock 构造、forward、CUDA graph、pipeline layout、model chunk schedule | 文件头声明 Modified from Megatron-LM；对应 upstream 主文件归一化相似度 0.378 |

### 其他（1）

| ID | 分组 | LoongForge 文件 | 推断 Megatron 来源 | 版本线索 | 升级关注点 | 证据 |
|---|---|---|---|---|---|---|
| LF-COPY-118 | 其他 | `loongforge/utils/utils.py` | megatron/training/utils.py + megatron/core/utils.py |  | 按目标 v0.19.2 API 逐函数核对 | 文件头声明 Modified from Megatron-LM |

## B 类：模型侧派生候选

| ID | 分组 | LoongForge 文件 | 推断 Megatron 来源 | 版本线索 | 升级关注点 | 证据 |
|---|---|---|---|---|---|---|
| LF-COPY-034 | 视觉/多模态模型 | `loongforge/models/encoder/base_vision_models/base_vision_model.py` | Megatron vision model / historical NVLM example |  | GPTModel 构造、forward、preprocess/postprocess、shared embedding/output、VPP schedule | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-036 | 视觉/多模态模型 | `loongforge/models/encoder/ernie4_5_vl_vision_models/ernie_vision_model.py` | Megatron vision model / historical NVLM example |  | GPTModel 构造、forward、preprocess/postprocess、shared embedding/output、VPP schedule | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-038 | 视觉/多模态模型 | `loongforge/models/encoder/internvl_vision_models/intern_vision_model.py` | Megatron vision model / historical NVLM example |  | GPTModel 构造、forward、preprocess/postprocess、shared embedding/output、VPP schedule | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-042 | 视觉/多模态模型 | `loongforge/models/encoder/llavaov1_5_vision_models/rice_vision_model.py` | Megatron vision model / historical NVLM example |  | GPTModel 构造、forward、preprocess/postprocess、shared embedding/output、VPP schedule | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-043 | 视觉/多模态模型 | `loongforge/models/encoder/minicpm_v_4_6_vision_models/minicpm_v_4_6_layer_spec.py` | megatron/core/models/vision/vit_layer_specs.py |  | ModuleSpec、TE/local linear、activation、MoE backend API | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-045 | 视觉/多模态模型 | `loongforge/models/encoder/moon_vision_models/moon_vision_layer_spec.py` | megatron/core/models/vision/vit_layer_specs.py |  | ModuleSpec、TE/local linear、activation、MoE backend API | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-046 | 视觉/多模态模型 | `loongforge/models/encoder/moon_vision_models/moon_vision_model.py` | Megatron vision model / historical NVLM example |  | GPTModel 构造、forward、preprocess/postprocess、shared embedding/output、VPP schedule | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-050 | 视觉/多模态模型 | `loongforge/models/encoder/qwen2_vl_vision_models/vision_model.py` | Megatron vision model / historical NVLM example |  | GPTModel 构造、forward、preprocess/postprocess、shared embedding/output、VPP schedule | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-052 | 视觉/多模态模型 | `loongforge/models/encoder/qwen3_5_vision_models/vision_model.py` | Megatron vision model / historical NVLM example |  | GPTModel 构造、forward、preprocess/postprocess、shared embedding/output、VPP schedule | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-055 | 视觉/多模态模型 | `loongforge/models/encoder/qwen3_vl_vision_models/vision_model.py` | Megatron vision model / historical NVLM example |  | GPTModel 构造、forward、preprocess/postprocess、shared embedding/output、VPP schedule | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-059 | 语言基础模型 | `loongforge/models/foundation/deepseek/deepseek_model.py` | megatron/core/models/gpt/gpt_model.py |  | GPTModel 构造、forward、preprocess/postprocess、shared embedding/output、VPP schedule | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-062 | 语言基础模型 | `loongforge/models/foundation/deepseek_v4/deepseek_v4_layer_spec.py` | megatron/core/models/gpt/gpt_layer_specs.py + moe_module_specs.py |  | ModuleSpec、TE/local linear、activation、MoE backend API | 模型侧派生候选：同名职责/同构方法，但未声明源码；1 条调用签名疑点 |
| LF-COPY-063 | 语言基础模型 | `loongforge/models/foundation/deepseek_v4/deepseek_v4_model.py` | megatron/core/models/gpt/gpt_model.py |  | GPTModel 构造、forward、preprocess/postprocess、shared embedding/output、VPP schedule | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-071 | 语言基础模型 | `loongforge/models/foundation/glm/glm_layer_spec.py` | megatron/core/models/gpt/gpt_layer_specs.py + moe_module_specs.py |  | ModuleSpec、TE/local linear、activation、MoE backend API | 模型侧派生候选：同名职责/同构方法，但未声明源码；已确认 1 个符号缺失 |
| LF-COPY-072 | 语言基础模型 | `loongforge/models/foundation/glm/glm_model.py` | megatron/core/models/gpt/gpt_model.py |  | GPTModel 构造、forward、preprocess/postprocess、shared embedding/output、VPP schedule | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-073 | 语言基础模型 | `loongforge/models/foundation/glm5_next/glm5_next_attention.py` | megatron/core/transformer/attention.py / multi_latent_attention.py |  | attention/MLA 构造、inference context、RoPE、packed sequence、CP group | 模型侧派生候选：同名职责/同构方法，但未声明源码；已确认 1 个符号缺失 |
| LF-COPY-074 | 语言基础模型 | `loongforge/models/foundation/glm5_next/glm5_next_layer_spec.py` | megatron/core/models/gpt/gpt_layer_specs.py + moe_module_specs.py |  | ModuleSpec、TE/local linear、activation、MoE backend API | 模型侧派生候选：同名职责/同构方法，但未声明源码；1 条调用签名疑点 |
| LF-COPY-075 | 语言基础模型 | `loongforge/models/foundation/glm5_next/glm5_next_model.py` | megatron/core/models/gpt/gpt_model.py |  | GPTModel 构造、forward、preprocess/postprocess、shared embedding/output、VPP schedule | 模型侧派生候选：同名职责/同构方法，但未声明源码；已确认 1 个符号缺失 |
| LF-COPY-077 | 语言基础模型 | `loongforge/models/foundation/internlm/internlm_model.py` | megatron/core/models/gpt/gpt_model.py |  | GPTModel 构造、forward、preprocess/postprocess、shared embedding/output、VPP schedule | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-078 | 语言基础模型 | `loongforge/models/foundation/kimi_k3/kimi_k3_attention.py` | megatron/core/transformer/attention.py / multi_latent_attention.py |  | attention/MLA 构造、inference context、RoPE、packed sequence、CP group | 模型侧派生候选：同名职责/同构方法，但未声明源码；3 条调用签名疑点 |
| LF-COPY-079 | 语言基础模型 | `loongforge/models/foundation/kimi_k3/kimi_k3_layer_spec.py` | megatron/core/models/gpt/gpt_layer_specs.py + moe_module_specs.py |  | ModuleSpec、TE/local linear、activation、MoE backend API | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-080 | 语言基础模型 | `loongforge/models/foundation/kimi_k3/kimi_k3_model.py` | megatron/core/models/gpt/gpt_model.py |  | GPTModel 构造、forward、preprocess/postprocess、shared embedding/output、VPP schedule | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-081 | 语言基础模型 | `loongforge/models/foundation/kimi_k3/kimi_k3_transformer_layer.py` | megatron/core/transformer/transformer_layer.py |  | TransformerLayer 构造、submodules、recompute、CUDA graph | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-084 | 语言基础模型 | `loongforge/models/foundation/llama/llama_model.py` | megatron/core/models/gpt/gpt_model.py |  | GPTModel 构造、forward、preprocess/postprocess、shared embedding/output、VPP schedule | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-087 | 语言基础模型 | `loongforge/models/foundation/mimo/mimo_model.py` | megatron/core/models/gpt/gpt_model.py |  | GPTModel 构造、forward、preprocess/postprocess、shared embedding/output、VPP schedule | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-091 | 语言基础模型 | `loongforge/models/foundation/minimax/minimax_model.py` | megatron/core/models/gpt/gpt_model.py |  | GPTModel 构造、forward、preprocess/postprocess、shared embedding/output、VPP schedule | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-097 | 语言基础模型 | `loongforge/models/foundation/qwen3_5/qwen3_5_model.py` | megatron/core/models/gpt/gpt_model.py | core_v0.15.0 | GPTModel 构造、forward、preprocess/postprocess、shared embedding/output、VPP schedule | 模型侧派生候选：同名职责/同构方法，但未声明源码；旧版结构相似度 0.154；最接近 core_v0.15.0 megatron/core/models/gpt/gpt_model.py |
| LF-COPY-101 | 语言基础模型 | `loongforge/models/foundation/qwen3_next/qwen3_next_model.py` | megatron/core/models/gpt/gpt_model.py | core_v0.12.0 | GPTModel 构造、forward、preprocess/postprocess、shared embedding/output、VPP schedule | 模型侧派生候选：同名职责/同构方法，但未声明源码；旧版结构相似度 0.412；最接近 core_v0.12.0 megatron/core/models/gpt/gpt_model.py |
| LF-COPY-001 | 数据/Tokenizer | `loongforge/data/blended_hf_dataset_builder.py` | megatron/core/datasets/blended_megatron_dataset_builder.py |  | 按目标 v0.19.2 API 逐函数核对 | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-002 | 数据/Tokenizer | `loongforge/data/blended_hf_dataset_config.py` | megatron/core/datasets/blended_megatron_dataset_config.py |  | 按目标 v0.19.2 API 逐函数核对 | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-004 | 数据/Tokenizer | `loongforge/data/encoder_strided_sampler.py` | megatron/legacy/data/data_samplers.py |  | 按目标 v0.19.2 API 逐函数核对 | 模型侧派生候选：同名职责/同构方法，但未声明源码；已确认 1 个符号缺失 |
| LF-COPY-005 | 数据/Tokenizer | `loongforge/data/multimodal/base/task_encoder.py` | examples/multimodal/dataset_helpers.py |  | 按目标 v0.19.2 API 逐函数核对 | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-006 | 数据/Tokenizer | `loongforge/data/multimodal/dataloader_provider.py` | examples/multimodal/dataloader_provider.py |  | 按目标 v0.19.2 API 逐函数核对 | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-007 | 数据/Tokenizer | `loongforge/data/multimodal/internvl/internvl_preprocess.py` | examples/multimodal/image_processing.py |  | 按目标 v0.19.2 API 逐函数核对 | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-008 | 数据/Tokenizer | `loongforge/data/multimodal/internvl/internvl_task_encoder.py` | 待逐文件追溯 |  | 按目标 v0.19.2 API 逐函数核对 | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-009 | 共享模型/算子 | `loongforge/models/common/cuda_graph_base.py` | megatron/core/full_cuda_graph.py |  | 按目标 v0.19.2 API 逐函数核对 | 模型侧派生候选：同名职责/同构方法，但未声明源码；已确认 1 个符号缺失 |
| LF-COPY-014 | 共享模型/算子 | `loongforge/models/common/experimental_attention_variant/sparse_mla_bwd.py` | megatron/core/transformer/experimental_attention_variant/ops/tilelang_sparse_mla_bwd.py |  | attention/MLA 构造、inference context、RoPE、packed sequence、CP group | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-017 | Diffusion | `loongforge/models/diffusion/qwen_image/qwen_image_layer.py` | megatron/core/transformer/transformer_layer.py |  | 按目标 v0.19.2 API 逐函数核对 | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-018 | Diffusion | `loongforge/models/diffusion/qwen_image/qwen_image_layer_spec.py` | GPT/ViT layer specs |  | ModuleSpec、TE/local linear、activation、MoE backend API | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-019 | Diffusion | `loongforge/models/diffusion/qwen_image/qwen_image_model.py` | megatron/core/models/gpt/gpt_model.py |  | GPTModel 构造、forward、preprocess/postprocess、shared embedding/output、VPP schedule | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-020 | Diffusion | `loongforge/models/diffusion/qwen_image/qwen_image_modules.py` | Megatron transformer modules |  | 按目标 v0.19.2 API 逐函数核对 | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-021 | Diffusion | `loongforge/models/diffusion/qwen_image/qwen_image_provider.py` | pretrain/model provider |  | 按目标 v0.19.2 API 逐函数核对 | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-022 | Diffusion | `loongforge/models/diffusion/qwen_image/qwen_image_rope.py` | megatron/core/models/common/embeddings/rope_utils.py |  | RoPE helper、CP group、packed sequence、MLA rotary mode | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-023 | Diffusion | `loongforge/models/diffusion/qwen_image/qwen_image_transformer_block.py` | megatron/core/transformer/transformer_block.py |  | TransformerBlock 构造、forward、CUDA graph、pipeline layout、model chunk schedule | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-024 | Diffusion | `loongforge/models/diffusion/wan/communications.py` | Megatron collective communication wrappers |  | 按目标 v0.19.2 API 逐函数核对 | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-025 | Diffusion | `loongforge/models/diffusion/wan/wan_attention.py` | megatron/core/transformer/attention.py |  | attention/MLA 构造、inference context、RoPE、packed sequence、CP group | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-026 | Diffusion | `loongforge/models/diffusion/wan/wan_layer.py` | megatron/core/transformer/transformer_layer.py + mlp.py |  | 按目标 v0.19.2 API 逐函数核对 | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-027 | Diffusion | `loongforge/models/diffusion/wan/wan_layer_spec.py` | GPT/ViT layer specs |  | ModuleSpec、TE/local linear、activation、MoE backend API | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-028 | Diffusion | `loongforge/models/diffusion/wan/wan_model.py` | megatron/core/models/gpt/gpt_model.py |  | GPTModel 构造、forward、preprocess/postprocess、shared embedding/output、VPP schedule | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-029 | Diffusion | `loongforge/models/diffusion/wan/wan_provider.py` | pretrain/model provider |  | 按目标 v0.19.2 API 逐函数核对 | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-030 | Diffusion | `loongforge/models/diffusion/wan/wan_transformer_block.py` | megatron/core/transformer/transformer_block.py |  | TransformerBlock 构造、forward、CUDA graph、pipeline layout、model chunk schedule | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-031 | Diffusion | `loongforge/models/diffusion/wan/wan_ulysses.py` | Megatron context-parallel attention utilities |  | 按目标 v0.19.2 API 逐函数核对 | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-032 | Diffusion | `loongforge/models/diffusion/wan/wan_utils.py` | megatron/training + core utils |  | 按目标 v0.19.2 API 逐函数核对 | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-033 | 其他 | `loongforge/models/dispatch.py` | Megatron multimodal model dispatch |  | 按目标 v0.19.2 API 逐函数核对 | 模型侧派生候选：同名职责/同构方法，但未声明源码；已确认 1 个符号缺失 |
| LF-COPY-035 | 视觉/多模态模型 | `loongforge/models/encoder/ernie4_5_vl_vision_models/ernie_encoder_spec.py` | megatron/core/models/vision/vit_layer_specs.py |  | 按目标 v0.19.2 API 逐函数核对 | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-044 | 视觉/多模态模型 | `loongforge/models/encoder/minicpm_v_4_6_vision_models/vision_model.py` | Megatron vision model / historical NVLM example |  | GPTModel 构造、forward、preprocess/postprocess、shared embedding/output、VPP schedule | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-047 | 视觉/多模态模型 | `loongforge/models/encoder/moon_vision_models/patch_merger_adapter.py` | Megatron multimodal vision adapter examples |  | 按目标 v0.19.2 API 逐函数核对 | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-048 | 视觉/多模态模型 | `loongforge/models/encoder/qwen2_vl_vision_models/adapter.py` | Megatron multimodal vision adapter examples |  | 按目标 v0.19.2 API 逐函数核对 | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-068 | 语言基础模型 | `loongforge/models/foundation/ernie4_5_vl/ernie_moe_layer.py` | megatron/core/transformer/moe/moe_layer.py |  | MoE expert/router、grouped GEMM、TP/EP group、aux loss | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-088 | 语言基础模型 | `loongforge/models/foundation/mimo/mimo_multi_token_prediction.py` | megatron/core/transformer/multi_token_prediction.py |  | 按目标 v0.19.2 API 逐函数核对 | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-099 | 语言基础模型 | `loongforge/models/foundation/qwen3_next/gated_deltanet.py` | megatron/core/transformer/attention.py + SSM layers |  | 按目标 v0.19.2 API 逐函数核对 | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-104 | 其他 | `loongforge/models/utils.py` | megatron/core/transformer/spec_utils.py |  | 按目标 v0.19.2 API 逐函数核对 | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-105 | 数据/Tokenizer | `loongforge/tokenizer/tokenizer.py` | megatron/training/tokenizer/tokenizer.py | core_v0.10.0 | 按目标 v0.19.2 API 逐函数核对 | 模型侧派生候选：同名职责/同构方法，但未声明源码；旧版结构相似度 0.019；最接近 core_v0.10.0 megatron/training/tokenizer/tokenizer.py |
| LF-COPY-109 | 训练入口/运行时 | `loongforge/train/megatron_trainer.py` | megatron/training/training.py |  | 按目标 v0.19.2 API 逐函数核对 | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-110 | 训练入口/运行时 | `loongforge/train/parser.py` | megatron/training/arguments.py |  | 按目标 v0.19.2 API 逐函数核对 | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-114 | 训练入口/运行时 | `loongforge/train/sft/sft_vlm.py` | pretrain_vlm.py |  | 按目标 v0.19.2 API 逐函数核对 | 模型侧派生候选：同名职责/同构方法，但未声明源码 |
| LF-COPY-115 | 训练入口/运行时 | `loongforge/train/sft/utils.py` | pretrain_gpt.py + Megatron dataloader utilities |  | 按目标 v0.19.2 API 逐函数核对 | 模型侧派生候选：同名职责/同构方法，但未声明源码；已确认 1 个符号缺失 |

B/C 类文件没有明确的 `Modified from Megatron-LM` 头，也没有足够证据认定整文件复制。它们进入候选清单的原因是：

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
- `loongforge/tokenizer/*`：部分文件与 Megatron tokenizer 同构，但主要受 tokenizer/Transformers 版本影响。
- `patches/TransformerEngine_v2.9/*`：属于 TE patch，不是 Megatron Python 源码拷贝。
- 只通过 import 使用 Megatron、没有复制实现主体的 145 个文件：继续由 API 调用扫描维护，不放进本清单。

## 建议的 Review 顺序

1. 先 review A 类中的 P0 训练入口和共享基类。
2. 再按模型族 review A 类的 model/attention/layer/expert/layer-spec。
3. 对 B 类逐文件判断“整文件复制 / 片段复制 / 仅 API 依赖”。
4. 对确认复制且 upstream 已有等价实现的文件，优先改为 import 或薄 wrapper。
5. 对必须保留的 LoongForge 分叉，在文件中明确 header、源 upstream SHA/version、分叉原因和同步 owner。

## 证据文件

- [inventory.csv](inventory.csv)：118 个文件和置信度、推断 upstream 路径、升级关注点。
- [inventory.json](inventory.json)：同一清单的 JSON 版本。
- [similarity_results.csv](similarity_results.csv)：680 个 LoongForge Python 文件的行级结构相似度。
- [token_similarity_results.csv](token_similarity_results.csv)：token 级结构相似度，用于发现经过大规模改写的模型和数据代码。
- [ISSUE_MAP.md](ISSUE_MAP.md)：现有 Issue 与 Copy ID 的映射。
- [scan_similarity.py](../../../tools/migration/megatron_copy_audit/scan_similarity.py)：相似度扫描脚本。
- [scan_token_similarity.py](../../../tools/migration/megatron_copy_audit/scan_token_similarity.py)：token 级补充扫描脚本。
- [build_inventory.py](../../../tools/migration/megatron_copy_audit/build_inventory.py)：清单生成脚本。

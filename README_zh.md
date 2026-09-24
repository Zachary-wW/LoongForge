<p align="right"><sub><a href="./README.md">English</a> | <b>简体中文</b></sub></p>

<div align="center">

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)"  srcset="./docs/assets/images/logo/banner-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./docs/assets/images/logo/banner.svg">
    <img alt="LoongForge" src="./docs/assets/images/logo/banner.svg" width="682">
  </picture>
</p>

<h3 align="center">更快地训练 LLM、VLM、Diffusion 与具身模型</h3>

<p align="center">
  <a href="https://github.com/baidu-baige/LoongForge/stargazers"><img src="https://img.shields.io/github/stars/baidu-baige/LoongForge?style=flat&logo=github&color=4F46E5" alt="GitHub stars"></a>
  <a href="https://hub.docker.com/u/loongforge"><img src="https://img.shields.io/badge/Docker-loongforge-2496ED?logo=docker&logoColor=white" alt="Docker Hub 镜像"></a>
  <a href="./CONTRIBUTING.md"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen?logo=github&logoColor=white" alt="欢迎 PR"></a>
  <a href="https://github.com/baidu-baige/LoongForge/issues/80"><img src="https://img.shields.io/badge/WeChat-Group-07C160?logo=wechat&logoColor=white" alt="加入微信群"></a>
  <a href="https://discord.gg/RnY39D6CM"><img src="https://img.shields.io/badge/Discord-Join_Us-5865F2?logo=discord&logoColor=white" alt="加入 Discord 社区"></a>
</p>

<p align="center">
  <a href="#performance"><img src="https://img.shields.io/badge/⚡_Speedup-up_to_5.04x-3B4FD8" alt="训练吞吐相比开源基线最高提升 5.04 倍"></a>
  <a href="#models"><img src="https://img.shields.io/badge/📦_Models-40%2B_ready_to_run-7C3AED" alt="40+ 开箱即用的模型示例"></a>
  <img src="https://img.shields.io/badge/🖥_Hardware-NVIDIA%2BKunlun-EC4899" alt="同时支持 NVIDIA GPU 与昆仑芯 XPU">
  <img src="https://img.shields.io/badge/🏭_Production-5000%2B_XPUs-DB2777" alt="生产验证，最大规模 5,000+ XPU">
</p>

<p align="center">
  <a href="https://loongforge.readthedocs.io/zh-cn/latest/index.html"><b>📖 文档</b></a>
  &nbsp;·&nbsp;
  <a href="#quickstart"><b>⚡ 快速开始</b></a>
  &nbsp;·&nbsp;
  <a href="./examples/"><b>🧪 训练示例</b></a>
  &nbsp;·&nbsp;
  <a href="#performance"><b>📊 性能表现</b></a>
  &nbsp;·&nbsp;
  <a href="#models"><b>🏛️ 支持模型</b></a>
  &nbsp;·&nbsp;
  <a href="https://baidu-baige.github.io/LoongForge/"><b>🌐 官网</b></a>
</p>

---

<h6 align="left">示例：LoongForge 加速具身模型训练，DreamZero 吞吐达基线的 4.38 倍，loss 曲线保持对齐</h6>

<p align="center">
  <a href="https://baidu-baige.github.io/LoongForge/assets/video/dreamzero-comparison.mp4">
    <picture>
      <source media="(prefers-reduced-motion: reduce)" srcset="./docs/assets/images/demo/dreamzero-poster.jpg">
      <img alt="DreamZero 训练左右对照：LoongForge 吞吐达到基线的 4.38 倍，训练 loss 曲线保持对齐" src="./docs/assets/images/demo/dreamzero-loop.webp" width="100%" />
    </picture>
  </a>
</p>

</div>

## 💡 为什么选择 LoongForge？

**LoongForge** 是一款开源训练框架，由百度智能云 [百舸团队](https://cloud.baidu.com/product/aihc.html) 开发，旨在为主流 **LLM、VLM、Diffusion 与具身模型** 提供[更快的训练速度](#performance)，从而显著降低成本。

- **易用** —— 为支持的每个模型提供[开箱即用的配置](./configs/models/)与[启动示例](./examples)，整体覆盖预训练、持续预训练、SFT 与 LoRA 等训练范式。
- **高性能** —— 采用多后端架构（Megatron-LM 与 torch-native）构建，并针对不同类别的模型，从并行策略、显存优化、通信隐藏、算子效率等维度做**深度优化**，同时保证**训练 loss 曲线与基线对齐**。
- **源自生产** —— 由 [AIAK-Training-LLM](https://cloud.baidu.com/doc/AIHC/s/Alyo476jr) 开源而来，一套服务于教育、计算机视觉与具身智能等领域企业客户的训练加速套件，**最大生产规模达 5,000+ XPU**。

> 🐉 LoongForge 名字源于中国传统 **龙舟**，象征协同发力与破浪前行。

<a id="quickstart"></a>
## ⚡ 快速开始

从下面四个案例中选择一个，按 **准备环境 → 准备权重和数据 → 启动训练** 的顺序开始。每个案例都列出了所需输入和对应的仓库脚本。

### 1. 准备环境

**NVIDIA GPU：** 需要 Linux、兼容的 NVIDIA 驱动、Docker 和 NVIDIA Container Toolkit。使用当前代码构建统一镜像，让代码与依赖保持匹配：

```bash
git clone --recurse-submodules https://github.com/baidu-baige/LoongForge.git
# 在 LoongForge 的父目录运行构建命令。
docker build --build-arg COMPILE_ENV=hopper \
  -t loongforge:local -f LoongForge/docker/Dockerfile .
mkdir -p loongforge-workspace
docker run --gpus all --ipc=host -it --rm \
  -v "$(pwd)/loongforge-workspace:/workspace/workspace" \
  -w /workspace/LoongForge loongforge:local bash
```

根据 GPU 架构将 `COMPILE_ENV` 设为 `ampere`、`hopper` 或 `blackwell`。也可以使用带版本号的[预构建镜像](https://hub.docker.com/u/loongforge)，配合该镜像内附带的训练脚本使用。

源码安装请按[安装指南](https://loongforge.readthedocs.io/zh-cn/latest/get_started/installation.html)准备 PyTorch、TransformerEngine 补丁版本及模型专用依赖。

**昆仑芯 XPU：** 请使用 [XPU 安装指南](https://loongforge.readthedocs.io/zh-cn/latest/kunlun_tutorial/install_p800.html)和 [`examples_xpu/`](./examples_xpu/) 中的脚本。

后续命令均在**容器内的 `/workspace/LoongForge`** 执行；源码安装用户请在仓库根目录执行，并将 `RUN_ROOT` 改为可写的工作目录。挂载的工作目录用于保留下载文件、日志和 checkpoint，退出容器后仍可访问。

```bash
export LOONGFORGE_PATH="$PWD"
export MEGATRON_PATH="$LOONGFORGE_PATH/third_party/Loong-Megatron"
export PYTHONPATH="$MEGATRON_PATH:$LOONGFORGE_PATH:${PYTHONPATH:-}"
export RUN_ROOT=/workspace/workspace
mkdir -p "$RUN_ROOT"
```

### 2. 选择一个案例

以下均为**单机 8 卡配置**。显存需求随模型、序列长度和 batch size 变化；更换运行规模前请检查启动脚本。两个 Qwen 脚本在文件内设置 `GPUS_PER_NODE=8`；Pi0.5 支持通过环境变量 `GPUS_PER_NODE` 调整卡数，Wan 使用 `GPUS`。

| 任务 | 本页案例 | 所需输入 |
| --- | --- | --- |
| LLM 指令微调 | [Qwen2.5-0.5B](#example-llm) | HF 权重 → MCore checkpoint；Alpaca JSON |
| VLM 指令微调 | [Qwen2.5-VL-3B](#example-vlm) | HF 权重 → MCore checkpoint；带 Energon 元数据的图文 WebDataset |
| VLA 微调 | [Pi0.5](#example-vla) | Pi0.5 权重；PaliGemma tokenizer；LeRobot v3.0 数据 |
| 视频扩散模型训练 | [Wan2.2 I2V-A14B](#example-diffusion) | 高噪声 / 低噪声 MCore checkpoint；预处理后的视频 latent |

<a id="example-llm"></a>
### 示例 A：Qwen2.5-0.5B 指令微调

下载模型及 Alpaca 格式数据集（包含 `instruction`、`input`、`output` 字段）。Tokenizer 已包含在模型目录中：

```bash
hf download Qwen/Qwen2.5-0.5B-Instruct \
  --local-dir "$RUN_ROOT/models/Qwen2.5-0.5B-Instruct"
hf download yahma/alpaca-cleaned --repo-type dataset \
  --include alpaca_data_cleaned.json --local-dir "$RUN_ROOT/data/alpaca"
```

将 HF 权重转换为 MCore 格式，张量并行和流水线并行均设为 1，与训练脚本保持一致：

```bash
python tools/convert_checkpoint/module_convertor/model.py \
  --load_platform=huggingface --save_platform=mcore \
  --config_file configs/models/qwen2.5/qwen2_5_0_5b.yaml \
  --convert_file configs/models/qwen2.5/ckpt_convert/qwen2_5_convert_llm.yaml \
  --tensor_model_parallel_size=1 --pipeline_model_parallel_size=1 \
  --load_ckpt_path "$RUN_ROOT/models/Qwen2.5-0.5B-Instruct" \
  --save_ckpt_path "$RUN_ROOT/checkpoints/qwen2.5-0.5b" \
  --safetensors --no_save_optim --no_load_optim

DATA_PATH="$RUN_ROOT/data/alpaca/alpaca_data_cleaned.json" \
TOKENIZER_PATH="$RUN_ROOT/models/Qwen2.5-0.5B-Instruct" \
CHECKPOINT_PATH="$RUN_ROOT/checkpoints/qwen2.5-0.5b" \
TENSORBOARD_PATH="$RUN_ROOT/logs/qwen2.5-0.5b" \
bash examples/qwen2.5/finetuning/sft_qwen2.5_0.5b.sh
```

脚本运行 5,000 步，每 500 步保存到 `CHECKPOINT_PATH`，TensorBoard 日志写入 `TENSORBOARD_PATH`。如需短跑验证，请修改脚本中的 `--train-iters`、`--lr-decay-iters` 和 `--save-interval`。自定义数据字段与 packing 的用法见 [LLM SFT 教程](https://loongforge.readthedocs.io/zh-cn/latest/llm_tutorial/quick_start_llm_sft.html)。

<a id="example-vlm"></a>
### 示例 B：Qwen2.5-VL-3B 图文指令微调

<details>
<summary><b>准备图文数据并启动 VLM SFT</b></summary>

准备 `Qwen/Qwen2.5-VL-3B-Instruct` 权重，将图文对话转换为带 Energon 元数据的 WebDataset 分片。[VLM 数据指南](https://loongforge.readthedocs.io/zh-cn/latest/vlm_tutorial/dataset_conversion.html)提供了原始数据的转换步骤。

修改 [Qwen2.5-VL 权重转换脚本](./examples/qwen2.5_vl/checkpoint_convert/convert_qwen2.5_vl_3b_hf_to_mcore.sh)中的 `LOAD` 和 `SAVE`，分别指向 HF 输入目录和 MCore 输出目录，保留 `ETP=1`、`DTP=1`、`PP=1`。然后执行：

```bash
bash examples/qwen2.5_vl/checkpoint_convert/convert_qwen2.5_vl_3b_hf_to_mcore.sh

# 将三个输入路径替换为上面准备好的资源。
DATA_PATH=/path/to/multimodal_wds \
TOKENIZER_PATH=/path/to/Qwen2.5-VL-3B-Instruct \
CHECKPOINT_PATH=/path/to/qwen2.5-vl-3b-mcore \
TENSORBOARD_PATH="$RUN_ROOT/logs/qwen2.5-vl-3b" \
bash examples/qwen2.5_vl/sft/sft_qwen2_5_vl_3b.sh
```

该配置冻结图像编码器，使用 `CHECKPOINT_PATH` 加载和保存权重。请按任务修改脚本中的 `--train-iters`、`--lr-decay-iters` 和 `--save-interval`（默认分别为 50,000 / 50,000 / 10,000,000）。对话格式与 packing 的用法见 [VLM SFT 教程](https://loongforge.readthedocs.io/zh-cn/latest/vlm_tutorial/quick_start_vlm_sft.html)。

</details>

<a id="example-vla"></a>
### 示例 C：在 LIBERO 上微调 Pi0.5

<details>
<summary><b>下载 LeRobot 数据，运行 20 步 VLA 示例</b></summary>

Pi0.5 直接加载 HF 权重，在线读取 LeRobot v3.0 数据。下载权重、tokenizer 和数据集；PaliGemma tokenizer 需要先取得模型访问权限并执行 `hf auth login`：

```bash
hf download lerobot/pi05_base --local-dir "$RUN_ROOT/models/pi05_base"
hf download google/paligemma-3b-pt-224 \
  --include 'tokenizer*' 'special_tokens_map.json' 'added_tokens.json' 'config.json' \
  --local-dir "$RUN_ROOT/models/paligemma-3b-pt-224"
hf download lerobot/libero_10 --repo-type dataset \
  --local-dir "$RUN_ROOT/data/libero_10"

TOKENIZER_PATH="$RUN_ROOT/models/paligemma-3b-pt-224" \
CHECKPOINT_PATH="$RUN_ROOT/models/pi05_base" \
DATA_PATH="$RUN_ROOT/data/libero_10" \
OUTPUT_DIR="$RUN_ROOT/outputs/pi05" \
GPUS_PER_NODE=8 TRAIN_ITERS=20 SAVE_INTERVAL=20 \
bash examples/embodied/pi05/run_pi05_ddp_finetune.sh
```

这次短跑用于检查训练流程；正式微调时增大 `TRAIN_ITERS`。训练产物写入 `OUTPUT_DIR`，TensorBoard 日志位于 `OUTPUT_DIR/tensorboard`。[Pi0.5 教程](https://loongforge.readthedocs.io/zh-cn/latest/embodied_tutorial/quick_start_pi05.html)介绍了 FSDP、ZeRO-1 和策略评测；世界-动作模型可从 [DreamZero 教程](https://loongforge.readthedocs.io/zh-cn/latest/embodied_tutorial/quick_start_dreamzero.html)开始。

</details>

<a id="example-diffusion"></a>
### 示例 D：Wan2.2 图生视频训练

<details>
<summary><b>预处理视频数据并启动扩散模型训练</b></summary>

准备 `Wan-AI/Wan2.2-I2V-A14B` 权重，以及包含 `video,prompt` 两列的 `metadata.csv` 和视频文件。按 [Wan 准备指南](https://loongforge.readthedocs.io/zh-cn/latest/wan_tutorial/quick_start_wan_training.html)安装预处理依赖（`diffsynth==1.1.8`），并将高噪声和低噪声模型权重分别转换为 MCore。随后预处理视频并启动：

```bash
LOONGFORGE_ROOT="$LOONGFORGE_PATH" \
DATASET_BASE_PATH=/path/to/video_dataset \
DATASET_METADATA_PATH=/path/to/video_dataset/metadata.csv \
WAN22_MODEL_ROOT=/path/to/Wan2.2-I2V-A14B \
bash examples/wan/preprocess.sh wan2.2 "$RUN_ROOT/data/wan-preprocessed"

DATASET_PATH="$RUN_ROOT/data/wan-preprocessed" \
HIGH_NOISE_CHECKPOINT_PATH=/path/to/high_noise_mcore \
LOW_NOISE_CHECKPOINT_PATH=/path/to/low_noise_mcore \
TENSORBOARD_PATH="$RUN_ROOT/logs/wan2.2" \
GPUS=8 \
bash examples/wan/pretrain_wan2.2_i2v_a14b.sh
```

脚本依次训练高噪声和低噪声模型，分别使用各自的 checkpoint 目录加载和保存权重。请按任务修改脚本中的训练步数和保存间隔。

</details>

### 3. 扩展到自己的任务

| 想做什么 | 从哪里开始 |
| --- | --- |
| 使用自己的语料预训练 | [LLM 预训练](https://loongforge.readthedocs.io/zh-cn/latest/llm_tutorial/quick_start_llm_pretrain.html)或 [VLM 预训练](https://loongforge.readthedocs.io/zh-cn/latest/vlm_tutorial/quick_start_vlm_pretrain.html) |
| 更换模型或并行策略 | 查看 [`examples/`](./examples/) 及对应的 [`configs/models/`](./configs/models/)；并行策略和 batch 参数在启动脚本中 |
| 加载或导出模型权重 | [`tools/convert_checkpoint/`](./tools/convert_checkpoint/) 及各模型的 `checkpoint_convert/` 示例 |
| 在昆仑芯 XPU 上训练 | [XPU 教程](https://loongforge.readthedocs.io/zh-cn/latest/kunlun_tutorial/README.html)及 [`examples_xpu/`](./examples_xpu/) |

<a id="performance"></a>
## 📊 性能表现

各模型相对主流开源基线的训练吞吐加速——每个模型与其基线的对比，均在相同机型与相同训练超参数下测得：

<p align="center">
  <img alt="LoongForge 相对开源基线的训练吞吐加速——从 Qwen3-VL 的 1.45 倍到 DeepSeek-V3.2 Lite 的 5.04 倍" src="./docs/assets/images/benchmark_speedup.png" width="860" />
</p>

> DeepSeek-V3.2 Lite 为 DSA 算子级优化的结果，受测试环境规模限制，在减层配置下验证。<br>
> 数据为特定时间点的测试结果，双方实现持续演进，数值可能随之变化。

## 🏗️ 架构

由于不同类别、不同规模的模型，最优训练策略不同，LoongForge 采用多后端架构。

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)"  srcset="./docs/assets/images/architecture/loongforge-architecture-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="./docs/assets/images/architecture/loongforge-architecture.svg">
    <img alt="LoongForge 架构：面向 LLM / VLM / 扩散模型的 Megatron 栈，与面向具身模型的 torch-native 栈" src="./docs/assets/images/architecture/loongforge-architecture.svg" width="100%">
  </picture>
</p>

- **Megatron 栈** —— 面向 LLM、VLM 与 Diffusion 模型。基于 [patch 过的 Megatron-LM](https://github.com/baidu-baige/Loong-Megatron) 构建，并扩展了 MoE 并行、组件级异构并行、长序列优化等能力。
- **Torch-Native 栈** —— 面向具身模型（VLA 与 WAM）。独立的 [torch-native 子系统](./loongforge/embodied)，支持 **DDP / ZeRO-1 / FSDP / HSDP**，并针对典型模型做了深度性能优化，涵盖 I/O、通信策略、kernel 效率等。

## ✨ 核心特性

**🚀 基座模型**

* **MoE EP 通信优化** —— All2All / 激活卸载 / 计算全链路重叠，在 DeepSeek-V3、Qwen3-MoE 等模型上相对上游 Megatron-LM 实现**进一步显存降低**。
* **MoE 专家负载均衡** —— 基于拓扑感知算法动态复制热点专家，均衡专家并行（EP）负载，开销相较业界方案最大可降低 **74%**。[[TAOT 论文](https://arxiv.org/pdf/2608.03676)]
* **自适应 FP8 训练** —— 面向 LLM 和 VLM 的端到端 FP8，支持标准 **blockwise FP8**；可选**自适应**模式根据 GEMM 形状与效率逐算子选择最佳精度。
* **自定义融合算子** —— 为 DSA 类模型设计的 **FusedDSA** 等融合 Kernel —— TileLang 版本已开源，高性能 CUDA 版本在百度百舸平台提供。
* **长序列训练** —— 基于**上下文并行（CP）**与**分块流水线调度**，将 LLM 训练扩展至长序列场景。

**🧩 多模态模型**

* **灵活的多模态组合** —— 通过配置即可将可互换的 ViT 与 LLM 组件自由组装为 VLM（如 **GLM-5.2 + MoonViT**），无需编写模型代码。
* **异构并行** —— 针对模型不同组件（如 ViT vs LLM）独立配置 TP / DP / 重计算 / 冻结策略，获得最优吞吐与显存占用。[[blog](https://baidu-baige.github.io/LoongForge/blog/2026-05-loongforge-heterogeneous-parallel-training.html)]
* **Encoder-Decoder 解耦训练** —— 将 ViT 与 LLM 拆分为独立任务，消除 Encoder 带来的流水线气泡。
* **DP 负载均衡** —— 基于负载感知的数据重分发，缓解序列打包不均衡问题，显著提升多节点扩展效率。[[blog](https://baidu-baige.github.io/LoongForge/blog/2026-05-loongforge-dp-load-balancing.html)]

**🤖 具身模型**

* **VLA 与 WAM 训练** —— 面向 **VLA 与世界-动作模型（WAM）** 的独立 **torch 原生 DDP/FSDP** 子系统，与 Megatron 核心解耦，支持 **DDP / ZeRO-1 / FSDP / HSDP** 多种分布式策略。[[README](./loongforge/embodied)]
* **FP8 通信优化** —— 在支持的 NVIDIA GPU 上，用 FP8 压缩跨卡通信量的可选优化，覆盖两种并行策略：**FSDP2** 场景对参数 AllGather 做按 block 的 FP8 delta 压缩，**DDP** 场景做 FP8 梯度 all-reduce。[[使用方法](./docs/source_zh/features/fp8_communication.md)]
* **逐模型深度定制优化** —— 针对当前覆盖的每个模型深度优化训练代码，涵盖 I/O、通信策略、算子效率等维度，实测相对官方基线 **1.79×–4.38× 加速**（见[性能表现](#performance)）。
* **统一评测** —— 在 **LIBERO / CALVIN / SimplerEnv / RoboTwin** 上评测训练出的策略，覆盖度持续完善。

**🧰 工作流与兼容性**

* **丰富的流水线与数据工具** —— 开箱即用的 **Pretrain / MidTrain / SFT / LoRA** 流水线，内置数据集格式转换与序列打包能力。
* **灵活的 Checkpoint 机制** —— 支持离线 **Megatron ↔ HuggingFace** 双向转换，以及在线原生 HF 加载/保存，全流程无格式壁垒。
* **异构硬件** —— 通过轻侵入式插件设计，原生支持 **NVIDIA GPU** 与**昆仑芯 XPU**。

> 📖 深入阅读：[LLM](https://loongforge.readthedocs.io/zh-cn/latest/llm_tutorial/features_index.html) · [VLM](https://loongforge.readthedocs.io/zh-cn/latest/vlm_tutorial/features_index.html) · [具身模型](https://loongforge.readthedocs.io/zh-cn/latest/embodied_tutorial/overview.html)

<a id="models"></a>
## 🏛️ 支持的模型

下表概览了各类别的代表模型；展开完整列表可进入模型对应的训练示例。各变体与训练能力见[模型支持矩阵](https://loongforge.readthedocs.io/zh-cn/latest/get_started/support_model.html)。

| 类别 | 代表模型 |
| --- | --- |
| LLM | DeepSeek、Qwen、LLaMA、MiniMax、GLM、Kimi |
| VLM | Qwen-VL、Qwen3.5–3.8、Kimi、MiniCPM-V、InternVL、LLaVA、GLM + MoonViT |
| Diffusion | Wan2.1 / 2.2、Qwen-Image-Edit |
| 具身模型 | Pi0.5、GR00T、xVLA、Wall-OSS、FastWAM、LingBot-VA、Cosmos3、DreamZero |

<details>
<summary><b>查看模型矩阵</b></summary>

<table width="100%">
<colgroup>
<col width="25%">
<col width="25%">
<col width="25%">
<col width="25%">
</colgroup>
<thead align="center" valign="bottom">
<tr><th width="25%">LLM</th><th width="25%">VLM</th><th width="25%">Diffusion</th><th width="25%">Embodied</th></tr>
</thead>
<tbody valign="top">
<tr>
<td valign="top">
<ul>
<li><a href="examples/deepseek_v2/">DeepSeek-V2</a> ✅</li>
<li><a href="examples/deepseek_v3/">DeepSeek-V3/V3.2</a> ✅</li>
<li><a href="examples/deepseek_v4/">DeepSeek-V4</a> ✅</li>
<li><a href="examples/llama2/">LLaMA2</a> ✅</li>
<li><a href="examples/llama3/">LLaMA3</a> ✅</li>
<li><a href="examples/llama3.1/">LLaMA3.1</a> ✅</li>
<li><a href="examples/qwen/">Qwen</a> ✅</li>
<li><a href="examples/qwen1.5/">Qwen1.5</a> ✅</li>
<li><a href="examples/qwen2/">Qwen2</a> ✅</li>
<li><a href="examples/qwen2.5/">Qwen2.5</a> ✅</li>
<li><a href="examples/qwen3/">Qwen3</a> ✅</li>
<li><a href="examples/qwen3_next/">Qwen3-Next</a> ✅</li>
<li><a href="examples/minimax/">MiniMax-M2.1/2.5/2.7</a> ✅</li>
<li><a href="examples/mimo/">MIMO</a> ✅</li>
<li><a href="examples/glm5/">GLM-5</a> ✅</li>
<li><a href="examples/glm5.2/">GLM-5.2</a> ✅</li>
<li><a href="examples/kimi_k3/">Kimi-K3</a> ✅</li>
</ul>
</td>
<td valign="top">
<ul>
<li><a href="examples/qwen2.5_vl/">Qwen2.5-VL</a> ✅</li>
<li><a href="examples/qwen3_vl/">Qwen3-VL</a> ✅</li>
<li><a href="examples/qwen3.5/">Qwen3.5</a> ✅</li>
<li><a href="examples/qwen3.6/">Qwen3.6</a> ✅</li>
<li><a href="examples/qwen3.8/">Qwen3.8</a> ✅</li>
<li><a href="examples/kimi_k2.x/kimi_k2.5/">Kimi-K2.5/2.6</a> ✅</li>
<li><a href="examples/kimi_k3/">Kimi-K3</a> ✅</li>
<li><a href="examples/minicpm_v_4_6/">MiniCPM-V-4.6</a> ✅</li>
<li><a href="examples/glm5.2_vit/">GLM-5.2 + MoonViT</a> ✅</li>
<li><a href="examples/glm5_next/">GLM-5.3-flash</a> ✅</li>
<li><a href="examples/ernie4.5/">ERNIE4.5-VL</a> ✅</li>
<li><a href="examples/llava_onevision_1.5/">LLaVA-OneVision-1.5</a> ✅</li>
<li><a href="examples/internvl2.5/">InternVL2.5</a> ✅</li>
<li><a href="examples/internvl3.5/">InternVL3.5</a> ✅</li>
<li><a href="examples/custom/">CustomCombinedModel Example</a> ✅</li>
</ul>
</td>
<td valign="top">
<ul>
<li><a href="examples/wan/">Wan2.1</a> ✅</li>
<li><a href="examples/wan/">Wan2.2</a> ✅</li>
<li><a href="examples/qwen_image/">Qwen-Image-Edit-2511</a> ✅</li>
</ul>
</td>
<td valign="top">
<ul>
<li><a href="examples/embodied/pi05/">Pi0.5</a> ✅</li>
<li><a href="examples/embodied/groot_n1_6/">GR00T-N1.6</a> ✅</li>
<li><a href="examples/embodied/groot_n1_7/">GR00T-N1.7</a> ✅</li>
<li><a href="examples/embodied/xvla/">xVLA</a> ✅</li>
<li><a href="examples/embodied/wall_oss_0_5/">Wall-OSS-0.5</a> ✅</li>
<li><a href="examples/embodied/fastwam/">FastWAM</a> ✅</li>
<li><a href="examples/embodied/lingbot_va/">LingBot-VA</a> ✅</li>
<li><a href="examples/embodied/cosmos3/">Cosmos3</a> ✅</li>
<li><a href="examples/embodied/dreamzero/">DreamZero</a> ✅</li>
</ul>
</td>
</tr>
</tbody>
</table>

</details>

## 🌟 基于 LoongForge 训练

基于 LoongForge 或其前身 AIAK-Training-LLM 训练的开源模型：

| 模型 | 亮点 |
|------|------|
| [**LLaVA-OneVision-2.0**](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2) | 新一代多模态模型，配套全新的 VideoCaption 与 Spatial 数据集 |
| [**Innovator-VL**](https://github.com/InnovatorLM/Innovator-VL/tree/main) | 面向高级推理的科学多模态大模型 |
| [**LLaVA-OneVision-1.5**](https://github.com/EvolvingLMMs-Lab/LLaVA-OneVision-2/tree/1.5) | 面向多模态训练民主化的全开源框架 |
| [**Qianfan-VL**](https://github.com/baidubce/Qianfan-VL) | 面向企业的领域增强视觉-语言模型，参数量覆盖 3B ~ 70B |

## 📂 代码结构

<details>
<summary><b>📁 目录树</b></summary>

```
LoongForge/
├── loongforge/                   # 核心训练框架
│   ├── train/                    # 训练入口与训练器
│   │   ├── pretrain/             #   预训练（LLM、VLM）
│   │   ├── sft/                  #   SFT（LLM、VLM、InternVL、ERNIE）
│   │   └── diffusion/            #   Diffusion（WAN、Qwen-Image）
│   ├── models/                   # 统一的模型抽象层
│   │   ├── foundation/           #   LLM 主干（LLaMA、Qwen、DeepSeek、...）
│   │   ├── encoder/              #   视觉编码器（ViT、Qwen-VL、InternVL、...）
│   │   ├── omni_models/          #   多模态组合
│   │   ├── diffusion/            #   Diffusion 模型（WAN、Qwen-Image）
│   │   └── common/               #   公共 Layer 与工具
│   ├── embodied/                 # LoongForge-Embodied：独立的 torch-native（DDP/FSDP）具身
│   │                             #   （VLA + 世界-动作）训练子系统，详见 loongforge/embodied/README_zh.md
│   ├── data/                     # 数据流水线（多模态、视频、DP 负载均衡）
│   ├── tokenizer/                # Tokenizer
│   └── utils/                    # 配置映射、常量等
├── third_party/Loong-Megatron/   # Patched Megatron-LM（git submodule）
├── configs/                      # Hydra YAML 配置（模型、数据）
├── examples/                     # GPU 启动脚本
├── examples_xpu/                 # 昆仑芯 XPU 启动脚本
├── tools/                        # Checkpoint 转换、数据预处理
├── ops/                          # 自定义融合算子（含开源的 TileLang 版本）
├── patches/                      # TransformerEngine 补丁
├── docker/                       # Dockerfile（GPU & XPU）
├── tests/                        # 端到端测试（YAML 驱动）
└── docs/                         # 文档
```

</details>

## 🔥 最新动态

- **[2026/09]** ✨ 新增 **[GLM-5.3-flash](./examples/glm5_next/)** 训练支持。
- **[2026/09]** ✨ 新增 **[Kimi-K3](./examples/kimi_k3/)** 的 LLM 与 VLM BF16 训练支持。
- **[2026/09]** ⚡ 新增优化后的 **[DreamZero Wan2.2-5B FSDP recipe](./examples/embodied/dreamzero/run_dreamzero_wan22_5b_full_fsdp_finetune.sh)**，集成 cache-aware 数据加载、attention block 编译、冻结模块处理与 FSDP2 Delta-FP8 Param AllGather。

<details>
<summary><b>📅 更多</b></summary>

- **[2026/08]** 🤖 新增 **[Wall-OSS-0.5](./examples/embodied/wall_oss_0_5/)** VLA 训练支持，并通过自定义融合算子提升训练吞吐。
- **[2026/08]** 📄 发布 **[TAOT 论文](https://arxiv.org/abs/2608.03676)** —— 通过拓扑感知的动态专家副本放置，优化 **MoE** 训练中的专家并行（**EP**）负载不均衡，相较业界方案开销最大可降低 **74%**，案例实测 **1.43× 加速**。[[blog](https://baidu-baige.github.io/LoongForge/blog/2026-08-taot-topology-aware-expert-placement.html)]
- **[2026/08]** ✨ 新增 **GLM-5.2** 训练支持，并提供 **[GLM-5.2 + MoonViT](./configs/models/glm5.2_vit/)** 自定义组合[示例](./examples/glm5.2_vit/)，可用于为 GLM 扩展多模态能力。
- **[2026/08]** ✨ 新增 **MiniCPM-V-4.6** 与 **Qwen3.8-27B** 训练支持。
- **[2026/08]** 🧪 Embodied 栈新增统一[**评测模块**](./loongforge/embodied/eval/)，当前已覆盖 **Pi0.5 / xVLA / GR00T**，持续扩展中。
- **[2026/07]** 🐳 统一**预构建 Docker 镜像** —— LLM / VLM / VLA / Diffusion 全部模型家族共用同一镜像。
- **[2026/07]** 🤖 发布 **[LoongForge-Embodied](./loongforge/embodied)** —— 面向具身模型（Pi0.5、GR00T-N1.6/N1.7、xVLA、LingBot-VA、FastWAM、DreamZero、Cosmos3）的 torch-native DDP/FSDP 训练子系统，实测最高 **4.38× 加速**。[[blog](https://baidu-baige.github.io/LoongForge/blog/2026-07-announcing-loongforge-embodied.html)]
- **[2026/07]** ✨ 新增 **Qwen-Image-Edit-2511** 训练支持。
- **[2026/07]** ✨ 新增 **DeepSeek-V4-Flash / DeepSeek-V4-Pro** 训练支持。
- **[2026/06]** 🤖 扩展 VLA 模型覆盖，新增 **GR00T N1.6**；GR00T 训练实现 **2.3× 加速**。[[blog](https://baidu-baige.github.io/LoongForge/blog/2026-06-loongforge-groot-n16-acceleration.html)]
- **[2026/05]** ⚡ **Wan 2.2** 训练 **加速 116%**，并新增 CP（上下文并行）与数据 packing 策略支持。
- **[2026/05]** ✨ 新增 **Kimi K2.5 / K2.6** 训练支持，并支持 **INT4 / NVFP4** PTQ 量化能力。
- **[2026/05]** 🎉 **v0.1.0** —— LoongForge 首个正式版本发布。
- **[2026/05]** 🌟 支持 **LLaVA-OneVision-2.0** 模型训练并协助其公开发布。
- **[2026/04]** 🧩 新增 **MiniMax-M2.7** 在 NVIDIA GPU 与昆仑芯 XPU 上的训练支持。
- **[2026/04]** 🚀 LoongForge 源码在 GitHub 上正式公开。[[blog](https://zhuanlan.zhihu.com/p/2031006068797600446)]
- **[2025/10]** 🌟 基于AIAK-Training-LLM（LoongForge 前身）支持 **LLaVA-OneVision-1.5** 模型训练并协助其公开发布。[[blog](https://mp.weixin.qq.com/s/1y7Br15pBpUZ-90j5OGncA)]

</details>

## 📝 引用

如果您觉得 LoongForge 对您的工作有帮助，请引用本项目：

```bibtex
@software{LoongForge2026,
  title  = {LoongForge: A high-performance framework for training LLMs, VLMs, diffusion, and embodied models},
  author = {{The LoongForge Authors}},
  year   = {2026},
  url    = {https://github.com/baidu-baige/LoongForge}
}
```

如果您在 LoongForge 中使用 TAOT 进行 MoE 训练，可以引用我们的论文：

```bibtex
@article{zhang2026taot,
  title   = {{TAOT}: Topology-Aware Optimal Transport for Dynamic Expert Replica Placement in {MoE} Training},
  author  = {Zhang, Lingyun and Zhang, Henghua and Gu, Shilei and Mo, Kai and Han, Shuai and Li, Shiyong and Wang, Yanpeng and Shen, Dou},
  journal = {arXiv preprint arXiv:2608.03676},
  year    = {2026},
  url     = {https://arxiv.org/abs/2608.03676}
}
```

## 🤝 参与贡献

我们非常欢迎社区贡献 —— 无论是 Bug 报告、功能提案还是 PR。在提交前请阅读 [贡献指南](./CONTRIBUTING.md)。

非常感谢 LoongForge 的所有贡献者：

<a href="https://github.com/baidu-baige/LoongForge/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=baidu-baige/LoongForge&v=2026-09-04" alt="LoongForge contributors" />
</a>

## 🙏 致谢

LoongForge 的构建离不开 NVIDIA 的 [Megatron-LM](https://github.com/NVIDIA/Megatron-LM)，并从 [HuggingFace Transformers](https://github.com/huggingface/transformers)、[LLaMA-Factory](https://github.com/hiyouga/LlamaFactory)、[Megatron-Bridge](https://github.com/NVIDIA-NeMo/Megatron-Bridge)、[LeRobot](https://github.com/huggingface/lerobot) 等优秀开源项目，以及所支持模型的官方实现（如 [OpenPI](https://github.com/Physical-Intelligence/openpi)、[NVIDIA Isaac GR00T](https://github.com/NVIDIA/Isaac-GR00T)）中汲取了灵感。衷心感谢这些社区所做的杰出贡献；同时也特别感谢 [LINUX DO](https://linux.do/) 社区，为技术交流提供了友善的空间，并对开源分享给予支持。

<a id="contact"></a>
## 💬 联系我们

- **GitHub Issue** —— 问题反馈、使用疑问与功能建议。[提交 Issue](https://github.com/baidu-baige/LoongForge/issues/new/choose)。
- **开发者社区** —— **微信群**、**小红书**等。[点此加入](https://github.com/baidu-baige/LoongForge/issues/80)。
- **邮件** —— 企业落地、大规模部署、商务合作，以及任何其他话题。[loongforge@baidu.com](mailto:loongforge@baidu.com)。

## 📄 开源协议

LoongForge 基于 [Apache License 2.0](./LICENSE) 发布。部分源文件改编自第三方开源项目，请以各文件头部标注的版权与署名信息为准。

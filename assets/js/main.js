// LoongForge site — shared interactions (with i18n + GitHub icon + theme)
(function () {
  'use strict';

  // ===== i18n =====
  const I18N = {
    en: {
      'nav.home': 'Home', 'nav.features': 'Features', 'nav.models': 'Models',
      'nav.docs': 'Docs', 'nav.blog': 'Blog', 'nav.about': 'About', 'nav.contact': 'Contact',
      'nav.github': 'Star',
      'footer.tagline': 'Modular, scalable training framework for LLM / VLM / VLA / Diffusion.',
      'footer.project': 'Project', 'footer.resources': 'Resources', 'footer.loong': 'Baige Loong Series',
      'footer.copyright': '© 2026 LoongForge Authors · Apache License 2.0 · Built with ♥ by the Baidu Baige Team',
      'footer.quickstart': 'Quick Start', 'footer.modelconfigs': 'Model Configs',
      'footer.examples': 'Examples', 'footer.contributing': 'Contributing',
      'footer.training': 'Training framework', 'footer.workflow': 'Agent framework',

      'hero.badge': '🐉 Part of the Baidu-Baige Loong open-source series',
      'hero.subtitle_html': 'A <b>modular</b>, <b>scalable</b>, <b>high-performance</b> training framework for <b>LLMs</b>, <b>VLMs</b>, <b>diffusion</b>, and <b>embodied</b> models — built on Megatron-LM with native NVIDIA GPU & Kunlun XPU support.',
      'hero.cta.start': '🚀 Get Started',
      'hero.cta.github': '⭐ View on GitHub',
      'hero.cta.docs': '📚 Read the Docs',
      'hero.stat.speedup': 'Max training speedup',
      'hero.stat.xpu': 'XPU production scale',
      'hero.stat.modal': 'LLM · VLM · VLA · Diffusion',
      'hero.stat.fp8': 'Adaptive FP8 precision',

      'hero.vp.1.k': 'Easy',
      'hero.vp.1.t': 'One framework, broad coverage',
      'hero.vp.1.d': 'Full coverage of mainstream open-source LLMs, VLMs, MoE, diffusion, and VLA models. Ready-to-run configs and launch scripts included.',
      'hero.vp.2.k': 'Efficient',
      'hero.vp.2.t': 'Up to ~5× training speedup',
      'hero.vp.2.d': 'Deep performance optimizations — fused kernels, adaptive FP8, MoE A2A overlap, and multimodal pipeline scheduling.',
      'hero.vp.3.k': 'Multi-chip',
      'hero.vp.3.t': 'NVIDIA GPU & Kunlun XPU',
      'hero.vp.3.d': 'Native heterogeneous hardware support — one framework, minimal migration between GPU and XPU.',

      'news.title': '🔥 Latest News', 'news.all': 'All posts →',

      'feat.title': '✨ Key Features',
      'feat.subtitle_html': 'A quick tour of what sets LoongForge apart.',

      'feat.cat.1.title': 'MoE',
      'feat.cat.1.subtitle': 'Communication, compute and offload, all in parallel.',
      'feat.cat.1.item.1.t': 'Tri-Stream Overlap',
      'feat.cat.1.item.1.d': 'MoE EP comm × compute × offload in parallel — higher throughput than upstream.',

      'feat.cat.2.title': 'Multimodal',
      'feat.cat.2.subtitle': 'Parallelism built for VLM / VLA.',
      'feat.cat.2.item.1.t': 'Model Composition',
      'feat.cat.2.item.1.d': 'Swap ViT × LLM for VLMs via YAML.',
      'feat.cat.2.item.2.t': 'Heterogeneous Parallelism',
      'feat.cat.2.item.2.d': 'Independent TP / PP / DP per model component.',
      'feat.cat.2.item.3.t': 'Disaggregated Training',
      'feat.cat.2.item.3.d': 'Decoupled ViT / LLM scheduling kills pipeline bubbles.',
      'feat.cat.2.item.4.t': 'DP Load Balancing',
      'feat.cat.2.item.4.d': 'Fixes packing-induced imbalance at cluster scale.',

      'feat.cat.3.title': 'Performance',
      'feat.cat.3.subtitle': 'Precision, kernels, long-sequence throughput.',
      'feat.cat.3.item.1.t': 'Adaptive FP8',
      'feat.cat.3.item.1.d': 'Per-operator FP8 decisions by GEMM shape.',
      'feat.cat.3.item.2.t': 'Fused Operators',
      'feat.cat.3.item.2.d': 'FusedDSA / Sparse MLA kernels for end-to-end speedup.',

      'feat.cat.4.title': 'Performance',
      'feat.cat.4.item.1.t': 'ChunkPipe',
      'feat.cat.4.item.1.d': 'Chunked long-sequence pipelining toward million-length contexts.',

      'feat.cat.5.title': 'Usability',
      'feat.cat.5.subtitle': 'Plug into the HuggingFace + Megatron world without friction.',
      'feat.cat.5.item.1.t': 'HF ↔ Megatron',
      'feat.cat.5.item.1.d': 'Bidirectional checkpoint conversion + online HF load/save.',

      'feat.cat.6.title': 'Training',
      'feat.cat.6.item.1.t': 'Pretrain + SFT + LoRA',
      'feat.cat.6.item.1.d': 'One codebase covers key training stages.',


      'models.title': '🏛️ Supported Models',
      'models.subtitle': 'From 0.5B-parameter SLMs to 480B MoE giants — all batteries-included.',
      'models.more.t': 'and more…',
      'models.more.d': 'Adding new architectures is as easy as a YAML + registry entry.',
      'models.custom.t': 'CustomCombinedModel',
      'models.custom.d_html': 'Compose any ViT + any LLM backbone via a YAML file. <a href="https://github.com/baidu-baige/LoongForge/blob/master/configs/models/custom/qwen_vit_llama3_8b.yaml" target="_blank" class="text-indigo-500 underline">Example →</a>',
      'models.diff.d': 'Image-to-video diffusion with optimized kernels.',
      'models.vla.d': 'Vision-Language-Action model with torch.compile-enabled paligemma expert.',
      'models.groot.d': 'NVIDIA-style embodied VLA model, production-tuned for large-batch SFT.',

      'qs.title': '🚀 Quick Start',
      'qs.subtitle': 'YAML-driven, five steps from install to launch.',
      'qs.s1.tab': 'Install', 'qs.s2.tab': 'Compose', 'qs.s2.opt': 'optional', 'qs.s3.tab': 'Configure', 'qs.s4.tab': 'Weights', 'qs.s5.tab': 'Launch',
      'qs.s1.desc': 'Docker is the recommended path — a single image bundles CUDA/XPU toolchains, the patched Megatron submodule, and TransformerEngine, so every developer and every node trains from the same environment. Source install is also fully supported for advanced setups.',
      'qs.s1.note_html': 'For source install and the full build-arg matrix, see <a class="underline" href="https://loongforge.readthedocs.io/en/latest/get_started/installation.html" target="_blank" rel="noopener">NVIDIA install guide</a> · <a class="underline" href="https://loongforge.readthedocs.io/en/latest/kunlun_tutorial/install_p800.html" target="_blank" rel="noopener">Kunlun XPU install guide</a>.',
      'qs.s2.desc': 'Optional — only needed if you want a custom combination. VLMs in LoongForge are declarative: a YAML wires a <b>vision encoder</b>, <b>projector</b>, and <b>language backbone</b> together. Mainstream open-source models are already built in under <a class="underline" href="https://github.com/baidu-baige/LoongForge/tree/master/configs/models" target="_blank" rel="noopener">configs/models/</a> — and for VLMs you can swap any of the three components (e.g. replace the LLM backbone) with a single-line config change, no Python edits required.',
      'qs.s3.desc': 'The outer layer is fully Megatron-compatible, so familiar training args keep working. Inside, Hydra-style overrides let each model component pick its own parallelism (TP / PP / EP / DP) and freeze policy — ideal for heterogeneous VLM training where ViT and LLM backbones have very different compute profiles.',
      'qs.s4.desc': 'LoongForge is bidirectional with HuggingFace: load HF weights directly at startup (no manual conversion), keep training in Megatron-native format internally, and export back to HF format when the run finishes — so downstream serving stacks keep working unchanged.',
      'qs.s5.desc': 'Just run a ready-made script. 20+ model families ship with production-tuned launchers under <a class="underline" href="https://github.com/baidu-baige/LoongForge/tree/master/examples" target="_blank" rel="noopener">examples/</a> (NVIDIA GPU) and <a class="underline" href="https://github.com/baidu-baige/LoongForge/tree/master/examples_xpu" target="_blank" rel="noopener">examples_xpu/</a> (Kunlun XPU). The snippet below shows the core <code class="mono">torchrun</code> pattern used across all examples.',

      'powered.title': '🌟 Powered by LoongForge',
      'powered.subtitle': 'Open-source projects trained on LoongForge — ordered from newest to earliest.',
      'powered.new': 'NEW',
      'powered.1.t': 'LLaVA-OneVision-2.0',
      'powered.1.d': 'Next-generation fully open multimodal model — improved data, training recipe, and scaling.',
      'powered.2.t': 'LLaVA-OneVision-1.5',
      'powered.2.d': 'Fully open framework for democratized multimodal training.',
      'powered.3.t': 'Qianfan-VL',
      'powered.3.d': 'Domain-enhanced universal vision-language models.',

      'cta.title': 'Join the Community',
      'cta.desc': 'Report bugs, propose features, contribute code, or just say hi. We ❤️ community.',
      'cta.issue': 'Open an Issue', 'cta.pr': 'Send a PR', 'cta.about': 'About the Project',

      'docs.hero.title': 'Documentation',
      'docs.hero.desc_html': 'A curated index of installation, tutorials, and reference material. Full API & deep tutorials live on <a href="https://loongforge.readthedocs.io/en/latest/index.html" target="_blank" class="underline text-amber-300">ReadTheDocs</a>.',

      'blog.hero.title': 'Engineering Blog',
      'blog.hero.desc': 'Releases, performance deep-dives, and stories from the LoongForge team.',

      'about.hero.title': 'About LoongForge',
      'about.hero.desc': 'A training framework born from real-world, large-scale production workloads — and shared back with the community.',

      'bench.title': '📊 Benchmark',
      'bench.subtitle': 'Measured in v0.1.0 on A800 across LLM, VLM, and VLA workloads.',
      'bench.baseline': '1.0× baseline',
      'bench.ds.title': 'DeepSeek-V3.2 · DSA operator-level optimizations',
      'bench.ds.desc': 'Validated on reduced-layer configuration — enabling 64K sequence training.',
      'hw.title': '💎 Hardware Compatibility',
      'hw.subtitle': 'One codebase, two silicon stacks — production-ready on NVIDIA GPU and Baidu Kunlun XPU.',
      'hw.nv.t': 'NVIDIA GPU',
      'hw.nv.d': 'Built on the community Megatron + TransformerEngine ecosystem, with LoongForge optimizations layered on top.',
      'hw.xpu.t': 'Kunlun XPU',
      'hw.xpu.d': 'XPU Plugin mechanism shields the upper stack from adaptation differences, while integrating an XPU-specific optimization toolchain.',

      'comm.title': '🤝 Community',
      'comm.subtitle': 'Built in the open — join discussions, report issues, and contribute.',
      'comm.1.t': 'GitHub Issues', 'comm.1.d': 'File bug reports and feature requests.',
      'comm.2.t': 'Discussions',    'comm.2.d': 'Ask questions and share experiences.',
      'comm.3.t': 'Contributing',   'comm.3.d': 'Read the guide and send your first PR.',
      'comm.4.t': 'Join us on WeChat', 'comm.4.d': 'Scan the QR code in our README to join the developer group.',

      'stats.stars': 'GitHub Stars',
      'stats.contrib': 'Contributors',
      'stats.license': 'License',
      'stats.series': 'Baige Loong Series',

      'contact.badge': 'WeChat Community',
      'contact.title': 'Join the WeChat community',
      'contact.desc': 'Scan the QR code to join our WeChat group — ask questions, share benchmarks, and connect with the LoongForge team and fellow developers.',
      'contact.cta.view': 'View on GitHub',
      'contact.cta.discuss': 'Start a discussion',
      'contact.qr.fallback': 'QR image unavailable — visit GitHub README for the latest QR code.',
      'contact.qr.hint': 'Scan with WeChat to join',
    },
    zh: {
      'nav.home': '首页', 'nav.features': '特性', 'nav.models': '模型',
      'nav.docs': '文档', 'nav.blog': '博客', 'nav.about': '关于', 'nav.contact': '联系',
      'nav.github': 'Star',
      'footer.tagline': '面向 LLM / VLM / VLA / Diffusion 的模块化、可扩展训练框架。',
      'footer.project': '项目', 'footer.resources': '资源', 'footer.loong': '百舸 Loong 系列',
      'footer.copyright': '© 2026 LoongForge Authors · Apache License 2.0 · 由百度百舸团队用 ♥ 构建',
      'footer.quickstart': '快速开始', 'footer.modelconfigs': '模型配置',
      'footer.examples': '示例', 'footer.contributing': '贡献指南',
      'footer.training': '训练框架', 'footer.workflow': '智能体框架',

      'hero.badge': '🐉 百度百舸 Loong 开源家族成员',
      'hero.subtitle_html': '面向 <b>LLM</b>、<b>VLM</b>、<b>Diffusion</b> 与<b>具身智能</b>模型的<b>模块化</b>、<b>可扩展</b>、<b>高性能</b>训练框架 —— 基于 Megatron-LM 深度定制，原生支持 NVIDIA GPU 与昆仑 XPU。',
      'hero.cta.start': '🚀 立即上手',
      'hero.cta.github': '⭐ 访问 GitHub',
      'hero.cta.docs': '📚 阅读文档',
      'hero.stat.speedup': '最大训练加速',
      'hero.stat.xpu': 'XPU 集群规模',
      'hero.stat.modal': '覆盖模态：LLM/VLM/VLA/Diffusion',
      'hero.stat.fp8': '端到端自适应精度',

      'hero.vp.1.k': '易用',
      'hero.vp.1.t': '一套框架，广泛覆盖',
      'hero.vp.1.d': '全面覆盖主流开源 LLM、VLM、MoE、扩散与 VLA 模型。内置开箱即用的模型配置与启动脚本。',
      'hero.vp.2.k': '高效',
      'hero.vp.2.t': '训练加速最高 ~5×',
      'hero.vp.2.d': '深度性能优化 —— 融合算子、自适应 FP8、MoE A2A 通信计算重叠、多模态流水调度。',
      'hero.vp.3.k': '多芯',
      'hero.vp.3.t': 'NVIDIA GPU 与昆仑 XPU',
      'hero.vp.3.d': '原生异构硬件支持 —— 同一套框架，GPU 与 XPU 之间迁移成本极低。',

      'news.title': '🔥 最新动态', 'news.all': '查看全部 →',

      'feat.title': '✨ 关键特性',
      'feat.subtitle_html': 'LoongForge 差异化能力速览。',

      'feat.cat.1.title': 'MoE',
      'feat.cat.1.item.1.t': '通信/计算/Offload 三流并行',
      'feat.cat.1.item.1.d': 'MoE EP 通信 × 计算 × Offload 三流并行，吞吐优于上游。',

      'feat.cat.2.title': '多模态',
      'feat.cat.2.item.1.t': '模型拼接',
      'feat.cat.2.item.1.d': '通过 YAML 自由拼接 ViT × LLM 构建 VLM。',
      'feat.cat.2.item.2.t': '异构并行',
      'feat.cat.2.item.2.d': '不同组件独立设置 TP / PP / DP。',
      'feat.cat.2.item.3.t': '分离训练',
      'feat.cat.2.item.3.d': 'ViT 与 LLM 解耦调度，消除流水气泡。',
      'feat.cat.2.item.4.t': 'DP 负载均衡',
      'feat.cat.2.item.4.d': '修复 packing 带来的 DP 倾斜。',

      'feat.cat.3.title': '性能',
      'feat.cat.3.item.1.t': '自适应 FP8',
      'feat.cat.3.item.1.d': '按 GEMM 形状逐算子决策是否启用 FP8。',
      'feat.cat.3.item.2.t': '融合算子',
      'feat.cat.3.item.2.d': 'FusedDSA / 稀疏 MLA 融合算子，端到端加速。',

      'feat.cat.4.title': '性能',
      'feat.cat.4.item.1.t': 'ChunkPipe',
      'feat.cat.4.item.1.d': '长序列分块流水，面向百万级上下文。',

      'feat.cat.5.title': '易用性',
      'feat.cat.5.item.1.t': 'HF ↔ Megatron',
      'feat.cat.5.item.1.d': '双向检查点转换 + 在线 HF 加载/保存。',

      'feat.cat.6.title': '训练范式',
      'feat.cat.6.item.1.t': 'Pretrain + SFT + LoRA',
      'feat.cat.6.item.1.d': '同一套代码覆盖关键训练阶段。',

      'models.title': '🏛️ 支持的模型',
      'models.subtitle': '从 0.5B 小模型到 480B MoE 巨兽 —— 开箱即用。',
      'models.more.t': '以及更多…',
      'models.more.d': '新增架构只需一个 YAML + 注册表条目即可。',
      'models.custom.t': '自定义组合模型',
      'models.custom.d_html': '通过 YAML 自由组合任意 ViT + 任意 LLM 主干。<a href="https://github.com/baidu-baige/LoongForge/blob/master/configs/models/custom/qwen_vit_llama3_8b.yaml" target="_blank" class="text-indigo-500 underline">示例 →</a>',
      'models.diff.d': '图生视频扩散模型，搭配优化内核。',
      'models.vla.d': '视觉-语言-动作模型，paligemma 专家支持 torch.compile。',
      'models.groot.d': 'NVIDIA 风格具身 VLA 模型，针对大 batch SFT 做了生产级优化。',

      'qs.title': '🚀 快速开始',
      'qs.subtitle': 'YAML 驱动，五步从安装到启动。',
      'qs.s1.tab': '安装', 'qs.s2.tab': '组装模型', 'qs.s2.opt': '可选', 'qs.s3.tab': '训练配置', 'qs.s4.tab': '权重', 'qs.s5.tab': '启动',
      'qs.s1.desc': '推荐使用 Docker —— 单一镜像打包 CUDA/XPU 工具链、已打补丁的 Megatron 子模块以及 TransformerEngine，让每位开发者、每个节点都在完全一致的环境中训练。如需更灵活的部署方式，也完整支持源码安装。',
      'qs.s1.note_html': '如需源码安装与完整 build-arg 矩阵，请参考 <a class="underline" href="https://loongforge.readthedocs.io/en/latest/get_started/installation.html" target="_blank" rel="noopener">NVIDIA 安装指南</a> · <a class="underline" href="https://loongforge.readthedocs.io/en/latest/kunlun_tutorial/install_p800.html" target="_blank" rel="noopener">昆仑 XPU 安装指南</a>。',
      'qs.s2.desc': '可选步骤 —— 仅当你需要自定义组合时再来这一步。LoongForge 中的 VLM 是声明式的：一份 YAML 即可把<b>视觉编码器</b>、<b>Projector</b> 与<b>语言主干</b>串起来。主流开源模型均已内置于 <a class="underline" href="https://github.com/baidu-baige/LoongForge/tree/master/configs/models" target="_blank" rel="noopener">configs/models/</a> 目录 —— 针对 VLM 场景，三者中的任一组件（例如替换 LLM 主干）都可以通过一行 YAML 配置完成切换，无需改动 Python 代码。',
      'qs.s3.desc': '外层完全兼容 Megatron，熟悉的训练参数可以直接复用；内层通过 Hydra override 为每个模型组件单独指定并行策略（TP / PP / EP / DP）与 freeze 行为 —— 非常适合 ViT 与 LLM 主干计算特性差异较大的异构 VLM 训练。',
      'qs.s4.desc': '与 HuggingFace 双向打通：启动时可直接加载 HF 权重，无需手工转换；训练过程中内部使用 Megatron 原生格式，训练结束一键导出回 HF 格式 —— 下游推理服务链路无需改动。',
      'qs.s5.desc': '直接运行现成脚本即可。20+ 模型家族在 <a class="underline" href="https://github.com/baidu-baige/LoongForge/tree/master/examples" target="_blank" rel="noopener">examples/</a>（NVIDIA GPU）与 <a class="underline" href="https://github.com/baidu-baige/LoongForge/tree/master/examples_xpu" target="_blank" rel="noopener">examples_xpu/</a>（昆仑 XPU）下均提供生产级启动脚本，下方片段展示了所有 example 通用的 <code class="mono">torchrun</code> 启动模式。',

      'powered.title': '🌟 由 LoongForge 驱动',
      'powered.subtitle': '基于 LoongForge 训练的开源项目 —— 按时间从新到旧排列。',
      'powered.new': 'NEW',
      'powered.1.t': 'LLaVA-OneVision-2.0',
      'powered.1.d': '新一代完全开放的多模态模型 —— 在数据、训练配方与 Scale 上全面升级。',
      'powered.2.t': 'LLaVA-OneVision-1.5',
      'powered.2.d': '面向多模态训练民主化的完全开放框架。',
      'powered.3.t': 'Qianfan-VL',
      'powered.3.d': '领域增强的通用视觉-语言模型。',

      'cta.title': '加入社区',
      'cta.desc': '报告 Bug、提出建议、贡献代码，或只是打个招呼。我们 ❤️ 社区。',
      'cta.issue': '提交 Issue', 'cta.pr': '发起 PR', 'cta.about': '项目介绍',

      'docs.hero.title': '文档中心',
      'docs.hero.desc_html': '一份精选的安装、教程与参考资料索引。完整的 API 与深度教程托管在 <a href="https://loongforge.readthedocs.io/en/latest/index.html" target="_blank" class="underline text-amber-300">ReadTheDocs</a> 上。',

      'blog.hero.title': '工程博客',
      'blog.hero.desc': '来自 LoongForge 团队的版本发布、性能深挖与案例分享。',

      'about.hero.title': '关于 LoongForge',
      'about.hero.desc': '一个诞生于真实大规模生产负载、回馈开源社区的训练框架。',

      'bench.title': '📊 性能基准',
      'bench.subtitle': '基于 v0.1.0，在 A800 上覆盖 LLM、VLM、VLA 工作负载实测。',
      'bench.baseline': '1.0× 基线',
      'bench.ds.title': 'DeepSeek-V3.2 · DSA 算子级优化',
      'bench.ds.desc': '基于裁剪层数配置实测 —— 可支持 64K 序列训练。',

      'hw.title': '💎 硬件兼容性',
      'hw.subtitle': '同一套代码，两套芯片栈 —— NVIDIA GPU 与百度昆仑 XPU 均已生产化落地。',
      'hw.nv.t': 'NVIDIA GPU',
      'hw.nv.d': '基于社区 Megatron + TransformerEngine 生态，在此之上构建并扩充 LoongForge 自研优化。',
      'hw.xpu.t': '昆仑 XPU',
      'hw.xpu.d': '采用 XPU Plugin 机制，向上屏蔽 XPU 适配差异，同时集成 XPU 专属优化技术栈。',

      'comm.title': '🤝 开源社区',
      'comm.subtitle': '开放共建 —— 欢迎参与讨论、反馈问题、贡献代码。',
      'comm.1.t': 'GitHub Issues', 'comm.1.d': '提交 Bug 报告与功能请求。',
      'comm.2.t': '讨论区',         'comm.2.d': '提问交流与经验分享。',
      'comm.3.t': '贡献指南',       'comm.3.d': '阅读指南，发起你的第一个 PR。',
      'comm.4.t': '加入微信群',     'comm.4.d': '扫描 README 中的二维码，加入开发者交流群。',

      'stats.stars': 'GitHub Stars',
      'stats.contrib': '贡献者',
      'stats.license': '开源协议',
      'stats.series': '百舸 Loong 系列',

      'contact.badge': '微信社区',
      'contact.title': '加入微信社区',
      'contact.desc': '扫码加入我们的微信群 —— 提问交流、分享实践、与 LoongForge 团队及开发者直接对话。',
      'contact.cta.view': '在 GitHub 查看',
      'contact.cta.discuss': '发起讨论',
      'contact.qr.fallback': '二维码暂不可用 —— 请前往 GitHub README 获取最新二维码。',
      'contact.qr.hint': '使用微信扫码加入',
    }
  };

  const LANG_KEY = 'lf-lang';
  function getLang() {
    const saved = localStorage.getItem(LANG_KEY);
    if (saved === 'zh' || saved === 'en') return saved;
    const nav = (navigator.language || '').toLowerCase();
    return nav.startsWith('zh') ? 'zh' : 'en';
  }
  function applyI18n(lang) {
    const dict = I18N[lang] || I18N.en;
    document.documentElement.lang = lang === 'zh' ? 'zh-CN' : 'en';
    document.querySelectorAll('[data-i18n]').forEach(el => {
      const key = el.dataset.i18n;
      if (dict[key] != null) el.textContent = dict[key];
    });
    document.querySelectorAll('[data-i18n-html]').forEach(el => {
      const key = el.dataset.i18nHtml;
      if (dict[key] != null) el.innerHTML = dict[key];
    });
    // Update toggle button labels
    document.querySelectorAll('[data-lang-btn]').forEach(btn => {
      btn.classList.toggle('active', btn.dataset.langBtn === lang);
    });
    // Language-aware ReadTheDocs links
    applyDocsLinks(lang);
    // Language-aware blog links (Latest News cards, etc.)
    applyBlogLinks(lang);
    // Notify pages (blog list, etc.) that language changed
    try { window.dispatchEvent(new Event('lf:langchange')); } catch (e) {}
  }

  function applyBlogLinks(lang) {
    document.querySelectorAll('[data-href-en][data-href-zh]').forEach(a => {
      a.href = lang === 'zh' ? a.dataset.hrefZh : a.dataset.hrefEn;
    });
  }

  function applyDocsLinks(lang) {
    const base = lang === 'zh'
      ? 'https://loongforge.readthedocs.io/zh-cn/latest/index.html'
      : 'https://loongforge.readthedocs.io/en/latest/index.html';
    document.querySelectorAll('[data-docs-link]').forEach(a => { a.href = base; });
  }
  function setLang(lang) {
    localStorage.setItem(LANG_KEY, lang);
    applyI18n(lang);
  }

  // ===== Dark mode =====
  const THEME_KEY = 'lf-theme';
  const root = document.documentElement;
  const savedTheme = localStorage.getItem(THEME_KEY);
  const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
  if (savedTheme === 'dark' || (!savedTheme && prefersDark)) root.classList.add('dark');

  function toggleTheme() {
    root.classList.toggle('dark');
    localStorage.setItem(THEME_KEY, root.classList.contains('dark') ? 'dark' : 'light');
    updateThemeIcon();
  }
  function updateThemeIcon() {
    document.querySelectorAll('[data-theme-icon]').forEach(el => {
      el.textContent = root.classList.contains('dark') ? '☀️' : '🌙';
    });
  }

  // ===== Mobile menu =====
  function toggleMobileMenu() {
    const m = document.getElementById('mobile-menu');
    if (m) m.classList.toggle('hidden');
  }

  // ===== Copy to clipboard =====
  function attachCopyButtons() {
    document.querySelectorAll('pre').forEach(pre => {
      if (pre.dataset.copyAttached) return;
      pre.dataset.copyAttached = '1';
      pre.style.position = 'relative';
      const btn = document.createElement('button');
      btn.className = 'copy-btn';
      btn.type = 'button';
      btn.textContent = 'Copy';
      btn.addEventListener('click', async () => {
        const code = pre.querySelector('code') || pre;
        try {
          await navigator.clipboard.writeText(code.innerText);
          btn.textContent = 'Copied!';
          setTimeout(() => (btn.textContent = 'Copy'), 1400);
        } catch (e) {
          btn.textContent = 'Failed';
        }
      });
      pre.appendChild(btn);
    });
  }

  // ===== Tabs =====
  function initTabs() {
    document.querySelectorAll('[data-tabs]').forEach(group => {
      const btns = group.querySelectorAll('[data-tab]');
      const panels = group.querySelectorAll('[data-panel]');
      btns.forEach(btn => {
        btn.addEventListener('click', () => {
          const key = btn.dataset.tab;
          btns.forEach(b => b.classList.toggle('active', b.dataset.tab === key));
          panels.forEach(p => p.classList.toggle('hidden', p.dataset.panel !== key));
        });
      });
    });
  }

  // ===== Reveal =====
  function initReveal() {
    const els = document.querySelectorAll('.reveal');
    if (!('IntersectionObserver' in window)) { els.forEach(e => e.classList.add('visible')); return; }
    const io = new IntersectionObserver(entries => {
      entries.forEach(en => {
        if (en.isIntersecting) { en.target.classList.add('visible'); io.unobserve(en.target); }
      });
    }, { threshold: 0.08 });
    els.forEach(e => io.observe(e));
  }

  // ===== TOC spy =====
  function initTocSpy() {
    const links = document.querySelectorAll('.toc-link[href^="#"]');
    if (!links.length) return;
    const targets = [...links].map(l => document.querySelector(l.getAttribute('href'))).filter(Boolean);
    if (!targets.length) return;
    const io = new IntersectionObserver(entries => {
      entries.forEach(en => {
        if (en.isIntersecting) {
          const id = '#' + en.target.id;
          links.forEach(l => l.classList.toggle('active', l.getAttribute('href') === id));
        }
      });
    }, { rootMargin: '-40% 0px -55% 0px' });
    targets.forEach(t => io.observe(t));
  }

  // ===== Scroll progress bar =====
  function initScrollProgress() {
    const bar = document.getElementById('scroll-progress');
    if (!bar) return;
    const upd = () => {
      const h = document.documentElement;
      const scrolled = h.scrollTop / Math.max(1, h.scrollHeight - h.clientHeight);
      bar.style.width = (scrolled * 100).toFixed(2) + '%';
    };
    window.addEventListener('scroll', upd, { passive: true });
    window.addEventListener('resize', upd);
    upd();
  }

  // ===== Quick Start step tabs =====
  function initQuickStart() {
    const root = document.getElementById('qs-interactive');
    if (!root) return;
    const tabs = root.querySelectorAll('.qs-tab');
    const panels = root.querySelectorAll('.qs-panel');
    tabs.forEach(tab => {
      tab.addEventListener('click', () => {
        const step = tab.dataset.qsStep;
        tabs.forEach(t => t.classList.toggle('active', t === tab));
        panels.forEach(p => p.classList.toggle('active', p.dataset.qsPanel === step));
      });
    });
  }

  // ===== Init =====
  document.addEventListener('DOMContentLoaded', () => {
    applyI18n(getLang());
    updateThemeIcon();
    attachCopyButtons();
    initTabs();
    initReveal();
    initTocSpy();
    initScrollProgress();
    initQuickStart();
    window.LF = { toggleTheme, toggleMobileMenu, setLang };

    document.querySelectorAll('[data-theme-toggle]').forEach(el => el.addEventListener('click', toggleTheme));
    document.querySelectorAll('[data-mobile-toggle]').forEach(el => el.addEventListener('click', toggleMobileMenu));
    document.querySelectorAll('[data-lang-btn]').forEach(btn =>
      btn.addEventListener('click', () => setLang(btn.dataset.langBtn))
    );

    // GitHub stars + contributors (public REST API; no auth; client-side cache)
    // Star tile auto-hides below STARS_MIN (avoids showing a small number that
    // would read as negative social proof; self-heals once we cross the threshold).
    const STARS_MIN = 500;
    const starEls = document.querySelectorAll('#gh-stars, [data-gh-stars]');
    const starTiles = document.querySelectorAll('[data-gh-stars-tile]');
    const contribEl = document.getElementById('gh-contrib');
    const fmt = n => n >= 1000 ? (n / 1000).toFixed(1) + 'k' : String(n);
    const showStarTile = n => {
      if (typeof n !== 'number' || n < STARS_MIN) return;
      starEls.forEach(el => el.textContent = fmt(n));
      starTiles.forEach(el => el.hidden = false);
    };
    const CACHE_KEY = 'lf-gh-stats-v1';
    const CACHE_TTL = 60 * 60 * 1000; // 1 hour
    let cached = null;
    try { cached = JSON.parse(localStorage.getItem(CACHE_KEY) || 'null'); } catch (e) { }
    if (cached && Date.now() - cached.ts < CACHE_TTL) {
      showStarTile(cached.stars);
      if (contribEl && cached.contrib) contribEl.textContent = cached.contrib;
    }
    const next = { ts: Date.now(), stars: cached?.stars, contrib: cached?.contrib };
    if (starEls.length) {
      fetch('https://api.github.com/repos/baidu-baige/LoongForge')
        .then(r => r.ok ? r.json() : null)
        .then(d => {
          if (d && typeof d.stargazers_count === 'number') {
            next.stars = d.stargazers_count;
            showStarTile(d.stargazers_count);
            try { localStorage.setItem(CACHE_KEY, JSON.stringify(next)); } catch (e) { }
          }
        })
        .catch(() => { });
    }
    if (contribEl) {
      fetch('https://api.github.com/repos/baidu-baige/LoongForge/contributors?per_page=1&anon=true')
        .then(r => {
          const link = r.headers.get('Link') || '';
          const m = link.match(/page=(\d+)>;\s*rel="last"/);
          if (m) {
            next.contrib = m[1] + '+';
            contribEl.textContent = next.contrib;
            try { localStorage.setItem(CACHE_KEY, JSON.stringify(next)); } catch (e) { }
          } else {
            return r.json().then(arr => {
              if (Array.isArray(arr)) {
                next.contrib = String(arr.length);
                contribEl.textContent = next.contrib;
                try { localStorage.setItem(CACHE_KEY, JSON.stringify(next)); } catch (e) { }
              }
            });
          }
        })
        .catch(() => { });
    }
  });
})();

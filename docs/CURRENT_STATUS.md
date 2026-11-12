# 当前状态 · v0.7.2

v0.7.2 收束已验收的 M3：在 v0.7.1 的文字对话和模块化 SDK 上加入 KV/上下文存盘、跨进程续聊及逐会话采样配置。Llama/Qwen2 模块、INT8/F64 数值路径和 runtime V1 ABI `0x00060003` 保持原有约定。新增 `sampling`、`session` 和不透明状态接口，推理仍为单线程。

用户在 2026-11-11 完成固定实机验收，8/8 项用例通过；32 次调用以及 REVIEW/FULL 原始证据已独立复核。相同模型、数值构建、后端与容量下，跨进程续聊完整快照逐字节一致；reset 恢复数值状态与随机种子，原 V1 累计生成计数仍保持累计语义。发布整理不改变已验收的源码、公开头或 CMake 数值构建定义。

新增 `bash scripts/build-cli.sh`，可在源码根目录生成 `micius_cli/micius_cli.exe`，附带运行所需的 PCRE2 DLL。保存目录仍以启动时的工作目录为基准。

[发布说明和证据](RELEASE_0_7_2.md) · [快照、采样与边界](M3-PERSISTENCE-AND-SAMPLING.md) · [构建操作](BUILD_AND_CONFIGURATION.md)

## v0.7 数值与性能基线（历史）

2026-11-01 的首个 INT8 版本以 I25-P1 验收通过的实现为基础。发布入口为 `int8-release-windows`，F64 对照入口为 `f64-reference-windows`。公共 V1 ABI 为 `0x00060003`。

| 项目 | v0.7 当时的发布状态 |
|---|---|
| GGUF 输入 | v2/v3，受支持的 Llama 架构及 Q8_0 张量。目标机验收使用 SmolLM2-360M-Instruct-Q8_0 |
| 输入与输出 | 显式 token ID，逐 token prefill、greedy decode、F32 logits 和 K/V 复制 |
| 执行 | 单 context 单线程权威前向。公开配置调用 speculative 模式会收到 unsupported 错误 |
| INT8 | 每 32 个激活独立量化，精确整数块点积，按序 F64 反缩放。形状合适时走四行私有面板，尾行形状沿 I22 消费 |
| Attention | F32 KV，有序 Score64 与 Value64，满足形状及浮点前提时启用共享 Key、查询复用及 Value 分块 |
| 工具链 | 发布验证使用 Windows x64 Clang64、AVX2/FMA，INT8 发布配置禁用 VNNI |
| 适用范围 | CLI 输入是 token ID。Qwen2.5 在 I7 只有 llama.cpp 记录。文本分词、聊天模板、多核单 token 加速与其他 GGUF 架构超出本版验收范围 |

I25 对 I22 的正式复跑使用两篇文本、每篇 2,048 个位置和六对重复。总体配对吞吐提升 **18.4635%**。两篇六次吞吐中位数分别为 I25 **33.3705、33.3961 tok/s**，I22 **28.0643、28.2315 tok/s**。总体收益使用同轮比值的篇内中位与跨篇几何平均。运行号为 `i25-p1-20261031-102126-e089f2f3`。[完整实验](history/int8/I25.md)

加载及解码结束后的私有提交量比 I22 低约 **207 MiB**。绑定期间两种布局短暂重叠，生命周期峰值中位数从约 **1,593.0 MiB** 增至 **1,982.5 MiB**。加载峰值决定此模型启动所需的容量。[内存口径](INT8_PERFORMANCE_AND_MEMORY.md)

I5 在固定 32,768 个自然文本目标上测得 W8A8 的 PPL 为 **14.41502**，F64 为 **14.38900**。这是 I1 数值路线的直接质量证据。I25 的直接输出对照覆盖两条指定文本，发布构建另以合成模型逐步对照封存程序。[质量评估](INT8_NUMERICS_AND_QUALITY.md) · [发布验证](VALIDATION_AND_BENCHMARKING.md)

最后一次与 llama.cpp 无 VNNI 的正式同场比较是 I21，I19 达到对手吞吐的 **79.5984%**。I25 的直接比较对象为 I22。

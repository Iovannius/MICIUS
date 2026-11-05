# MICIUS v0.7.1 发布说明

v0.7.1 将 M1 的 GGUF 文件会话与架构模块、M2 的分词和文字对话纳入正式发布。默认构建为无 VNNI 的 AVX2 INT8，F64 使用独立配置。模型推理仍由单个线程执行。软件包、CMake 和 CLI 使用发布版本号，直接绑定张量的 V1 C ABI 编号仍为 `0x00060003`。

CLI 的启动字符画按原稿排版。四个方块从左到右显示为红、黄、蓝、绿。设置 `NO_COLOR=1`、使用 `TERM=dumb` 或重定向输出时，标记显示为 `░▒▓█`。字符画已编入 CLI，最长一行占 106 列。[字符画原稿](../src/app/MICIUS.txt)

## 使用范围

| 能力 | 本版行为 |
|---|---|
| 模型接入 | 按 GGUF 的 `general.architecture` 选择链接的 `llama` 或 `qwen2` Parser。Qwen2 路径支持已验证的 Q/K/V bias 和分半 RoPE。 |
| 文本 | 已验收 SmolLM2-360M-Instruct Q8_0 和 Qwen2.5-1.5B-Instruct Q8_0 的字节 BPE、角色模板、流式 UTF-8 回复和多轮 greedy decode。其他 GGUF 需满足架构、张量、词表与模板要求。 |
| 命令 | 命令模式输入 `load "模型路径"`，随后输入 `chat`，也可直接使用 `micius_cli --chat MODEL --context 1024`。会话使用 `You:` 和 `Assistant:`。规范命令为 `generate`、`clear`、`exit`，旧拼写 `run`、`cls`、`quit` 已移除。 |
| 历史与日志 | 成功完成的轮次共用内存中的 token 历史和 KV。可用 `:log start "会话记录.txt"` 保存可读文本，`:log stop` 停止记录。 |
| 目录与快照 | 文件会话可按模型 SHA-256 和会话 ID 创建独立的 `contexts/<id>/kv/` 目录。离开聊天或执行 `:reset` 会清空聊天状态。KV 数据写盘、跨进程恢复和续聊列入 v0.7.2 的工作范围。 |

## 验收来源

下面两项是在用户机器上用发布前的 M1、M2 程序完成的。源码随附两份实机结果及各自的核对协议。M1 发布副本中的会话目录已改为相对路径，M2 结果保留了原始结果文件的字节。

| 记录 | 输入与观察 | 结论 |
|---|---|---|
| M1 · 2026-11-03 | 两款模型、两种后端，各 8 个固定位置和 reset。SmolLM2 分别逐字节复现各自参考。Qwen2.5 对冻结 llama.cpp 的最高去均值 logits 相对 L2 分别为 3.0472%（INT8）和 10.1486%（F64），通过预设 35% 上限及 NLL 检查。 | `FUNCTIONAL_CHECK_PASSED` |
| M2 · 2026-11-05 | 两款模型、两种后端，各执行两轮文字对话并重置重放首轮。SmolLM2 输出 `Blue.`，Qwen2.5 输出 `Blue`，共 12 个回复自然终止。各组上下文长度分别为 42 → 80 → 42 和 39 → 60 → 39。 | `TEXT_CHAT_FUNCTIONAL_PASS` |

M1 运行编号为 `m1-20261103-133223-a9c37bb5`，原 REVIEW 的 SHA-256 为 `2d9f31cb912a821e470d52ff2c5911b06247d4c05b6d9a4404edb3683adfb219`。[M1 实机结果](evidence/m1-functional-result.json) · [M1 协议](../validation/m1/protocol.json)

M2 运行编号为 `m2-20261105-131145-70b30e84`，原 REVIEW 的 SHA-256 为 `aa36117a56ee83d05e3e54f7da42eb10f8a8cd75250876a39d81073582e76d0c`。普通文本的 5,800 个 SmolLM2 token ID 和 4,125 个 Qwen2.5 token ID 以及首轮和续轮模板均匹配冻结参考。这组问答检验短对话功能，长期质量和解码吞吐使用各自的研究记录。[M2 实机结果](evidence/m2-functional-result.json) · [M2 协议](../validation/m2/protocol.json)

## 在自己的机器复核

从本源码包根目录运行 `python -B scripts/verify_release.py` 核对发布清单。两个 Bash 构建入口分别运行 INT8 和 F64 的原生测试，包括冻结 CLI 输出、文件会话、分词、日志和多轮对话。可按 [构建与配置](BUILD_AND_CONFIGURATION.md) 安装 SDK，使用 [独立文件调用方](../examples/file-consumer/CMakeLists.txt) 或 [独立文本调用方](../examples/text-consumer/CMakeLists.txt) 验证链接。

复现实机实验时，从源码包根目录运行 `bash scripts/run_m1.sh all` 或 `bash scripts/run_m2.sh all`，通过 `MICIUS_MODELS` 指定含两份 GGUF 的目录。只需构建和合成检查时使用各脚本的 `prepare`。模型、日志、缓存和生成的二进制均位于源码包之外。

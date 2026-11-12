# M3 · v0.7.2 的会话存盘与采样

M3 基于已验收的 v0.7.1，增加可恢复的 KV/上下文快照、逐会话采样配置，以及独立可链接的 `sampling`、`session` 模块。模型算子、分词规则和聊天模板沿用原实现，推理仍为单线程。M3 已于 2026-11-11 完成固定实机验收，并纳入 v0.7.2；其范围见 [发布说明](RELEASE_0_7_2.md)。

## 使用流程

启动普通 CLI 后，输入 `load "模型路径"`，再输入 `chat`。首次进入会创建配置并保存空会话；界面显示目录。输入仍使用 `You:` / `Assistant:`。

完成一轮会话、`:reset`、`:save`、正常 `:exit` 或输入 EOF 都会保存。每轮保存成功后，退出进程再恢复可继续使用原来的 KV。Ctrl+C 取消当前轮：已显示的局部回答不纳入历史；下一次保存会恢复到上一轮完整状态。如果进程被强制结束，只保留最后一次成功保存的快照。

```text
load "models/qwen2.5-1.5b-instruct-q8_0.gguf"
chat
:config
:save
:exit
chat list
chat resume 这里替换为32位会话ID
```

也可以直接运行：

```bash
./build-int8-release/bin/micius_cli.exe --chat "models/qwen2.5-1.5b-instruct-q8_0.gguf" --context 1024
./build-int8-release/bin/micius_cli.exe --chat "models/qwen2.5-1.5b-instruct-q8_0.gguf" --context 1024 --session YOUR_32_HEX_ID
```

恢复必须使用相同模型内容、数值构建、后端和 context 容量。`chat [N]`、`chat resume ID [N]` 或 `--max-new N` 可临时覆盖本次调用的回答长度，CLI 会显示该覆盖。其他采样参数来自文件。`:help`、`:config`、`:reload`、`:save`、`:reset`、`:log start "file.txt"`、`:log stop` 和 `:exit` 是聊天内命令；命令壳的三个旧别名仍不接受。

## 模型与会话目录

默认根目录为当前工作目录的 `.micius-cache/`，可通过 `MICIUS_CACHE_ROOT` 指定。目录键来自 GGUF 完整内容的 SHA-256；移动或改名同一文件不会变更模型身份，修改文件内容会创建不同身份。

```text
.micius-cache/
  <model-sha256>/
    model.txt
    contexts/
      <32位context-id>/
        .lock
        sampling.conf
        kv/
          chat.mcs
```

同模型各会话有独立的 KV、历史、配置和随机数状态。每次文件加载仍按 M1 约定预留一个新目录，恢复旧会话时可能留下未使用的空目录；`chat list` 仅列出已经创建聊天配置的上下文。旧版 v0.7.1 的空目录没有可恢复的 KV，不能凭目录重建已退出的对话。

会话锁由操作系统句柄保持，关闭进程后自动释放；`.lock` 文件存在不表示会话仍被占用，不需要删除它。同一目录不能被两个实例同时读写；同一文件模型的引擎不能同时打开两个聊天句柄。不同会话可以独立存储，这尚不等于 Agent 编排或多线程推理已经实现。

## sampling.conf

首次配置采用原来的贪心行为，示例：

```ini
version=1
temperature=0
top_k=0
top_p=1
repeat_penalty=1
repeat_last_n=0
seed=0
max_new_tokens=128
```

可以使用纯文本编辑器改为例如 `temperature=0.7`、`top_k=40`、`top_p=0.9`、`repeat_penalty=1.1`、`repeat_last_n=64`。配置在每次用户输入进入前向计算之前重新读取；也可通过 `:reload` 立即校验。`:config` 显示最近已加载的有效配置。

| 字段 | 约束与语义 |
| --- | --- |
| version | 必须为 1；其他字段缺省时采用上述默认值 |
| temperature | 有限且不小于 0；0 为贪心，不消耗随机数 |
| top_k | 无符号 32 位整数；0 不裁剪，否则保留最多 k 项 |
| top_p | 有限且在 (0,1]；裁剪后至少保留一项 |
| repeat_penalty | 有限且大于 0；1 不惩罚；正 logit 除以系数，负 logit 乘以系数 |
| repeat_last_n | 无符号 32 位整数；0 不惩罚，否则检查最近 n 个已消费 token，每个 ID 只惩罚一次 |
| seed | 无符号 64 位整数；修改 seed 会重新开始随机数序列，保留同一 seed 则继续当前状态 |
| max_new_tokens | 正的无符号 32 位整数；受 context 剩余容量限制 |

重复惩罚范围包含系统模板、用户输入及本轮已生成 token，不只包含助手文字。处理顺序为：可生成 token 筛选 → 重复惩罚 → 温度/top-k → softmax/top-p → 随机抽样。top-k 的排序在温度缩放前完成；正温度不改变排序。并列时选择较小 token ID。非默认采样排除不能解码为普通字节的控制 token，但保留结束 token。默认温度 0、惩罚关闭时直接使用原有贪心结果。

随机数发生器为明确实现的 SplitMix64，每次成功的随机抽样消耗一次状态。为了沿用 M2 的轮次结束判定，达到回答长度上限时，最后一个已消费 token 后仍计算下一次选择；它可能不显示，但随机数进度会存盘。取消或失败轮次不提交这部分进度。`:reset` 清空历史与 KV，并将随机状态回到最近加载的 seed。

文件上限为 16 KiB；重复键、未知键、非法数字和超范围参数均拒绝，保留已有会话。快照包含上次保存时的有效配置，恢复后下一轮仍会按当前文件读取参数；因此复现续聊时必须保持配置不变。

## 快照内容与恢复边界

`kv/chat.mcs` 保存：模型 SHA-256、架构、格式版本、配置、随机数状态、完整 token 历史、轮数、数值构建身份、FP 控制模式、有效 KV 前缀及最后 logits。没有模型权重、指针或临时 scratch。原 V1 的累计生成计数作为诊断值保存；reset 不把这个累计计数变成对话 token 数。

有效 KV 数据字节数为 `2 × layers × sequence_length × kv_heads × head_dim × sizeof(float)`，另有最后一行 logits、token 历史与元数据。磁盘快照不会把未使用的 context 容量全部写出。当前保存使用临时序列化缓冲；保存时的瞬时进程内存可能高于稳定推理状态，不主张存盘没有额外内存。

恢复先校验 SHA-256 校验和、模型、形状、后端/源码/构建指纹、浮点控制、长度、有限数值和 token 范围，再将已验证数据复制到引擎。失败不会替换当前数值状态。构建指纹较保守：不同编译器、编译参数、源码字节或 INT8/F64 后端可能不能互相恢复。当前不提供跨版本快照迁移、异构后端恢复或跨平台兼容承诺。

保存先写同目录的独占临时文件并刷新，然后原子替换目标。测试覆盖目标文件被占用导致的替换失败：旧快照保留。磁盘或操作系统在最终落盘期间故障仍受其持久化语义限制，不能据此声称验证过断电场景。校验和用于检测损坏，不能证明一个经过第三方恶意重签的快照可信。

## 模块与 ABI

| CMake target / 公开头 | 职责与依赖 |
| --- | --- |
| `MICIUS::core` / `micius_state.h` | 有效 KV/logits 的不透明导入导出；原 `micius_runtime.h` 和 V1 ABI 保持不变 |
| `MICIUS::sampling` / `micius_sampling.h` | logits 到 token 的独立采样；无需 GGUF、文本或推理核心 |
| `MICIUS::session` / `micius_session.h` | 目录、配置、历史、快照和聊天生命周期；链接 text、GGUF、sampling 与 core |

新增接口单独使用 V1 版本号和结构长度。文件模型和引擎由调用方持有，聊天句柄借用它们；先关闭聊天，再关闭文件模型。聊天打开期间不能直接驱动、reset 或重新绑定借用的引擎。C 接口成功 Reply 只提交内存，应用必须显式调用 `Micius_SaveChatV1`；CLI 自动完成保存。取消/失败的部分 KV 在下一次 Reply/Save 时重建，取消返回的 context_length 可能仍是部分引擎进度。只有 `Micius_CancelChatV1` 可跨线程请求取消，其余调用不允许并发。

raw state API 不携带 GGUF 内容身份，直接调用者必须保证已绑定相同权重；需要完整身份校验的应用应使用 session API。raw import 验证失败返回 0，export 返回 1 成功、0 容量查询/不足、-1 错误。快照字节是版本化的不透明实现格式，调用方不应直接改写。

示例见 [外部 C 会话程序](../examples/session-consumer/session_probe.c) 和 [独立采样程序](../examples/sampling-consumer/sampling_probe.c)。

## 验收记录与复现

构建阶段只运行合成模型：原有数值契约、贪心兼容、确定种子采样、不同进程恢复、reset、损坏/错模型/错形状/错后端拒绝、浮点控制、同目录锁、保存失败保留及外部 C 链接。

[固定实机协议](../validation/m3/protocol.json) 覆盖两款模型 × INT8/F64 × 贪心/固定种子，共 8 个用例。每个用例分别执行连续两轮、独立首轮、跨进程续轮和 reset 重放，共 32 次短进程调用。两轮上限各 32 token，context 为 512，不包含性能计时。贪心输出额外逐项对照已封存 M2 记录。

```bash
# 在本源码包根目录；可设置 MICIUS_MODELS 指定两份 GGUF 的目录
bash scripts/run_m3.sh all
```

完整构建和准备可以单独执行 `prepare`，随后 `run RESULT_DIRECTORY` 启动一次新的验收。中断、缺少记录、输出差异、非法配置或身份变化均不能被标为通过；要重跑应使用新的 `all`，不得拼接中断采样。结果同时封存 REVIEW（摘要、原始文字、诊断及散列）和 FULL（另含完整快照与 raw state）。

交互模式可使用 `bash scripts/run_m3.sh chat qwen INT8`；恢复时追加 `--session ID --context 1024`。这是日常使用入口，不会自动记为实机验收通过。

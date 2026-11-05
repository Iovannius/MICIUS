# M2 · 单线程文字对话、分词与会话日志

v0.7.1 收录 M2 的可独立链接文本模块 `MICIUS::text`。目标模型是 SmolLM2-360M-Instruct Q8_0 和 Qwen2.5-1.5B-Instruct Q8_0。两款模型均已在 INT8 和 F64 下通过固定的两轮对话及重置重放。[实机结果](evidence/m2-functional-result.json)

## 开始聊天

在发布源码根目录打开 Git Bash。将模型放入 `models/`，或用 `MICIUS_MODELS` 指定模型目录。默认使用 INT8 和 1,024 个上下文位置。首次从下列脚本启动会先构建并验证 SDK。

```bash
bash scripts/run_m2.sh chat qwen
```

将 `qwen` 改为 `smol` 可切换模型，模型名后可加 `F64`。在 **You:** 后直接输入一行自然语言，保留空格，按 Enter 提交。回复以 **Assistant:** 开头，逐字节 token 组成完整 UTF-8 字符后再显示。回复采用贪心选择。默认每轮最多生成 128 个 token。命令模式仍显示 `>` 并提供数字 ID 的 `generate`。

| 聊天内的输入 | 作用 |
|---|---|
| 普通文本 | 提交一个用户轮次，沿用当前对话的 KV 与已完成历史 |
| `:reset` | 清空本次对话，下轮重新加入该模型的系统消息 |
| `:log start "会话记录.txt"` | 向指定 UTF-8 路径追加可读记录，允许空格 |
| `:log stop` | 关闭记录 |
| `:exit` | 离开独立聊天程序，或从 Shell 内的聊天返回命令模式 |
| 回复期间 Ctrl+C | 中止并丢弃当前轮，下一次有效输入会从已提交历史恢复 |

每次启动新聊天使用独立会话。到达回复长度上限时保留已生成的回复并封闭该轮；上下文不足以容纳完整输入、至少一个回复位置和轮末标记时，拒绝新输入并保留历史。可用 `:reset` 开始新对话。实际模型权重以 GGUF 文件为准。首次运行检查 SHA-256。可用 `MICIUS_MODELS` 指定两份 GGUF 所在目录，`MICIUS_CACHE_ROOT` 指定 context 目录的位置。

独立 SDK 的命令行入口允许指定上下文与回复长度。请将安装前缀的 `bin` 及 Clang64 运行库目录加入 PATH。

```bash
micius_cli --chat models/qwen2.5-1.5b-instruct-q8_0.gguf --context 1024 --max-new 128
```

在已有命令 Shell 中，也可先用 `load "模型文件.gguf"`，再输入 `chat`。命令模式的 `load` 默认申请模型声明的上下文容量，内存有限时可使用带 `--context` 的直接入口。

## 分词和模板边界

通用读取器保留文件中词表、合并顺序、token 类型和模板的信息。词表按 GPT-2 的可逆字节映射解码，按每个模型的 Unicode 预分词规则划分，再按合并排名与从左到右的顺序执行 BPE。SmolLM 先隔离 Unicode 数字，Qwen2 有自己的缩写、符号和换行规则。普通用户内容中的 `<|im_start|>` 等控制符拼写作为文字处理。角色标记由模板组装逻辑显式插入。SmolLM2 的真实词表缺少 21 个单字节符号。遇到该表无法无损表达的输入，本轮会收到诊断，已完成的对话可继续。

两个目标 GGUF 的原始模板分别以 SHA-256 确认，对应普通用户/助手对话、模型自带的默认系统消息、生成前缀及轮末标记。Qwen2 的工具调用分支、任意 Jinja 执行及未经支持的模板超出本单元范围。未识别的模板在进入聊天时报告原因，已绑定的数值会话仍可使用。模型输出的 `im_end` 或 `endoftext` 结束回复，其他非文本 ID 触发诊断。截断回复仍为下一轮加入规范的结束标记。终端将非法或不完整 UTF-8 显示为替换字符，控制字节不会执行终端命令。

[独立 C 调用示例](../examples/text-consumer/token_probe.c) 仅链接 `MICIUS::text`，可以打开词表、查询所需 token 数量并获取普通 token 的原始字节。原始字节片段可能跨越 UTF-8 字符。容量不足不写部分结果。文本接口版本为 `0x00010000`，生命周期与缓冲区说明见[公开头文件](../include/micius_text.h)。

文本构建通过 pkg-config 查找 `libpcre2-8`，启用 UTF 和 Unicode 属性。Windows 验证环境为 MSYS2 Clang64，动态链接时需提供对应的 PCRE2-8 运行库。仅需要原数值 SDK 时可在新的构建目录指定 `-DMICIUS_ENABLE_TEXT=OFF`。安装后的文本调用方用 `find_package(MICIUS 0.7.1 CONFIG REQUIRED COMPONENTS text)`。[独立 CMake 示例](../examples/text-consumer/CMakeLists.txt) · [PCRE2 Unicode 说明](https://www.pcre.org/current/doc/html/pcre2unicode.html)

## 日志

会话日志默认关闭。开启后，命令、对话角色和用户可见回复写入同一份追加文件。保留正文与换行，去除终端颜色及控制序列。普通输出按行或轮次刷新；遇到写入错误会提示并关闭记录，可重新打开。命令模式沿用 `log start` 与 `log stop`。

内部诊断记录保留 64 字节的事件布局，每个参数独立存储类型和值。显式开启失败时不会启动日志后台线程；停止接收后排空已接收事件，队列满造成的丢弃会在关闭时写出总数。后台日志只由显式调用方开启。对话与实机检查走单线程推理和同步的会话记录路径。

## 功能验收

```bash
bash scripts/run_m2.sh all
```

`prepare` 只做源码身份、两套构建、CTest、安装与外部 C 调用方验证。随后可运行 `run <本包的PREPARED目录>`。`all` 新建独立目录完成两个阶段。中断保留证据，同一个已启动的目录不能重复拼接运行。交互式 `chat` 可以单独使用，它不会把手动对话登记为正式验收。

实机阶段先核对两份 GGUF 与原始参考的身份、固定中英混合文本的全部 token ID 及两个模板的首轮与续轮前缀。参考由 llama.cpp 0.6.0 以只读词表模式生成，模板由它的 Jinja 实现渲染。随后在 INT8、F64 两套正式 CLI 上对每份模型进行两轮对话，清空后重放首轮。机器检查要求有可见文字输出、终止原因有效、第二轮上下文增长，重放的文字字节、生成数量、序列长度和结束原因一致。有意义的答复内容可直接在 `turns.json` 中人工阅读。完成状态为 `TEXT_CHAT_FUNCTIONAL_PASS`，结果及原始记录封存在本包的 `exports/`。本轮不计速度，也不把固定问答外推为综合质量结论。[固定输入与参考身份](../validation/m2/protocol.json)

M1 的数值回归口径与 ABI 说明另见 [M1 研究说明](M1-MODULES-AND-ABI.md)。

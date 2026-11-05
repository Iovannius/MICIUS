# M1 · 单线程模块 SDK 和 Qwen2 接入

v0.7.1 包含 M1 验收的模块和接口。M1 从 v0.7 提交 `65d51713ac905b9f84cab865241c0e3cc9432a41` 派生。验证环境为 Windows x64、MSYS2 Clang64、CMake、Ninja 和 Python。模型功能复核使用数字 token ID，所有请求在同一 context 上依次执行。[正式发布与验收记录](RELEASE_0_7_1.md)

## 从发布源码根目录复核

```bash
export MICIUS_MODELS=C:/Projects/MICIUS/models
bash scripts/run_m1.sh all
```

`all` 在源码包的 `results/` 下新建目录，构建并安装两套 SDK，运行合成验证，然后对 `MICIUS_MODELS` 中的两个固定 GGUF 做四次功能检查。也可以把模型放在调用时所在目录的 `models/` 中。工具链默认在 `/c/msys64/clang64/bin`，可通过 `MICIUS_CLANG64_BIN` 设置。两套模型串行加载。测试应用只申请八个位置的 KV 容量，权重仍需完整装载。

只运行构建、安装和合成检查使用 `prepare`。随后可用 `run <PREPARED目录>` 从相同核心工作目录开始实机检查。运行器保留各次日志、输入、输出、比较值、构建和程序身份。完整结果归档至 `exports/`。一次已启动的检查保留原目录，再次执行 `all` 会取得全新的目录。

## 从安装包链接

| 安装目标 | 边界 |
|---|---|
| `MICIUS::core` | 公开 V1 C ABI、模型绑定、单线程前向、KV、logits 和 reset |
| `MICIUS::gguf` | 文件映射、容器边界检查、张量转换、文件会话和目录。依赖 core |
| `MICIUS::parser_llama` | Llama 元数据与张量名映射，导出 `Micius_LlamaParserV1()` |
| `MICIUS::parser_qwen2` | Qwen2 元数据、张量名映射和 QKV/RoPE 语义，导出 `Micius_Qwen2ParserV1()` |

在全新的构建目录使用 `cmake --preset int8-release-windows`、`cmake --build --preset int8-release-windows`、`cmake --install build-int8-release --prefix <sdk>` 产生 SDK。F64 使用单独的 `f64-reference-windows` 构建和安装前缀。SDK 使用静态链接，消费者启用 C++ 链接器，业务调用代码可用 C11。组件可用 `MICIUS_PARSER_LLAMA`、`MICIUS_PARSER_QWEN2` 选择构建。普通 CLI 至少需要一种架构模块。Windows 的文件摘要使用系统 BCrypt，类 Unix 实现使用 OpenSSL Crypto。

独立调用方使用 `find_package(MICIUS 0.7.1 CONFIG REQUIRED COMPONENTS gguf parser_qwen2)`，再链接 `MICIUS::gguf` 和 `MICIUS::parser_qwen2`。调用方显式传入所选模块的描述表，每次打开模型按 GGUF 的 `general.architecture` 精确匹配。未来编译新的静态 Parser 后，只要实现同一接口、能映射到已有前向语义，即可通过调用方注册并重链接。新的数学语义须扩展运行时并单独验证。两份 SDK 可各自支持应用，应用应选择其中一个数值后端。[外部 CMake 示例](../examples/file-consumer/CMakeLists.txt) · [纯 C 调用示例](../examples/file-consumer/model_probe.c)

## ABI、版本与所有权

| 接口 | 版本和调用方责任 |
|---|---|
| 运行时 V1 | `MICIUS_ABI_VERSION=0x00060003`，原公开头文件逐字节继承。保留 C 链接、平台默认 C 调用约定和自然对齐。调用时提供真实结构大小、版本及 tensor 条目步长。Win64 上 `MiciusConfig` 56 字节，`MiciusTensorEntryV1` 80 字节，块 scale 16 字节。 |
| 架构模块 | `MICIUS_GGUF_PARSER_ABI=0x00010000`。描述表含版本、结构大小、架构名、配置描述和张量映射回调。回调应以 0 或 NULL 报告拒绝；参数和元数据字符串在回调期间借用。映射结果必须是输入字符串或静态字符串，宿主立即复制。缺少必需字段、已存在但类型错误的字段和未知语义须报告失败。 |
| 文件会话 | `MICIUS_GGUF_FILE_ABI=0x00010000`。初始化 `MiciusGGUFOpenOptionsV1` 的版本与完整大小并传入 UTF-8 路径和 Parser 列表。重复架构、短结构、错误版本或未注册架构会拒绝打开。失败返回 NULL，调用方的诊断缓冲区接收错误说明。 |
| 生命周期 | 文件会话拥有映射、转换缓冲区和引擎。获取到的引擎、配置和字符串均由会话借出，到 close 为止有效。通过引擎调用原 V1 的 prefill、decode、复制 logits/KV 和 reset。由 `Micius_CloseGGUFV1` 释放文件会话，借用的引擎不得独立销毁或重新绑定。原 V1 的纯内存绑定及其数据生存期仍按原契约。 |

对这组 V1 表和回调固定字段布局与签名，结构大小用于截断检查。改变回调输入、语义位定义或所有权规则时，应使用可区分的新版本。外部模块和宿主使用兼容的目标 ABI。公开边界只传 C 数据与不透明句柄。[运行时头文件](../include/micius_runtime.h) · [文件与 Parser 头文件](../include/micius_gguf.h)

## 架构与失败路径

Llama 的原始相邻配对 RoPE 由冻结轨迹验证。Qwen2 使用头内分半配对。在严格检查每层 Q/K/V 的 bias 类型、形状和有限性之后，运行时在投影后加入 bias，再对 Q、K 应用 RoPE。GQA 按 KV 头分组。显式输出矩阵和可支持的 Q8 tied embedding 遵循绑定检查。目标 Qwen2.5-1.5B GGUF 含 28 层、12 个查询头、2 个 KV 头、头维度 128、隐藏宽度 1536、FFN 宽度 8960、词表 151936。参考路径按实际 FFN 宽度申请工作区，旧的四倍隐藏宽度约束属于 worker 工作区。

当前文件模块支持限定的 dense full-causal 形状、全维 RoPE 及本版张量集合。未知架构、额外的不支持张量、非平凡 RoPE scaling、滑动窗口、不支持的类型或缺少 bias 均应在加载时报告诊断。参数 `context_length` 的 0 表示使用模型容量，显式正值不得超过模型上限。新会话完成解析、绑定和目录创建后，Shell 才替换原会话。

## 上下文目录和命令

设置 `cache_root` 后，成功打开模型实际建立 `<root>/<GGUF内容SHA-256>/contexts/<独立会话ID>/kv/`。同名但内容不同的 GGUF 分属不同模型目录。同一个模型的不同会话以原子创建的目录区分。通过 `Micius_FileContextDirectoryV1` 查询 context 根目录。reset 只清内存中的序列和 KV，目录标识保持有效。close 释放模型资源，保留已创建的目录供宿主检查和管理。目录中当前没有序列化 KV，磁盘快照及恢复需另定格式。传 NULL 的 C 调用方可完全使用内存；空字符串是非法路径。

Shell 取 `MICIUS_CACHE_ROOT`，默认使用当前工作目录的 `.micius-cache/`。交互命令为 `help`、`exit`、`load`、`stats`、`debug workers`、`clear`、`log` 和 `generate`。旧拼写 `quit`、`cls`、`run` 分别迁移为 `exit`、`clear`、`generate`。旧拼写走未知命令分支，帮助只列规范名称。

## 验收口径

- 合成与 ABI。三种 Llama 模型各 160 步，两后端分别对照 v0.7 的完整输出。小型 Qwen2 使用独立标量前向核对全部 logits，另以错误 RoPE 排列和省略 bias 作为负对照。覆盖宽 FFN、错误注册、缺失张量、失败替换、无磁盘会话以及多个活跃 context。
- 安装。SDK 安装后移动到新前缀。使用独立 CMake 工程构建仅 Llama、仅 Qwen2 和双模块程序，并验证实际链接行及缺少模块时的行为。同一份用 v0.7 头文件编译的 C 对象分别重链接两套新 SDK。每种组合记录去符号 PE 的字节数、整份程序的 `.text` 容量和导入 DLL。该口径包含宿主与静态链接部分；动态依赖的体积另计。
- 实机。I7-P1 的 s00 输入固定取前八个位置，第九个 token 提供最后一个目标。两后端的 SmolLM2 完整 logits 要求分别与冻结 I1/F64 逐字节相同。Qwen2 的八个完整词表行与冻结 llama.cpp 对照。逐位置去均值 logits 相对 L2 上限为 0.35，八个位置目标 NLL 差绝对值的均值上限为 0.75 nats。这两条是预先选定的短程接入筛选线。每个程序还验证达到容量时返回 context-full，以及 reset 后逐 token prefill 再 decode 的末行和预测能逐字节复现。此实验只判定指定模型的功能接入，不更新吞吐或语料质量结论。[输入、参考身份与门槛](../validation/m1/protocol.json)

# 运行时结构与 V1 接口

依据：当前归档源码与 [公开头文件](../include/micius_runtime.h)。日期：2026-10-03。ABI 为 `0x00060003`；CMake 项目版本 0.6.0 和 CLI 标题 v0.6-rebuild 是不同层面的标识。

## 模块职责

| 位置 | 主要职责 |
| --- | --- |
| src/app/MiciusGGUFParser.cpp、MiciusTensorCodec.cpp | 读取模型元数据、处理张量表示并向引擎提供 V1 绑定数据 |
| src/micius_runtime.cpp | C ABI、配置与张量边界验证、错误状态、公开逐 token 调用 |
| src/AI/AICoreEngine.cpp | 模型绑定、逐层 reference 前向、KV 与 logits 生命周期 |
| src/AI/ReferenceGEMVAVX2.cpp、PreparedGEMVInput.h | checked / prevalidated Q8_32 GEMV 与调用期 F64 输入 |
| src/AI/ReferenceAttentionAVX2.cpp | 有序 attention score 与 value 消费 |
| src/AI/PreparedGEMVWeights.cpp、ReferenceF32WeightGEMVAVX2.cpp | 保留的 PW2 实验依赖；当前 NONE |
| src/app/MiciusShell.cpp | CLI、真实模型基准、结果与诊断输出 |

受支持的权威路径是 Llama 架构下已有 loader/绑定规则覆盖的模型与单线程 reference。真实目标验收使用 SmolLM2 360M Instruct Q8_0；这不构成任意 GGUF 架构、量化类型或模型的通用兼容声明。

代码中保留 speculative、worker、旧池化等结构不表示它们已完成公开功能验收。当前请求 speculative execution 会被拒绝。部分 rotary 维度配置也不在当前支持范围内；不要绕过配置校验强行启用。

## 一次 reference 前向

输入 token 查表后，各层依次执行归一化、QKV 投影、RoPE、该层 KV 提交与 attention、O 投影及残差，再执行 FFN gate/up、激活组合、down 与残差。最后经过输出归一化和 LM head，得到 logits 和 greedy next token。

KV 按层和位置保存。达到 max_seq_len 后返回 context-full；不能把旧注释中的“环形”理解为会自动安全覆盖最旧的权威上下文。

“F64 reference”指受约束的计算与累加路径，不表示所有状态都改为 double。权重主存储仍为 Q8_32，激活、KV 和输出 logits 使用 F32。具体归约次序、scale 应用顺序和舍入位置属于数值契约的一部分。

当前 GEMV 不允许编译器随意融合乘加；这不等于整个工程不存在任何显式 FMA。attention 的有序实现有自己的运算顺序契约，E1 的条件化 FMA 则属于单独实验。

## 建立和绑定

1. 用 Micius_InitConfig 初始化配置，并填写模型拓扑、上下文长度、RMSNorm 与 RoPE 参数。
2. 用 Micius_CreateEngineV1 传入配置指针、sizeof(MiciusConfig) 和 MICIUS_ABI_VERSION。
3. 用 Micius_BindModelTensorsV1 传入带类型、尺寸和形状的条目及 sizeof(MiciusTensorEntryV1)。
4. 检查返回状态与错误；绑定成功后再进入逐 token 调用。
5. 不再使用时先销毁引擎，再释放引擎借用的模型数据。

MiciusTensorEntryV1 的维度按逻辑行主序解释，矩阵为 rows × columns。byte_size、data_type、rank、row_stride_bytes 与实际缓冲区必须一致。entry_size 决定条目步进，不能把它当成装饰参数。flags 当前使用 0，表示借用的数据在引擎使用期间有效。

Q/K 行必须已经是 GGUF 所需的 RoPE 相邻配对顺序，引擎不会再排列一次。再次排列会改变模型语义。

每个对称 32 元素权重块使用 MiciusQ8Block32Scale：一个 float scale、一个 int32 zero_point 和两个 padding word；zero_point 与 padding 必须为 0，scale 必须有限。GGUF Q8_0 的逐块 scale 不应合并成更大的量化块。

旧 Micius_BindModelTensors 和 Micius_LoadQuantWeights 缺少足够的安全绑定信息，保留符号但拒绝执行；新集成应使用 V1。

## 所有权与准备

| 数据 | 所有权与约束 |
| --- | --- |
| 通过 V1 借用的权重、embedding、norm 等数据 | 调用方保证有效且不可修改；保守做法是保持到引擎销毁 |
| 私有 scale descriptor | 绑定时验证并复制，由引擎管理 |
| packed QKV | 引擎拥有的打包缓冲区 |
| reference KV、工作区与 logits | 引擎拥有；reset 清理序列状态 |
| F64 输入 scratch | 每个 context 复用，容量按最大输入维度配置 |
| 某次准备后的 F64 输入视图 | 仅用于对应调用中的输入，不能跨输入修改后继续复用 |
| PW2 权重快照 | 当前 NONE，不分配实验 payload |

F64 输入 scratch 的主要数据容量为 8 × max(hidden_dim, intermediate_dim) 字节，另有对象与分配开销。准备阶段检查输入与相应浮点环境，FFN gate/up 共享同一个有效输入快照。失败条件不能被解释为允许无条件走预验证入口，应使用现有拒绝或 checked fallback 逻辑。

重绑定按候选状态构造、验证、提交；失败时保留原有效模型绑定。reset 清除会话序列、KV 和 logits 就绪状态，保留模型绑定。调用方不应将“reset”当作卸载模型或解除借用关系。

## Prefill 与 decode 语义

| API | 成功时的含义 |
| --- | --- |
| Micius_PrefillTokenV1 | 提交一个真实输入 token，不请求生成输出；返回 TOKEN_ACCEPTED |
| Micius_DecodeOneV1 | 提交一个输入 token，返回 greedy next token；返回 YIELD_TOKEN |
| Micius_GetSequenceLengthV1 | 已提交的输入 token 数 |
| Micius_CopyLastLogitsV1 | 复制最近一次成功前向的 logits；缓冲区至少 vocab_size 个 float |
| Micius_CopyReferenceKVV1 | 复制指定层与已提交位置的 K/V；每个输出缓冲区至少 num_kv_heads × head_dim 个 float |
| Micius_ResetEngineV1 | 开始新会话，保留绑定 |

对含 N 个 token 的非空 prompt，常用方式是 prefill 前 N−1 个，再对最后一个调用 DecodeOne 得到第一个生成 token。不要先 prefill 最后一个又用 DecodeOne 重复提交它。返回的 next token 只有在后续被作为输入提交时才进入 KV。

Micius_GetLastErrorCode / Micius_GetLastError 提供调用线程上的错误信息；应在失败后及时读取或复制。CONTEXT_FULL、NUMERIC、NOT_READY 等状态需要显式处理，不能无限重试同一个失败步骤，也不能把部分计算当成成功 logits。

V1 reference API 是当前新调用方的首选入口。旧 Micius_Step、遥测和树相关符号的存在不改变上述支持范围。当前文档不承诺同一 context 的并发调用安全性，也不将构建 preset 的 jobs=8 解释为八线程推理。

# 运行时与 V1 接口

v0.7 的 INT8 与 F64 共用 C ABI `0x00060003`。0.7.0 是源码发布版本，调用者从头文件读取 ABI 常量。模型走 `MICIUS_EXECUTION_REFERENCE` 的权威前向，数值后端由所链接的库决定。[公开头文件](../include/micius_runtime.h)

## 一次会话

1. 调用 `Micius_InitConfig`，填入拓扑、上下文上限、RMSNorm 与 RoPE 参数。用 `Micius_CreateEngineV1` 传入配置、结构大小和 ABI 号。
2. 用 `Micius_BindModelTensorsV1` 传入带类型、形状、长度和行步长的条目。矩阵按逻辑行主序。Q、K 使用已约定的 RoPE 相邻配对顺序。失败的重绑定保留原有效绑定。
3. 对长度 N 的非空 prompt，前 N−1 枚用 `Micius_PrefillTokenV1`，最后一枚用 `Micius_DecodeOneV1`。预测 token 在后续被提交时进入上下文。
4. `Micius_CopyLastLogitsV1` 需要至少 `vocab_size` 个 F32。`Micius_CopyReferenceKVV1` 按层和已提交位置复制 K/V。遇到 context-full、非法输入或数值异常应按返回码处理。
5. `Micius_ResetEngineV1` 清除序列、KV 和 logits 就绪状态，保留已绑定模型。销毁 context 后再释放外部借用张量。同一 context 的调用由调用方串行化。

## 加载与所有权

GGUF 原件由解析器只读映射。Q8_0 的每 32 个权重有一个 scale，加载器构造 V1 的 Q8_32 描述。绑定检查 finite scale、zero point、padding、形状、长度和 stride。embedding 按模型编码取得或转换为 F32。

| 对象 | 生存期 |
|---|---|
| 外部 V1 张量 | `flags=0` 表示借用。调用者按 ABI 保持数据有效。Shell 私有退休流程仅释放其自身拥有并核对过身份的缓冲 |
| 四行权重面板、F32 block scale | 绑定时准备。引擎持有至重新绑定或销毁 |
| 原 Q8 和公开 scale | 面板绑定成功后，加载器核对指针、长度、类型和别名，再退休旧 Q8。I22 行布局回退仍借用 Q8。已复制的公开 scale 可在绑定成功后释放 |
| 激活 scratch | context 内复用。每块产生有符号字节和 F64 scale。面板另用最多 `4 × max(hidden, intermediate)` 字节展开区 |
| F32 KV、logits 和工作区 | 引擎按拓扑与上下文分配。申请容量、已用位置和物理驻留分别统计 |

I25 先准备完整候选状态，再提交指针切换。词表或打包 QKV 的行数不满足四行条件时，整模型走 I22。分配或验证失败返回错误。故障注入曾逐一验证 91 个重绑定分配失败点。[I25 研究记录](history/int8/I25.md)

## 逐 token 前向

embedding、RMSNorm、QKV、RoPE、KV 写入、有序 attention、O 及残差、SwiGLU FFN、输出归一化和 LM head 依次运行。INT8 用于矩阵消费。FFN gate/up 共用量化快照和一次展开。KV、模型状态与 logits 用 F32，注意力保持各自固定的累加和舍入顺序。[数值契约](INT8_NUMERICS_AND_QUALITY.md)

实现入口见 [GGUF 解析器](../src/app/MiciusGGUFParser.cpp)、[Shell 加载器](../src/app/MiciusShell.cpp)、[V1 绑定](../src/micius_runtime.cpp)和[推理引擎](../src/AI/AICoreEngine.cpp)。[v0.6 运行时记录](history/v0.6/RUNTIME_ARCHITECTURE_AND_API.md)

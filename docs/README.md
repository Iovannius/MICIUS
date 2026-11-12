# MICIUS 技术文档

当前发布版本为 **v0.7.2**，纳入已验收的 M3。[发布说明](RELEASE_0_7_2.md) 记录实机结果和使用边界，[快照、采样和续聊](M3-PERSISTENCE-AND-SAMPLING.md) 说明功能、ABI 与复现协议。[v0.7.1 发布说明](RELEASE_0_7_1.md)、[M2 文字对话](M2-TEXT-CHAT.md)、[M1 模块 SDK](M1-MODULES-AND-ABI.md) 保留各自的历史验收范围。

v0.7 的 I1–I25 研究记录保留对应实验时的模型、程序和评估口径。

| 我想了解 | 文档 |
|---|---|
| 当前支持哪些模型和操作 | [当前状态](CURRENT_STATUS.md) |
| 编译、启动、选择 INT8 或 F64 | [构建与配置](BUILD_AND_CONFIGURATION.md) |
| 在自己的程序中接入 V1 | [运行时、张量与所有权](RUNTIME_ARCHITECTURE_AND_API.md) |
| 发布测试验证了什么 | [验证与测量口径](VALIDATION_AND_BENCHMARKING.md) |
| 激活量化的误差怎样影响自然文本 | [数值契约与质量评估](INT8_NUMERICS_AND_QUALITY.md) |
| 性能、分区间收益和内存取舍 | [性能与内存研究](INT8_PERFORMANCE_AND_MEMORY.md) |
| 每项候选为何采用或停止 | [I1–I25 实验台账](INT8_EXPERIMENT_LEDGER.md) |
| 从 F64 到 INT8 的决策顺序 | [开发历史](DEVELOPMENT_HISTORY.md) |
| 报告与构建身份 | [v0.7.2 源码清单](../RELEASE-MANIFEST.json) · [固定实机协议](../validation/m3/protocol.json) · [INT8 研究来源](evidence/int8-report-sources.json) · [v0.7 发布验证](evidence/v0.7-verification.json) |

## F64 参考与历史

v0.7 的 F64 preset 沿用 v0.6 的数值配置。历史报告中的“默认”和“下一步”以各自报告日期为准。

| 主题 | 资料 |
|---|---|
| v0.6 的状态、入口和复现记录 | [v0.6 文档快照](history/v0.6/README.md) |
| F64 候选与收尾 | [实验台账](F64_EXPERIMENT_LEDGER.md) · [CF7 收尾](F64_CF7_CLOSEOUT_2026-10-02.md) |
| Q8_32 的 F64 消费 | [AVX2](Q8_32_AVX2_BACKEND.md) · [双行归约](Q8_32_GEMV_ROW_PAIR_AVX2.md) · [scale 预验证](Q8_32_PREVALIDATED_SCALES.md) |
| 有序 Attention | [Value](ATTENTION_VALUE_AVX2_BACKEND.md) · [Score](ATTENTION_SCORE_TIME_QUAD.md) |
| 早期重建、候选和测量 | [历史索引](history/README.md) |

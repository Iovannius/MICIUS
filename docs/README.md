# MICIUS 文档入口

更新日期：2026-10-03。当前交付采用已验收的 F64 reference 基线：Q8_32 权重、AVX2 双行 GEMV、调用期 F64 输入快照和有序 attention。CF5 未进入当前运行时。

## 当前文档

| 文档 | 用途 |
| --- | --- |
| [当前状态](CURRENT_STATUS.md) | 当前启用路径、最新 Windows 验收、性能记录与已知边界 |
| [开发历史](DEVELOPMENT_HISTORY.md) | 从 reference 重建到 F64 阶段收尾的决策过程 |
| [运行时与 V1 接口](RUNTIME_ARCHITECTURE_AND_API.md) | 模块职责、数值边界、模型所有权与逐 token 调用语义 |
| [构建与配置](BUILD_AND_CONFIGURATION.md) | 已验收 preset、选项关系、目录与诊断功能 |
| [F64 实验台账](F64_EXPERIMENT_LEDGER.md) | 各候选的比较对象、证据范围及最终处置 |
| [验证与测量口径](VALIDATION_AND_BENCHMARKING.md) | bitwise 验证范围、吞吐可比条件与准备摊销 |
| [历史报告索引](history/README.md) | 本次补回的 RB1 至 CF4 阶段报告 |
| [F64 源码交付说明](F64_REFERENCE_2026-10-02.md) | 2026-10-02 覆盖包来源、构建入口与回退说明 |
| [CF7 收尾](F64_CF7_CLOSEOUT_2026-10-02.md) | CF5 未通过性能筛选的机器码和运行时证据 |

本次文档补全只增补 `docs/`，不更改源码、配置、脚本和既有文档。没有重新构建、运行测试或开展性能实验。输入为用户清理后的 `MICIUS (2).zip`；归档 SHA-256 为 `e65eb6410427459c2c5e5de6c5194704c4d783e4fa49319908fd95a9d061e41d`。对照既有 [运行时来源清单](f64-closeout-evidence/runtime-source.json)，48 个输入内容一致。

补全文档 ZIP 自带 `MICIUS/docs/`，解压到 `C:\Projects` 后，本页位于 `C:\Projects\MICIUS\docs\README.md`。包内 [增补清单](DOCS_UPDATE_2026-10-03.json) 记录新增文件散列；原根目录 `FILES.sha256` 保留为原源码交付包的历史清单，不用于覆盖本次新增文档。

## 既有设计文档

| 主题 | 文档 |
| --- | --- |
| checked Q8_32 AVX2 | [Q8_32_AVX2_BACKEND.md](Q8_32_AVX2_BACKEND.md) |
| 绑定边界 scale 预验证 | [Q8_32_PREVALIDATED_SCALES.md](Q8_32_PREVALIDATED_SCALES.md) |
| r5/r6 双行与寄存器约束 | [Q8_32_GEMV_ROW_PAIR_AVX2.md](Q8_32_GEMV_ROW_PAIR_AVX2.md) |
| 有序 V 聚合 | [ATTENTION_VALUE_AVX2_BACKEND.md](ATTENTION_VALUE_AVX2_BACKEND.md) |
| r7 query 缓存实验 | [ATTENTION_QUERY_F64_CACHE.md](ATTENTION_QUERY_F64_CACHE.md) |
| r8 四时间行 score | [ATTENTION_SCORE_TIME_QUAD.md](ATTENTION_SCORE_TIME_QUAD.md) |
| r9 compact scale 实验 | [Q8_32_COMPACT_SCALES.md](Q8_32_COMPACT_SCALES.md) |
| r1 F64 输入准备 | [F64_CLEANUP_R1.md](F64_CLEANUP_R1.md) |
| R8 ITT 取样历史 | [R8_DECODE_ITT_2026-09-08.md](R8_DECODE_ITT_2026-09-08.md) |

## 重建与早期验证记录

[重建计划](REMEDIATION_PLAN.md)、[首次验证](VERIFICATION_RESULTS.md)、[第二轮验证](SECOND_ITERATION_VERIFICATION.md)、[CMake 重建](CMAKE_RECONSTRUCTION_VERIFICATION.md)、[sandbox 重建](SANDBOX_REBUILD_VERIFICATION.md)、[Windows Clang64 修复](WINDOWS_CLANG64_BUILD_FIX.md) 和 [早期 checkpoint](CHANGELOG_CHECKPOINT.md) 保留各自形成时的状态。

这些文件中的“下一步”“尚待目标机测量”“默认关闭”描述其当时阶段。判断当前启用项应以当前状态、实际 preset 和最新验收记录为准。例如 r8 原文的目标机待验收状态不代表今天仍未验收；r7、r9 的代码留存也不代表当前已启用。

历史报告中的旧绝对路径、实验包文件名及外部结果引用属于来源线索。本次不补回被清理的构建目录，也不将所有原始实验材料重新嵌入源码包。

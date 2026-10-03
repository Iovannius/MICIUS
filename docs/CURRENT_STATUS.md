# 当前状态：F64 reference

记录日期：2026-10-03。范围：当前 Q8 存储、F64 数值路径及已有 bitwise 验证协议。

当前可用成果是已验收的 reference 基线与可复现的 Windows 构建配置。CF5 通过实验正确性检查，但 CF7 端到端性能筛选未通过，因此不在当前源码数值路径中。F64 阶段收尾不等于证明路线已经达到性能上限。

## 当前启用路径

| 项目 | 当前选择 |
| --- | --- |
| 权重 | Q8_32；每 32 个权重对应一个私有 scale descriptor |
| GEMV | AVX2 双行共享输入，F64 累加，参考 GEMV 的 fp-contract 关闭 |
| 输入准备 | 每个输入的 F64 快照；FFN gate/up 复用一次准备 |
| scale | V1 绑定边界预验证，保留 checked fallback |
| attention value | 保持原运算顺序的 AVX2 聚合 |
| attention score | 四个时间行共享 query 转换 |
| r7 query F64 缓存 | 关闭 |
| r9 compact scale | 关闭 |
| PW2 展开权重 | NONE；额外权重 payload 为 0 |
| D1、ITT | preset 中关闭；既有诊断入口保留 |
| CF5 | 未合入，无当前构建入口 |
| 公共 ABI | 0x00060003，权威执行模式为单线程 reference |

配置来源见 [CMakePresets.json](../CMakePresets.json)。CLI 输出中的 `experimental` 是后端标识的一部分，不能据此判断 CF5 已启用。源码中保留 PW2 和诊断依赖，也不表示这些实验已通过性能验收。

## 最新 Windows 交付验收

用户当前归档保留 [构建日志目录](../logs/f64-reference-build.VF6vjH/)：

| 记录 | 结果 |
| --- | --- |
| 源码来源检查 | 48 个运行时输入通过 |
| 编译器 | Clang 22.1.8，target x86_64-w64-windows-gnu |
| CMake / Ninja | 4.4.2 / 1.13.2 |
| 构建 | CLI、sandbox 与测试构建完成 |
| CTest | 11/11 通过，日志总时间 1.49 s |
| 编译目录 | 用户随后清理；历史构建日志保留 |

Clang commit 为 `6e4e79c2f86eeb534e324e583f2057dc9fd5ecab`。日志记录的 CLI SHA-256 为 `d14cb40ad442f6d2fff00873e2b2adbda7f59ba7cab0a6053181078f4aec2fec`，sandbox 为 `e369fd0c628d17d29feee57b872992b33fca0ed3c5c45d4863d73615a5c98917`。这些是历史构建日志中的身份记录；本次文档整理未重新生成或核验已删除的可执行文件。

本次只读取归档并核对 48 个来源输入，内容均与既有可信清单一致。此核对不等于在当前主机重跑目标机验收。

## 最新真实模型观察

用户在 `C:\Projects\MICIUS\build-f64-reference\bin\micius_cli.exe` 上回传的记录：

| 字段 | 值 |
| --- | --- |
| 模型 | smollm2-360m-instruct-q8_0.gguf |
| 协议 | prompt token=1，warmup=8，timed steps=128，CPU=4 |
| status / completed_steps | PASS / 128 |
| wall_seconds | 6.477332 |
| steps_per_second | 19.761224 |
| model_load_and_bind_seconds | 0.8336093 |
| final_token | 30 |
| token_checksum | 0x4f9cb9a464361ddf |
| last_logits_checksum | 0x796b3ffb22c87b08 |
| PW2 / D1 / ITT | NONE / 未编译且未请求 / 未请求控制回调 |

完整用户回传文本保存在 [本次运行记录](evidence/2026-10-03-user-model-benchmark.txt)。文件名日期是整理日期，原回传没有独立的执行时间戳。运行文本未包含可执行文件散列，因此不能仅凭路径把它与某一 EXE 散列做独立的身份认证。

用户另报告多次运行在过热降频前约为 **19.8 tok/s**。这支持“该短协议在当时环境可重复达到这一量级”的判断，可以作为当前可用表现。尚无完整重复样本、温度和频率序列，不能据此计算置信区间，也不能把它写成长时间稳态吞吐。

它不是新改动的配对 A/B，不能将其相对 CF7 BASE 的数值差异归为源码优化收益。128 步记录验证 token 序列摘要与最后一步 logits 摘要，不等价于 128 步所有 logits 的逐字节对照。

## CF7 的最终处置

| 指标 | CF7 已回传记录 |
| --- | --- |
| BASE / CF5 中位吞吐 | 18.874862 / 18.7785235 tok/s |
| 中位配对吞吐收益 | −0.546467854% |
| 配对获胜 | 1/6 |
| CF5 额外 payload | 1,096,581,764 B |
| CF5 准备中位时间 | 0.717743 s |
| BASE / CF5 load+bind 中位时间 | 0.766081 / 1.4857698 s |
| 决策 | 不将 CF5 作为性能优化合入当前运行时 |

该结果不足以断言 CF5 在所有场景稳定慢 0.55%；它足以说明本轮没有得到批准性能合入所需的正向证据。正确性通过与性能通过是不同条件。

## 已完成与仍未知

已完成的是可信 BASE 的来源核对、收尾交付，以及用户实际 Windows 构建和本协议运行确认。当前无需为文档补全重开已完成的测试。

仍未知的是持续热稳态表现、历史跨批次约两倍吞吐变化的因果分解，以及 CF7 中调用、缓存、带宽和频率因素的独立贡献。LeakSanitizer 曾受宿主条件限制关闭，不能记为泄漏检查通过。上述未知有明确边界，不妨碍保留已验收 BASE。

后续若提出新的 F64 候选，需要新的完整成本与正确性证据；当前不设 22–30 tok/s 的完成态预测。高效 reference 是工程目标；INT8/VNNI 是后续路线，不预先承诺收益。

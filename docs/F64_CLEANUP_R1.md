# F64 cleanup r1：输入准备与固定数值契约

日期：2026-09-09。状态：本地正确性验证完成，Windows 目标机性能和机器码审查待完成。`MICIUS_REFERENCE_GEMV_PREPARED_F64` 默认 **OFF**。

## 本轮实验问题

现有 R8 row-pair 已经在两行间共享输入转换，但不同 row-pair 仍会重复把同一输入从 F32 转成 F64。本轮测量一次准备、跨输出行复用，并在 FFN gate/up 之间共享准备结果是否改善完整 decode 吞吐。

权重仍按 Q8 存储，每个权重在热循环中拓宽一次。没有将完整权重预展开为 F64，也没有把这次必要拓宽称为重复工作。此前驻留数据预展开实验的约 68.8% 吞吐提升不作为本轮性能预测。

候选新增每个 reference Context 一块 `8 * max(hidden_dim, intermediate_dim)` 字节的输入 scratch。准备扫描验证有限值并转换，之后为只读快照；不使用按地址命中的持久缓存。F64 输入读取量比原 F32 输入更大，是否抵消转换减少的收益由真实模型 A/B 决定。

## 实现边界

| 文件/边界 | 改动 |
| --- | --- |
| `src/AI/ReferenceGEMVAVX2.cpp` | 将现有 Q8 reference GEMV 从混合内核文件中分出；保留原 F32 入口，增加 prepared F64 输入入口 |
| `src/AI/PreparedGEMVInput.h` | 内部借用视图、输入验证/准备、全局输出行范围接口 |
| `src/AI/AICoreEngine.cpp` | Context 拥有 scratch；普通 GEMV 每次准备，FFN gate/up 共同读取一次准备结果 |
| `include/AI/AICoreEngine.h` | 添加 scratch 成员；成员布局不随内部宏改变，分配由开关控制 |
| `tests/reference/r8_q8_frozen.cpp` | 独立保留原 R8 四链实现作为严格浮点测试参考，不调用候选内部 helper |
| `tests/prepared_gemv_tests.cpp` | 逐位、分区、尾部 contraction、失败边界、只读并发消费者测试 |
| CLI / sandbox | 输出后端、输入准备方式与 reference GEMV contraction 身份 |
| 旧 `r8-q8-f64-ab` 实验 | 更新引用源文件和源码指纹清单，保持实验可独立构建 |

保持原来的 `(input * integer_weight) * scale` 操作顺序、四条 F64 累加链、规约次序和最后 F32 输出转换。保留 checked fallback；prevalidated / compact 入口仍只允许原私有 V1 模型绑定边界使用。没有改变 C ABI、模型量化方式、KV/attention 算法或运行时调度。

## A/B 两边共同的浮点修复

原 `FAST_MATH=OFF` 不能单独代替明确的 contraction policy。分离后的 reference GEMV 编译单元使用 `-fno-fast-math -ffp-contract=off`，Clang 再显式关闭 contraction/reassociation；GCC 也指定 contraction off。MSVC 分支为 `/fp:strict`，本轮未在 MSVC 验证。这些设置同时作用于 OFF/ON 两边，其他 legacy 内核不随之全局改写。

选项语义参见 [Clang Users Manual](https://clang.llvm.org/docs/UsersManual.html) 和 [GCC Optimize Options](https://gcc.gnu.org/onlinedocs/gcc/Optimize-Options.html)。

新测试的 cols=33 消去算例要求结果为 F32 `0x00000000`。若把最后乘加融合，会得到可表示的 `0xa8800000`；测试同时确认该差别确实存在。这是尾部契约修复，因此不承诺复现历史二进制所有尾部位型。

本轮 OFF 是“分离编译单元并固定 contraction 后的 R8 F32 输入路径”；ON 再启用 prepared F64。两者吞吐之比衡量本轮准备/验证复用的整体效果，不能把相对历史 R8 二进制的变化都归因于它。目标 SmolLM2 日志仍以此前相同 workload 的 checksum 作为停止检查。

## 面向后续多核的约束

- 行接口接收完整矩阵/输出基址与全局 `[row_begin, row_end)`；非空范围开始必须为偶数行，结束为偶数行或矩阵末尾。奇数末行只有最后任务拥有，合法空范围不写输出。
- 每个输出行由一个消费者完整计算，不做跨线程浮点部分和。权重、scale、prepared activation 共享只读；当前累加器为调用栈/寄存器私有。
- 准备快照与 scratch 的所有者负责生命周期。所有消费者完成后才能复用、reset 或销毁；即使某个任务失败，也必须等其他任务结束，并禁止下游读取部分输出。
- 输入源在准备后可以变化；scratch 在消费期间不能变化。视图不以输入地址作为有效性依据。线程 MXCSR 控制位必须与准备时一致；异常 sticky flags 不参与相等检查。
- 本轮运行时为同步调用。测试用两个线程检查只读消费者和分区行为，不代表已实现单序列多核推理或线程池。

## 已完成的本地验证

验证环境：Linux、GCC 13.3、`ARCH=haswell`。以下都不是 Windows Clang 22.1.8 的目标机结果。

| 验证 | 结果与范围 |
| --- | --- |
| OFF 与 ON，Release / O3 / LTO ON | 两边各 9/9 CTest PASS |
| 新逐位测试 | 252 组形状/输入，含尾列、奇数末行、非对齐输入、反序分区；对照独立 R8 严格参考 |
| 失败与生命周期测试 | 无效视图、NaN/Inf、scratch 不足、范围/别名、控制位改变、准备失败后失效、输入地址复用等 PASS |
| 两个只读消费者 | 匹配 MXCSR、两种 FTZ/DAZ 环境下，分区结果逐位符合参考 |
| ASan + UBSan | 新内核与相关测试 PASS；环境不支持 LSan，运行时 `detect_leaks=0` |
| prepared + compact / prepared + checked | 每套运行新内核测试与 runtime contract，分别 2/2 PASS；LTO OFF |
| 无 AVX2 / 无 row-pair / fast-math 冲突 | CMake 和直接编译头文件均按预期拒绝 |
| sandbox smoke | 两边均 PASS，7 次验证；本地 CPU 0 的 checksum 均为 `0x182dd19b1fb45fb6` |
| 原驻留 A/B 实验兼容性 | 源文件分离后重建，1/1 自测 PASS；未重新采集该实验性能 |
| 本地生成代码 | GCC O3 非 LTO 的 reference GEMV 对象反汇编未出现 FMA；不等同于 Windows ThinLTO 热循环审查 |

独立参考提取前的 `src/AI/AIComputKrnl.cpp` 原始 SHA256：

```text
ee26c5fbbef2af8a4a70c1f6dac06c6e0d295a17c29ef6d26e6c4ff61b1a3312
```

本地 smoke checksum 只比较这两套本地构建，不能与历史 Windows sandbox checksum 直接比较。MSVC、其他架构、完整真实模型和候选 Windows 机器码均未在此环境验证。

## Windows 安装与测量

把修改包中的 `MICIUS-f64-cleanup-r1` 文件夹解压到 `C:\Projects\MICIUS` 内。安装器先核对全部待替换文件，接受原基线、已应用的新内容及仅 CRLF/LF 差别；遇到其他修改先停止。通过核对后备份原件并替换，安装失败会尝试恢复；无需 `git apply`、外部依赖目录或下载。

在 MICIUS 根目录的 Git Bash 一次执行：

```bash
bash <<'BASH'
set -euo pipefail
test -f CMakeLists.txt
bash MICIUS-f64-cleanup-r1/install.sh
bash scripts/build-f64-cleanup-windows.sh
bash scripts/run-f64-cleanup-windows.sh
BASH
```

构建目录要求尚不存在：`build-f64-cleanup-r1-baseline`、`build-f64-cleanup-r1-candidate`。可执行文件均在各自的 `bin/micius_cli.exe`。公共参数沿用 R8：Clang、Ninja、Release、native、ALIGN=32、legacy tuning/LTO/LLD/static ON、AVX2 row-pair/prevalidated ON、compact OFF、attention value/time-quad ON、query-cache/fast-math OFF。ITT OFF，本轮无需 VTune。

测量固定读取 `models/smollm2-360m-instruct-q8_0.gguf`，单线程 reference forward、CPU 4、token 1。先 OFF/ON 各一次 warmup=4、steps=32 smoke，再交错执行五对 warmup=8、steps=128。构建脚本两边运行全套 CTest；运行脚本校验源码/二进制/模型指纹、运行身份、CPU 起止、完成步数及历史 token/logits checksum。

准入条件在目标测量前固定：候选至少胜 4/5，普通中位数吞吐提升严格大于 2%，配对中位数提升严格大于 2%。未达到就保留 OFF、结束这项候选的性能尝试；不追加轮次追逐噪声。达到后仍需要审查实际 Windows 机器码，才讨论保留/冻结。

结果自动保存为 `f64-cleanup-ab.XXXXXX.zip`，包含各轮日志、统计、指纹、源码快照及实测 EXE/PDB；检测到 `llvm-objdump` 时导出完整反汇编。测量错误也会打包已取得日志并停止。`performance_gate=FAIL` 与脚本完整性错误分别报告；`f64_cleanup_target_comparison=PASS` 仅表示采集/数值检查完成，不代表性能通过。

## 后续阶段

本轮只提交这一项 F64 性能候选。复核目标结果后冻结明确的数值参考，避免无限扩张 F64 清理范围。独立原始参考与优化后的性能基线继续保留不同身份。

后续 AVX2 W8A8 将另行定义 activation 量化、舍入、零块、scale 和整数累加范围；相对 F64 使用明确误差预算及模型级差分，不要求量化路线变化后 bitwise 相同。AVX2 与之后 AVX-VNNI 的同算法实现才要求整数结果及相同浮点后处理一致。性能测试需包含准备成本，最终以真实模型完整 decode 判断。

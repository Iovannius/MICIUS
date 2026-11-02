# 构建、配置与目录约定

依据：当前 [CMakePresets.json](../../../CMakePresets.json) 与 [CMakeLists.txt](../../../CMakeLists.txt)。日期：2026-10-03。本次文档工作没有执行以下构建或运行入口。

## 已验收的 Windows 入口

工程默认位置为 C:\Projects\MICIUS。原 MICIUS Windows Bash 中的绝对路径入口为：

```bash
bash /c/Projects/MICIUS/scripts/build-f64-reference-windows.sh
```

其 preset 名为 f64-reference-windows，构建目录为 build-f64-reference。需要 Windows x64 Clang64、Python、Ninja 和支持当前 preset 的 CMake ≥ 3.21；源码使用 C++20。最新保留的目标记录为 Clang 22.1.8、CMake 4.4.2、Ninja 1.13.2。

脚本核对 48 个运行时来源输入及遗留实验源码，拒绝覆盖已有同名构建目录，随后构建并执行 CTest。它不自动运行真实模型性能实验。日志使用独立的 logs/f64-reference-build.* 目录。当前用户已清理构建目录，后续若需可执行文件应从该配置重新生成；历史日志本身不会恢复二进制。

详细覆盖与回退流程见 [原源码交付说明](../../F64_REFERENCE_2026-10-02.md)。本次 docs 增补不要求重新构建。

## Preset 的明确选择

下表选项均带 MICIUS_ 前缀，ARCH、ALIGN、CMAKE_BUILD_TYPE 除外。

| 选项 | 值 | 含义 |
| --- | --- | --- |
| CMAKE_BUILD_TYPE / ARCH / ALIGN | Release / native / 32 | 目标机本地构建配置 |
| ENABLE_LEGACY_TUNING | ON | 保留项目调优配置 |
| ENABLE_FAST_MATH | OFF | 保留有限值检查与既定数值边界 |
| ENABLE_LTO / USE_LLD / STATIC_RUNTIME | ON | 已验收的链接配置 |
| REFERENCE_GEMV_AVX2 | ON | Q8_32 AVX2 |
| REFERENCE_GEMV_ROW_PAIR_AVX2 | ON | 双行共享输入 |
| REFERENCE_GEMV_PREVALIDATED_SCALES | ON | 可信绑定边界预验证 |
| REFERENCE_GEMV_PREPARED_F64 | ON | F64 输入快照 |
| REFERENCE_GEMV_COMPACT_SCALES | OFF | 保留 descriptor 格式 |
| REFERENCE_ATTENTION_VALUE_AVX2 | ON | 有序 V 聚合 |
| REFERENCE_ATTENTION_QUERY_F64_CACHE | OFF | 不使用 r7 缓存 |
| REFERENCE_ATTENTION_SCORE_TIME_QUAD | ON | r8 四时间行 |
| PW2_SELECTION / PW2_BUDGET_MIB | NONE / 512 | 不启用展开权重；预算值不代表实际分配 |
| DECODE_PROFILE / BENCHMARK_ITT | OFF / OFF | 普通测量关闭诊断 |

BUILD_TESTING、MICIUS_BUILD_CLI 和 MICIUS_BUILD_SANDBOX 均为 ON。build preset 的 jobs=8 只控制编译并行度。

CMake 通用默认值与本 preset 不同：多个实验后端选项在通用配置中仍为 OFF。直接执行普通配置或复用旧 cache，不能据此声称复现了当前基线。ARCH=native 生成的二进制也不是面向所有 CPU 的可移植发行包。

## 配置关系

- ROW_PAIR_AVX2 依赖 GEMV_AVX2；PREPARED_F64 依赖这两项。
- PREVALIDATED_SCALES 与 PREPARED_F64 的相关路径要求可靠的有限值检查，不允许与 FAST_MATH 混用。
- PW2 的非 NONE 选择依赖 prepared 输入和 prevalidated descriptor scales，并要求 compact scales 关闭。
- 其他候选组合以 CMake 的显式配置校验为准；不要通过绕过检查得到一个仍自称已验收的配置。
- 更换编译器、LTO、ARCH 或数值选项后，原目标机机器码与性能数值不能直接继承。

## 目录与诊断的角色

| 目录或文件 | 用途 |
| --- | --- |
| src、include、tests | 当前实现、接口与测试来源 |
| scripts、CMakePresets.json | 可复现配置与检查入口 |
| docs/f64-closeout-evidence | 冻结来源、历史验证与机器码证据 |
| docs/history | 按日期保存的实验报告 |
| logs/f64-reference-build.* | 用户实际构建记录 |
| experiments、旧 results 或历史包 | 实验材料；存在不等于启用 |
| build-f64-reference | 可再生构建输出，当前已清理 |
| models | 本机模型数据，不由文档包提供 |

当前归档可能仍含旧结果目录内的 EXE/PDB 或反汇编，这是历史证据，不是当前构建目录仍存在的证明。

D1 与 ITT 在普通 preset 中关闭。保留的 ITT SDK、ROI 辅助代码或日志不意味着采集器正在运行；普通输出中的 VTUNE_ROI_BEGIN/END 标记也不能独立证明实际启用了 ITT 收集。以编译状态、请求状态和工具日志为准。未来确需 profiling 时再按对应历史协议使用独立构建，不把带诊断开销的数值混入普通吞吐。

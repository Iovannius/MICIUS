# R8 decode-only ITT instrumentation, r1 — 2026-09-08

状态：插桩实现及本地回归完成；Windows 真实模型检查、ITT/PMU 区间确认和微架构采样 **NOT RUN**。这是诊断补丁，不是新的 GEMV 性能候选。R9 compact-scales 继续默认 OFF。

基于当前带真实模型 benchmark 的 R9 工作副本，以 compact OFF 选择 R8 后端。原工作副本与旧归档保留。公共 ABI 仍为 `0x00060003`，16-byte descriptor ABI、GEMV/attention 数学和所有实验后端默认值均未改变。

## 修改及区间契约

| 文件 | 修改 |
| --- | --- |
| `CMakeLists.txt` | 新增默认 OFF 的 `MICIUS_BENCHMARK_ITT`、外部 SDK 路径 `MICIUS_ITT_ROOT`、SDK C 静态库及第八项 CTest |
| `src/app/MiciusShell.cpp` | 解析 `--itt-roi`；在加载前初始化并暂停，在既有计时循环两侧恢复/暂停；输出身份 |
| `src/app/BenchmarkIttRoi.h` | 私有控制适配及 RAII；错误或异常退出不遗留开启的区间 |
| `tests/benchmark_itt_roi_tests.cpp` | 生命周期、重复 Begin/End、异常退出、初始化失败、真实 SDK 无采集器及 CLI 加载前拒绝 |
| `scripts/build-r8-itt-windows.sh` | 与原 R8 配置对应的一次新目录构建和全部 CTest |
| `scripts/profile-r8-itt-windows.sh` | 一次原有 checksum 检查、一次 decode 区间 uarch-exploration 及报告导出 |
| 本文 | 实现、验证配置、证据边界和操作说明 |

只有同时编译 ITT 支持并传入 `--itt-roi`，benchmark 才控制采样。默认 OFF 不需要 C 编译器或 SDK。开启支持但不传此参数时，普通 benchmark 不调用 ITT 控制接口。

目标采集使用 `-start-paused`，排除 main 之前和 SDK 初始化之前的启动阶段。程序在模型加载前调用 pause，完成 prompt、warmup、缓冲区分配和身份输出后恢复采样；既有 128-step decode 计时结束立即暂停，再做最后 logits 拷贝、checksum 和输出。恢复/暂停调用在 `wall_seconds` 计时之外；硬件窗口包含两次时钟读取等少量边界指令。没有每 token 或每 GEMV 的插桩。[Intel start-paused 文档](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2023-1/start-paused.html)

SDK 初始化后通过 `__itt_get_collection_state()` 检查初始化成功，并检查 pause/resume 指针。仅检查指针的初版存在漏洞：无采集器时 SDK 仍保留非空空操作 stub。真实 SDK 负向运行发现此问题后已修复并补充回归。初始化成功不独立证明 PMU 已正确响应：`profiling_roi=itt_decode_only` 和 `itt_control_callbacks=bound` 是程序侧状态，仍须核对采集报告与时间线。脚本因此输出 `roi_hardware_boundary=REVIEW_REQUIRED`。

ITT SDK 固定为官方提交 `7579d45242ddc617c768e7ff4377395368342de5`，由同一工具链编译 `ittnotify_static.c`。不调用 SDK 的根 CMake，不改变 MICIUS 的优化配置，不链接 MSVC 预编译 `.lib`。SDK 不收入交付源码；首次 Windows 构建脚本在项目旁的 `MICIUS-deps/` 建立独立的固定提交副本。也可通过环境变量 `MICIUS_ITT_ROOT` 指定完整且包含 collection-state API 的 SDK。[官方 SDK 源码](https://github.com/intel/ittapi/tree/7579d45242ddc617c768e7ff4377395368342de5)

## 本地验证记录

Linux x86-64，GCC/G++ 13.3.0，CMake 4.4.3，Ninja 1.13.2。以下是最终源码的结果，不是 Windows 结果或性能数据。

| 检查 | 结果 |
| --- | --- |
| R8 实验组合，ITT 默认 OFF | 完整构建；8/8 CTest PASS |
| 同一 R8 组合，ITT ON | 完整构建；8/8 CTest PASS |
| ITT ON，ASan+UBSan | 完整构建；8/8 CTest PASS |
| LeakSanitizer | NOT RUN，显式 `detect_leaks=0` |
| ITT ON 缺少 SDK | 配置阶段按预期拒绝，PASS |
| ITT ON 无采集器 | 真实 SDK 初始化状态及 CLI 加载前拒绝，PASS |
| Windows Bash 脚本 | `bash -n` PASS；目标机执行 NOT RUN |
| 已有文件差异 | 仅修改 CMake 和 Shell；数值算子及 timed decode 循环逐字节未变 |
| 本轮 SmolLM2 全 logits/token parity | BLOCKED：本地模型副本仅 18,460,672 bytes，不能作为完整模型使用；未重建外部 oracle |
| Windows 模型 checksum 与硬件采集 | NOT RUN；下方脚本使用用户目标机已有完整模型 |

每种本地配置只建立一个目录。初版 OFF/ON 构建后，真实 SDK 负向检查揭示上述空操作 stub 问题，修改后做了增量构建和全量 CTest，没有重复 clean build。ASan 初次构建因 sandbox 对象文件为 0 bytes 而链接失败（undefined main）；保留该文件为 `.o.empty` 后仅重编译未完成对象和已修改源码，完整构建及测试通过。未因此修改 sandbox 源码。

两个普通构建的完整选项相同，ITT 开关及其 C/SDK 依赖除外：

```text
Generator=Ninja
CMAKE_CXX_COMPILER=g++
CMAKE_BUILD_TYPE=Release
ARCH=native
ALIGN=32
BUILD_TESTING=ON
MICIUS_ENABLE_LEGACY_TUNING=ON
MICIUS_ENABLE_LTO=ON
MICIUS_USE_LLD=ON (本地未找到 LLD，CMake 使用默认链接器)
MICIUS_STATIC_RUNTIME=ON
MICIUS_REFERENCE_GEMV_AVX2=ON
MICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2=ON
MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES=ON
MICIUS_REFERENCE_GEMV_COMPACT_SCALES=OFF
MICIUS_REFERENCE_ATTENTION_VALUE_AVX2=ON
MICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE=OFF
MICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD=ON
MICIUS_ENABLE_FAST_MATH=OFF
OFF: MICIUS_BENCHMARK_ITT 未指定，使用默认 OFF
ON:  MICIUS_BENCHMARK_ITT=ON; CMAKE_C_COMPILER=gcc
     MICIUS_ITT_ROOT=<LOCAL_WORKSPACE>/tooling/ittapi
```

ASan 配置继承 ON 的全部后端选项，仅覆盖：`CMAKE_BUILD_TYPE=Debug`，`LEGACY_TUNING/LTO/USE_LLD/STATIC_RUNTIME=OFF`；C/CXX flags 均为 `-fsanitize=address,undefined -fno-omit-frame-pointer`，EXE linker flags 为 `-fsanitize=address,undefined`。测试环境：`ASAN_OPTIONS=detect_leaks=0:halt_on_error=1`、`UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1`。首次编译 parallel 4，修正后的增量编译 parallel 2。

## Windows 操作

归档顶层附带 `r8-decode-itt-r1.patch`。在当前项目先 `git apply --check` 再应用补丁；不覆盖整棵工程。若检查失败，保留本地文件并报告冲突，不强行覆盖。补丁应用后在 MSYS2/Git Bash 执行：

```bash
cd /c/Projects/MICIUS
bash scripts/build-r8-itt-windows.sh
bash scripts/profile-r8-itt-windows.sh
```

第一条脚本创建 `build-r8-itt-decode`，保留 `build-r9-ab-r8-baseline`。配置逐项写在脚本中：原 Windows Clang Release/native/ALIGN32/LTO/LLD/static 配置与 R8 后端组合保持一致；只新增 C 编译器、ITT 开关和 SDK 路径。目录已存在会拒绝再次 clean build。CTests 应为 8/8。

第二条脚本使用 `models/smollm2-360m-instruct-q8_0.gguf`，先运行一次不传 `--itt-roi` 的新 executable，检查原先目标数据；通过后进行一次 `vtune -collect uarch-exploration -start-paused`。两次均为 prompt token 1、warmup 8、steps 128、CPU 4。期望 final token 30、token checksum `0x4f9cb9a464361ddf`、last logits checksum `0x796b3ffb22c87b08`。任一状态、身份、affinity 或 checksum 不符立即停止。这里的 checksum 检查不等于再次执行全轨迹逐字节 parity。

导出 summary、function hotspots、source-line hotspots、hardware events，保留完整采集目录和 executable/PDB。首先回传输出目录顶层 TXT/CSV/SHA256；无需重新采集已有 whole-process hotspots。若结果没有足够的 PMU 数据、pause 区间不符或模型加载成为主要硬件热点，应先处理区间证据，不能将其作为 decode-only 结果。[Intel uarch-exploration](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2026-0/run-general-exploration-analysis-command-line.html)、[hardware events report](https://www.intel.com/content/www/us/en/docs/vtune-profiler/user-guide/2024-0/hardware-events-report.html)

## 本轮要回答的性能问题

首要假设是 Q8 权重符号扩展、I32→F64 转换和 F64 逐元素乘加的指令/执行资源成本。已有目标机主循环反汇编显示寄存器内扩展，没有 F64 权重数组写回、YMM spill 或循环内 CALL。因此不能把 8 倍寄存器表示宽度直接计为 8 倍缓存/DRAM 流量；16-byte descriptor 流量是独立问题。

优先检查 decode 区间 GEMV 的 CPI、Retiring、Core Bound、Memory Bound 及其 L1/L2/L3/DRAM 子项，并用硬件事件与源码行/新 executable 反汇编定位转换、两次乘法和累加。高 Core Bound/低 Memory Bound 支持执行资源假设，但仍不能独立分离“类型转换”与“浮点算术”成本；高 Memory Bound 也不能自动证明存在 F64 缓存物化。FP 算术事件一般不等于转换指令事件；事件不可用时标记 NOT AVAILABLE，不编造比例或端口利用率。

不从旧 whole-process 的 85.59% CPU time 直接计算 decode 的 Amdahl 上界。不同层级的 Top-down 百分比不能直接相加；内联函数的显示条目需避免重复计数。未获得目标 PMU 数据前，不下 compute-bound 或 memory-bound 的正式结论。

本轮不修改 FMA contraction policy，不修改 GEMV 算法。将来 W8A8 activation quantization 或 FP32 accumulation 会改变数值契约，不能作为保持原 logits SHA 的位级等价替换。R8 F64 路径继续充当数值基准。

源码归档排除 build、模型、原始日志和 ITT SDK；包含本记录、脚本、增量补丁及源码摘要。交付后等待目标机采样结果。

# MICIUS CMake 行为重构验证

日期：2026-08-24

## 结论

当前 `CMakeLists.txt` 以用户提供的原文件为行为基线，并接入 v0.6-rebuild 的当前源码与三套测试。它没有退化为仅能构建当前 Linux 快照的简化清单；原有的架构调优、输出布局、链接策略和 Windows 调试信息路径仍被保留。

## 行为对应

| 原有行为 | 当前表达 |
|---|---|
| CMake 3.16 基线 | 保持 `cmake_minimum_required(VERSION 3.16)` |
| `ARCH` / `ALIGN` 缓存参数 | 名称与默认值保持，并增加输入校验及 `-march` 能力检查 |
| `compile_commands.json` | 保持开启 |
| `bin/` / `lib/` 输出 | 保持，并覆盖单配置与多配置生成器 |
| `Threads::Threads` | 保持为核心库公开依赖 |
| `GLOB_RECURSE CONFIGURE_DEPENDS` | 保持；只将 `MiciusCLIMain.cpp` 从静态库剥离，避免原文件的 Shell/Parser 重复编译 |
| `micius_core` / `micius_cli` | 保持；增加 `MICIUS::core` 别名 |
| O3、调试信息、frame pointer、native arch、展开、对齐、256-bit 偏好、关闭 stack protector | 由默认开启的 `MICIUS_ENABLE_LEGACY_TUNING` 保持 |
| ThinLTO | Clang/工具链支持时由 CMake IPO 启用；GCC 使用其受支持的 IPO 形式 |
| LLD | 查找 LLD 并执行编译器链接探测；不可用时明确使用默认链接器 |
| 静态 compiler runtime | 探测 `-static-libgcc/-static-libstdc++`；MSVC 使用一致的静态 CRT 属性 |
| Windows Clang CodeView/PDB | 保持；同时兼容 MinGW Clang 与 clang-cl 表达 |
| `micius_sandbox` | 路由和链接行为保持；现由 `tests/stress/AIEmuSandbox.cpp` 提供当前 V1 reference ABI 压力目标 |

## 必要调整

1. C++ 标准从 17 调整为 20。当前源码使用 `std::construct_at/std::destroy_at`，C++17 实测编译失败；继续声明 C++17 只会形成错误的构建契约。
2. `-ffast-math` 保留为 `MICIUS_ENABLE_FAST_MATH`，但默认关闭。实测开启后编译器消除了非有限值语义，导致无穷大 F32 权重未被 binder 拒绝，`runtime_contract_tests` 失败。
3. ThinLTO、LLD 与静态运行库不再无条件写死。原文件只能在特定 Clang/LLD 环境工作；当前版本在支持时采用原路径，不支持时给出配置诊断。

## 已执行验证

- C++17 编译探针：预期失败，定位到 `std::construct_at`。
- 原始 `-ffast-math` 探针：预期失败，定位到非有限 F32 权重契约。
- 默认 CMake 性能旗标的 GNU 等价构建：通过。
- GCC IPO 等价 `-flto=auto`、静态 `libgcc/libstdc++` 链接：通过。
- 按 CMake 源码分区生成 `libmicius_core.a`，仅以 `MiciusCLIMain.o` 链接 CLI：通过。
- `runtime_contract_tests`、`gguf_parser_tests`、`tensor_codec_tests`：全部通过。
- CLI `exit` smoke：通过。
- `ldd`：CLI 仅保留 glibc/libm 动态依赖，不依赖 `libstdc++` 或 `libgcc_s`。

## 未验证边界

当前执行环境没有 `cmake`，且临时依赖下载被环境中止，所以尚未在本环境真实执行 `cmake -S/-B` 和 `ctest`。CMake 文件已完成结构/分隔符审查及等价工具链构建验证，但不能把这些替代为本地真实 configure 证据。sandbox 已通过 Make 等价目标验证。

2026-08-25 的 Windows Clang64/Ninja 实机日志证明 CMake configure 与 generate 均成功，并正确发现 pthread、`-march=native`、LLD、静态运行库及全部目标。第一次 build 暴露的失败属于源码可移植性而非 CMake 生成失败：三套测试硬编码了旧 `src/` 头文件路径，内部枚举 `CONTEXT_FULL` 又与 Windows SDK 宏冲突。两处已修正，公共 ABI 未改变；Windows 完整 build/CTest 是否通过仍需修正后的实机复测，MSVC 仍未验证。

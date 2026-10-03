# Windows Clang64 构建修正

日期：2026-08-25

## 结论

首次 Windows Clang64/Ninja 构建的 CMake 配置和生成阶段成功。编译阶段存在两个确定的源码可移植性问题，现已修正：测试头文件路径依赖本地物理布局，以及内部枚举与 Windows SDK 宏重名。

## 修正内容

1. `tests/runtime_contract_tests.cpp`：`src/AIPrimitives.h` 改为 `AIPrimitives.h`。
2. `tests/gguf_parser_tests.cpp`：`src/MiciusGGUFParser.h` 改为 `MiciusGGUFParser.h`。
3. `tests/tensor_codec_tests.cpp`：`src/MiciusTensorCodec.h` 改为 `MiciusTensorCodec.h`。
4. 内部 `ReferenceStepResult::CONTEXT_FULL` 改为 `ReferenceStepResult::SEQUENCE_LIMIT_REACHED`，并同步引擎与运行时调用点。

裸头文件名由 `micius_core` 的 PUBLIC include roots 解析，因此同时适用于平铺 `src/` 树和 `include/AI`、`include/app` 布局。公共状态与错误码名称没有改动。

## 已完成验证

- Linux GCC 严格全量构建通过。
- `runtime_contract_tests`、`gguf_parser_tests`、`tensor_codec_tests` 全部通过。
- 在编译命令中人为定义 Windows 风格 `CONTEXT_FULL` 宏后，`AICoreEngine.cpp` 与 `micius_runtime.cpp` 语法检查通过。
- 全树不存在 `#include "src/..."`、内部裸 `CONTEXT_FULL` 枚举或旧调用点。

## 尚待实机确认

修正后的 Windows Clang64/Ninja 全量 build 和 CTest 尚未得到第二轮实机日志。原日志中的 signed-conversion 与 unused-field 信息是警告，不是本次停止构建的原因；应在构建恢复后单独清理，避免把警告整理与阻断性修复混成一次高风险改动。

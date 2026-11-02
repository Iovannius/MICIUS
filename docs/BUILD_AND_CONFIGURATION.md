# 构建与配置

Windows x64 发布配置使用 MSYS2 Clang64、CMake ≥ 3.21、Ninja 和 Python 3.8+。脚本以仓库为根目录，默认查找 `/c/msys64/clang64/bin`。自定义安装路径可设置 `MICIUS_CLANG64_BIN`。本次验证使用 Clang 22.1.8 和 C++20。

| 构建目标 | Bash 入口 | 产物目录 |
|---|---|---|
| v0.7 INT8 | `bash scripts/build-int8-release-windows.sh` | `build-int8-release/` |
| F64 数值参考 | `bash scripts/build-f64-reference-windows.sh` | `build-f64-reference/` |

脚本要求同名构建目录尚未存在，检查源码集合与散列后执行配置、构建和 CTest。日志与程序散列保存在 `logs/<preset>.*`。已有配置可执行增量构建。

```bash
cmake --preset int8-release-windows
cmake --build --preset int8-release-windows
ctest --preset int8-release-windows --output-on-failure
```

`bin/micius_cli.exe` 通过 `--model-benchmark` 或交互 `load` / `generate` 接入模型。`bin/micius_sandbox.exe` 使用合成张量验证 V1。CMake 静态库目标为 `MICIUS::core`。模型和构建产物在源码 ZIP 之外。

## 后端选择

`MICIUS_BACKEND` 选择构建期数值契约。INT8 与 F64 使用独立的构建目录。Clang/GNU 的新配置默认选 INT8，MSVC 默认选 F64。本版验证平台为 Windows x64 Clang64。`CUSTOM` 用于显式组合历史底层选项。

| 配置项 | INT8 | F64 |
|---|---|---|
| 激活与 GEMV | AVX2 W8A8，四行面板或旧布局回退 | F64 输入快照，Q8_32 AVX2 双行 |
| 权重与描述符 | V1 验证，成功绑定后按所有权退休旧缓冲 | V1 验证，F64 参考描述符 |
| Attention | I10/I11/I15/I17/I19 有序路径及原回退 | 有序 AVX2 Value、time-quad Score |
| 数值与诊断 | fast-math OFF，PW2 NONE，常规构建关闭 I2、I3、I4、I6、D1、ITT | 相同 |

INT8 配置选择 `I1_BACKEND=AVX2`、I10/I11/I13/I15/I17/I19、I22 描述符释放和 I25 面板。底层选项有 `MICIUS_` 前缀，阶段编号对应研究台账。Windows preset 使用 Release、Ninja、ThinLTO、LLD、静态运行库和 `ARCH=native`。INT8 编译及链接显式禁用工具链支持的 VNNI 扩展，生成的程序面向构建机指令集。

普通命令的 `gemv_backend` 根据当前模型绑定状态报告实际选择。四行路径为 `w8a8_avx2_panel4`，尾行回退为 `w8a8_avx2_row4`。

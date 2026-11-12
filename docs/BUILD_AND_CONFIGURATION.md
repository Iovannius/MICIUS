# 构建与配置

v0.7.2 可从解压后的源码根目录直接构建。两套配置分别安装到独立的前缀。原模块见 [M1 SDK](M1-MODULES-AND-ABI.md)，新增模块见 [M3 快照与采样](M3-PERSISTENCE-AND-SAMPLING.md)。M3 的固定实机验收已通过，详见 [发布记录](RELEASE_0_7_2.md)。

Windows x64 发布配置使用 MSYS2 Clang64、CMake ≥ 3.21、Ninja 和 Python 3.8+。默认启用的文本组件还需要 pkg-config 和 Clang64 的 PCRE2-8。运行时将 Clang64 的 `bin` 加入 `PATH`。脚本默认查找 `/c/msys64/clang64/bin`，自定义路径可设置 `MICIUS_CLANG64_BIN`。本次验证使用 Clang 22.1.8 和 C++20。

| 构建目标 | Bash 入口 | 产物目录 |
|---|---|---|
| INT8 完整开发构建 | `bash scripts/build-int8-release-windows.sh` | `build-int8-release/` |
| F64 数值参考 | `bash scripts/build-f64-reference-windows.sh` | `build-f64-reference/` |

脚本要求同名构建目录尚未存在，检查源码集合与散列后执行配置、构建和 CTest。日志与程序散列保存在 `logs/<preset>.*`。已有配置可执行增量构建。

```bash
cmake --preset int8-release-windows
cmake --build --preset int8-release-windows
ctest --preset int8-release-windows --output-on-failure
```

`bin/micius_cli.exe` 可通过 `--chat MODEL` 或交互 `load` / `chat` 开始文字对话，`--session ID` 或 `chat resume ID` 恢复。数字 token 入口为 `generate` 和 `--model-benchmark`。`bin/micius_sandbox.exe` 使用合成张量验证 V1。静态组件包括 `MICIUS::core`、`MICIUS::gguf`、`MICIUS::parser_llama`、`MICIUS::parser_qwen2`、`MICIUS::text`、`MICIUS::sampling` 和 `MICIUS::session`。可用 `cmake --install build-int8-release --prefix <prefix>` 安装 INT8 SDK。纯数值构建可配置 `-DMICIUS_ENABLE_TEXT=OFF`，不构建 text/session；sampling 可独立链接，不引入推理核心。

`scripts/run_m3.sh prepare` 会在新目录构建两套 SDK，运行 CTest、移动安装前缀并验证外部 C 链接。`all` 会在准备完成后运行固定实机检查。M3 快照含数值构建指纹，编译器、源码字节、编译选项或后端变更可能使旧快照无法恢复；不要将不同构建的 SDK 混在同一安装前缀。

## 简单 CLI 构建

```bash
bash scripts/build-cli.sh
./micius_cli/micius_cli.exe
```

新入口固定使用已验收的 `int8-release-windows` 配置，执行源码核对、全量构建和 CTest，然后将可执行文件与 PCRE2 DLL 发布到 `micius_cli/`。该目录名与 `micius_cli.exe` 的主名相同。中间文件及日志位于 `build-cli-int8/`，输出目录中另有许可证和 `BUILD-RECEIPT.json`。运行产物不计入源码发布清单。

新入口不覆盖已有 `micius_cli/` 或 `build-cli-int8/`；再次干净构建前由使用者保存或改名。失败时保留构建日志，不删除旧程序、模型或会话。仅在构建、测试和运行库检查全部成功后发布最终目录；`--help` 查看说明。路径含空格也可使用。

在源码根目录启动程序可继续使用根目录的 `models/` 和 `.micius-cache/`。从其他目录启动会改变相对路径；需要统一会话存储时设置 `MICIUS_CACHE_ROOT`。预编译 SDK 的两个 `bin/` 也附带 PCRE2 DLL；开发外部 text/session 客户端仍需安装 Clang64 的 pkg-config/PCRE2 开发文件。

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

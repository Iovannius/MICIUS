# MICIUS v0.6-rebuild 第一检查点验证记录

日期：2026-08-24  
平台：x86_64，g++ 13.3.0，GNU Make 4.3

## 结论

本检查点在合成模型和构造 GGUF fixtures 上形成了可重复的 correctness baseline；没有发现 ASan/UBSan 报告。该结论不延伸到用户真实 GGUF 的 logits/tokenizer 对齐，也不延伸到已禁用的 speculative 路径。

| 验证项 | 结果 | 证明内容 |
|---|---|---|
| 严格常规构建 | PASS | C++20、AVX2/FMA、`-Wall -Wextra -Wpedantic -Wconversion -Wshadow` 无警告 |
| clean 并行重建 `-j2` | PASS | 临时对象原子替换，所有目标成功链接 |
| Runtime contract | PASS | config/ABI、失败封闭、V1 tensor contract、确定性、reset、context-full |
| 两层 KV 归属 | PASS | 同位置 layer 0/1 的 K 按构造权重呈 1:2，V 独立为零；越界拒绝 |
| GGUF parser | PASS | 重复解析、未知版本、截断、alignment、scaling、rank/range、重复 key |
| Tensor codec | PASS | F16 正常/特殊值，Q8_0 解量化，64/128 列、多行、容量与负 scale 饱和 |
| ASan + UBSan | PASS | runtime/parser/codec 测试均无 sanitizer 报告（LeakSanitizer 单独禁用） |
| CLI smoke | PASS | CLI 正常启动并处理 `exit` |
| CMake 构建 | NOT RUN | 当前容器未安装 `cmake`；文件已生成但本轮没有执行证据 |
| 真实 GGUF 端到端 | NOT RUN | 未提供目标模型文件 |
| 参考 logits/KV parity | NOT RUN | 未提供可信实现的对照向量 |
| 文本 tokenizer/decoder | BLOCKED | 未提供目标 tokenizer metadata/向量；当前只接收与输出 token ID |
| Speculative execution | DISABLED | 创建与全部相关旧 ABI 均显式拒绝 |

## 执行命令

常规与并行回归：

```bash
make clean
make check -j2
```

Sanitizer 构建：

```bash
make BUILD_DIR=build-sanitize-final \
  CXXFLAGS='-std=c++20 -O1 -g -mavx2 -mfma -Wall -Wextra -Wpedantic -Wconversion -Wshadow -fsanitize=address,undefined -fno-omit-frame-pointer' \
  LDLIBS='-pthread -fsanitize=address,undefined' \
  build-sanitize-final/runtime_contract_tests \
  build-sanitize-final/gguf_parser_tests \
  build-sanitize-final/tensor_codec_tests -j1
```

Sanitizer 执行：

```bash
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
./build-sanitize-final/runtime_contract_tests

ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
./build-sanitize-final/gguf_parser_tests

ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
./build-sanitize-final/tensor_codec_tests
```

CLI smoke：

```bash
./build/micius_cli <<< 'exit'
```

## LeakSanitizer 说明

本容器对 ptrace/`/proc` 的限制会使 LeakSanitizer 自身初始化失败，因此最终 ASan 运行使用 `detect_leaks=0`。这只表示本环境没有独立的泄漏扫描证据；ASan 的越界/use-after-free 与 UBSan 检查仍启用并通过。生命周期风险另外由构造失败清理、析构顺序和重复 reset/bind 测试覆盖。

## 尚不能声称的内容

- 不能声称真实模型已经输出正常语言；
- 不能声称 GGUF 中 Q/K 排列、RoPE 与目标模型逐层完全一致；
- 不能声称 tokenizer/chat template 正确；
- 不能声称 speculative scheduler、verifier、epoch/GC 已修复；
- 不能用“程序能启动”代替真实 logits parity。

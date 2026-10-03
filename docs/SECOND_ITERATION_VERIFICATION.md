# MICIUS v0.6-rebuild 第二逻辑迭代验证报告

日期：2026-08-24  
公开 ABI：`0x00060003`  
平台：x86_64，g++ 13.3.0，GNU Make 4.3

## 最终结论

第二轮不是对第一检查点的表面加固，而是纠正了两个足以让真实模型 logits 失真的数学契约：Q/K 二次行置换与 Q8_0 相邻尺度有损合并。修复后，合成 reference 路径通过非对称、多头 GQA、位置 1 RoPE、两位置 attention、SwiGLU 和 LM-head 的独立标量 oracle；常规构建、ASan/UBSan 与 CLI 生命周期烟测均通过。

当前仍不能声称“真实 GGUF 已输出正常语言”。缺少目标模型文件、可信逐层/logits 对照以及 tokenizer/chat-template 向量，因此真实模型 parity 与自然语言 I/O 仍未验收；speculative execution 继续强制禁用。

## 根因、修复与证据

| 根因 | 旧行为 | 第二轮行为 | 回归证据 |
|---|---|---|---|
| Q/K 布局契约错误 | GGUF 已是 RoPE adjacent-pair 行序，Core 又执行 half-half → adjacent 置换 | Q/K/V 只做连续 packing；ABI 注释固定输入行序 | 每行不同 K 权重在 position 0 验证；position 1 独立计算每一对旋转值 |
| Q8_0 尺度丢失 | 两个 32-block 合成一个 64-block scale 并重新量化 | 每个 GGUF block 保留一个 scale 和原始 32 个 int8 | 不同/负/零 scale、`INT8_MIN`、32/96/128 列测试 |
| V1 entry 扩展伪支持 | 接受较大 `entry_size`，实际仍用 `entries[i]` 固定步长 | 按 `i * entry_size` 字节寻址，复制已知公共前缀 | 带 64 位扩展尾部的条目数组成功绑定 |
| 失败重绑破坏状态 | Core 在完成候选模型前释放旧 QKV | 候选模型完整构建后 `swap`，失败保留旧模型 | 有效绑定后提交畸形替换，旧模型仍可 prefill/decode |
| 首个生成 token 丢失 | final prompt decode 得到 T1，却先 decode(T1) 后只打印 T2 | 先 emit(T1)，再 decode(T1) 推进 | 代码时序复核、CLI 构建/生命周期 smoke；真实模型序列待集成验收 |
| 非终态死循环 | context-full/numeric 等落入 sleep/retry | 除 YIELD 外全部终止同步生成命令 | Runtime context-full 状态测试；Shell 分支穷尽复核 |
| UINT64 解析过度收窄 | 任意大于 uint32 的 UINT64 metadata 使整个 GGUF 失败 | 独立保存 uint64，只对配置字段做有界转换 | 8,000,000,000 参数计数 metadata 可解析，UINT64 alignment 可用 |
| 模型子集混淆 | LLaMA 标签下的 sliding-window/KV 维度差异未拒绝 | 非零 sliding window 与不匹配 key/value length 明确拒绝 | 构造 metadata fixtures |
| 成功 logits 被前置失败抹除 | 进入 step 立即清 `reference_has_logits` | precondition 通过、真正开始计算时才清 | context-full 后仍可复制最后一次成功 logits |
| Shell 暂存所有权窗口 | aligned allocation 可能在 vector 扩容异常前未登记 | 在任何分配前预留最坏容量 | ASan/UBSan 回归与异常边界代码复核 |

## 独立数学 oracle 覆盖

`TestAsymmetricGQAReferenceOracle` 不读取引擎内部中间结果来生成预期值，而是在测试侧独立计算：

1. 两个不同 prompt token 的 RMSNorm；
2. 2 query heads / 1 KV head 的非对称 Q/K/V 投影；
3. position 0 与 position 1 的 adjacent-pair RoPE；
4. 两个时间位置的 scaled dot-product 与稳定 softmax；
5. GQA 的 KV-head 共享映射；
6. 非零 O projection 与 residual；
7. 非零 gate/up/down 的 SwiGLU；
8. final RMSNorm 与三个独立 LM-head logits。

最终 logits 误差阈值为 `1e-4`。另一个两层测试直接读取每层 K/V，使用逐行不同权重验证 position 0 行序与 position 1 RoPE，并覆盖 reset、context-full 和确定性。

## 执行结果

| 验证项 | 结果 | 边界 |
|---|---|---|
| `make clean && make check -j2` | PASS | 严格 warning 集为空；CLI 与三套测试成功构建 |
| Runtime contract | PASS | ABI、扩展 stride、事务重绑、非对称 RoPE、GQA/SwiGLU oracle、KV、reset、context-full/logits |
| GGUF parser | PASS | 大 UINT64、UINT64 alignment、sliding-window/K length、重复/截断/range/alignment/scaling |
| Tensor codec | PASS | F16；Q8_0→Q8_32 逐块精确保留；容量/shape/非有限值 |
| ASan + UBSan | PASS | 三套测试无 sanitizer 报告；`detect_leaks=0` |
| 常规与 Sanitizer CLI smoke | PASS | 启动、REPL、`exit`、析构正常 |
| CMake | NOT RUN | 环境未安装 `cmake` |
| 真实 GGUF 逐层/logits parity | NOT RUN | 本轮没有目标模型与可信对照向量 |
| 文本 tokenizer/decoder | BLOCKED | 当前仅接收和显示 token ID |
| Speculative execution | DISABLED | 旧 Q8_64 SIMD/并发路径不属于公开 ABI，仍 fail closed |

## 复现命令

```bash
make clean
make check -j2
```

```bash
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
make BUILD_DIR=build-sanitize-iteration2 \
  CXXFLAGS='-std=c++20 -O1 -g -mavx2 -mfma -Wall -Wextra -Wpedantic -Wconversion -Wshadow -fsanitize=address,undefined -fno-omit-frame-pointer' \
  LDLIBS='-pthread -fsanitize=address,undefined' check -j1
```

```bash
printf 'exit\n' | ./build/micius_cli
printf 'exit\n' | ASAN_OPTIONS=detect_leaks=0 ./build-sanitize-iteration2/micius_cli
```

## 从本轮得到的正确经验

- 布局必须只有一个权威定义；不能让 parser、converter、binder 和 kernel 各自“猜一次”。
- 对称/全零测试会掩盖排列、head 映射和 scale 索引错误；oracle 必须刻意使用不对称数据。
- 能避免的有损格式转换就不应通过更复杂的饱和逻辑“修好”；保留源格式语义更可靠。
- “接受未来更大结构”必须真的按调用方 stride 寻址，不能只放宽 size 检查。
- 失败状态应区分“计算尚未开始”和“计算中途失败”，前者不应销毁最后一次成功结果。
- parser 对无关 metadata 应宽容保存，对进入执行语义的字段才严格收窄；模型架构差异则必须 fail closed。
- 没有 tokenizer 与真实 logits parity 时，打印 token ID 是诚实边界；任何伪文本解码都会重新制造“看似能跑、实际不正确”的问题。

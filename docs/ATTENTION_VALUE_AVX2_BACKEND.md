# Attention V 聚合 AVX2 实验后端

日期：2026-08-28

公开 ABI：`0x00060003`（未改变）

状态：实验、显式启用；scalar-F64 attention 仍为默认路径

## 1. 优化依据

固定 Raptor Lake-P CPU 4 的 Q8_32 AVX2 Hotspots 源码行报告中：

- QK 点积与缩放约占总 CPU Time 的 18.4%；
- V 加权聚合源码行至少占 25.6%；
- softmax `exp` 约占 4.0%；
- attention 可见合计不低于 48.5%。

V 聚合沿 dimension 的元素彼此独立，而每个 dimension 只依赖自己的历史 time 结果。因此可以只向量化 dimension，保留 time 顺序和每步 FP32 舍入；QK 则存在横向规约顺序问题，本阶段不修改。

## 2. 内核契约

`AccumulateAttentionValuesF32_AVX2` 接受未归一化 FP32 probability、F64 probability sum、带 element stride 的 FP32 V rows，以及 FP32 output。

- time 维仍严格串行；
- 每四个 dimension 使用 AVX2 packed-double FMA；
- 每个 time 后立即转换并回写 FP32；
- 每个向量结果仍检查 finite 与 FP32 范围；
- 1/2/3 个尾部元素使用标量 `std::fma`；
- 非有限 probability/value、非法参数和中间 FP32 溢出返回明确状态。

QK 的 `sqrt(head_dim)` 分母改为每 token 计算一次，但 dot 累加顺序和除法保持不变。

## 3. 构建与测试

基线与实验构建都启用已验证的 Q8_32 GEMV，只切换 attention-value 开关：

```bash
cmake -S . -B build-avx2-test -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMICIUS_REFERENCE_GEMV_AVX2=ON \
  -DMICIUS_REFERENCE_ATTENTION_VALUE_AVX2=OFF

cmake -S . -B build-attention-avx2-test -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMICIUS_REFERENCE_GEMV_AVX2=ON \
  -DMICIUS_REFERENCE_ATTENTION_VALUE_AVX2=ON

cmake --build build-avx2-test -j
cmake --build build-attention-avx2-test -j
ctest --test-dir build-avx2-test --output-on-failure
ctest --test-dir build-attention-avx2-test --output-on-failure
```

实验 sandbox 应输出：

```text
gemv_backend=q8_32_f32_avx2_experimental
attention_value_backend=f32_f64_avx2_order_preserving_experimental
```

## 4. 已完成预筛

- 96 组 sequence length/head dimension bitwise 差分：PASS；
- 非对齐、stride padding、非 4 倍数尾部：PASS；
- 非有限 probability sum/probability/value 与 FP32 中间溢出拒绝：PASS；
- 五个严格测试目标：PASS；
- 完整 runtime ASan+UBSan：PASS；
- scalar、仅 GEMV AVX2、GEMV+V AVX2 的完整 sandbox checksum：相同；
- 汇编确认 packed `vfmadd132pd`、`vcvtps2pd`、`vcvtpd2ps` 和向量范围 mask 已生成。

虚拟 EPYC 上仅 GEMV AVX2 与 GEMV+V AVX2 五组配对的中位吞吐分别为 2509.9 与 3282.9 steps/s，即 1.31×。运行中频率漂移明显，该数字不是目标机结论。

## 5. 尚未证明

- Windows Clang64 对新文件和第五个 CTest 的完整构建；
- 固定 Raptor Lake-P P-core 的实际 A/B 加速比与新热点分布；
- 真实 GGUF 逐层、logits 与 top-k parity；
- 其他编译器是否与当前 Clang/GCC FMA contraction 产生完全一致的 bit pattern。

这些门槛通过前，`MICIUS_REFERENCE_ATTENTION_VALUE_AVX2` 必须保持默认关闭。

## 6. Rejected numeric-validation batching experiment (r7 screening)

Two variants that delayed the AVX2 numeric-error branch were screened and
removed. A whole-sequence sticky mask lost 3.29% median operator throughput;
a shorter per-time-row mask still lost 1.86% over eleven longer alternating
runs and was slower in nine of them. Both retained bitwise valid-data output,
but their loop-carried mask dependency and register lifetime cost more than the
perfectly predictable checked branch. No sticky-validation build option remains
in the deliverable.

The additional regression case is retained: it proves that an F64 value just
above `FLT_MAX` can convert back to finite `FLT_MAX`, so replacing the existing
pre-conversion range predicate with a final finite-only check would be wrong.

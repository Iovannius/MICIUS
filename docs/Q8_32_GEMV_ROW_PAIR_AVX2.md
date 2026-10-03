# Q8_32 双行共享输入 AVX2 候选

日期：2026-08-30

## 结论

本阶段新增的是一个默认关闭的实验后端，不替换 r4 基线：

- 接受旧算子的多输出行并行思想；
- 保持当前 Q8_32 × F32、完整 `int8_t` 定义域和 F64 累加契约；
- 每两行共享 activation 加载及 F32→F64 转换；
- 每四个权重直接从各行加载并符号扩展，不再让四个 16-byte
  packed-weight 值跨越整个双行块存活；
- 每行仍使用原来的四条累加链、规约树和标量尾部顺序；
- checked、prevalidated 与奇数输出行尾部都有独立路径；
- 不采用旧版动态 activation 重量化、`maddubs` 整数点积、主动预取或强制对齐。

旧版 `abs(x) + sign(w, x) + maddubs` 并不是可依赖的 16 位自然溢出技巧。`maddubs` 的第二输入是有符号字节；当权重为 `-128` 且 activation 为负时，符号变换无法表示所需的 `+128`，并且该指令的 16 位阶段是饱和语义。它不满足当前完整 Q8_32 定义域，因此只作为历史代码保留。

## 为什么是两行而不是四行

当前每行有四个 `__m256d` 累加器。四行 tile 会让 16 个 YMM 寄存器全部成为长期累加状态，scale、权重扩展和 activation 转换只能挤占或溢出到栈。两行 tile 使用 8 个长期累加器，但这不自动保证无 spill：如果同时保持两个 scale 和四个 16-byte packed-weight 值，16-register AVX2 目标仍可能把一个累加器放到栈上。

证据修正：r5 最初的 GCC 13 汇编由 `-march=native` 在支持 AVX-512 的主机上产生，并使用了 `ymm18`–`ymm24`。因此其中“无 YMM spill”的结论不适用于 Raptor Lake 的 16-register AVX2 环境。重新使用 `-march=haswell -mno-avx512f` 后，r5 双行热循环确实出现每块一次累加器栈写和一次栈读。

r6 改为四字节直接权重加载。严格 AVX2 汇编验证结果：

- 两行每个 32 列块：8 次 `vcvtps2pd`；
- r4 逐行实现处理两行：16 次 `vcvtps2pd`；
- 权重扩展生成 `vpmovsxbd xmm, m32`，不再需要跨调用保存 shifted 16-byte block；
- 双行主热循环：0 次 YMM 栈访问；
- prevalidated 双行函数体从 2085 bytes 缩至 1968 bytes。

## 开关和运行时报告

新开关：

```text
MICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2=ON
```

它要求：

```text
MICIUS_REFERENCE_GEMV_AVX2=ON
```

默认值为 `OFF`。启用后 sandbox 应输出：

```text
gemv_backend=q8_32_f32_avx2_row_pair_direct_load_experimental
```

scale 可信边界仍由既有开关和 engine 私有状态控制，没有扩大调用者权限。

## 已完成验证

- 112 组 GEMV shape/value 差分；
- 原 checked、原 prevalidated、新 checked、新 prevalidated 四路径位级一致；
- 覆盖输入维度 1、31、32、33、63、64、65、129、4096；
- 覆盖输出维度 1、2、3、4、5、7、16；
- 覆盖非对齐指针、`int8_t=-128/127`、负 activation、零/负 scale 和奇数行尾部；
- 覆盖空指针、非有限 activation、非法第二行 descriptor 和 FP32 输出溢出；
- 五个既有契约测试在 r5 Windows 构建中均通过；
- r6 严格 AVX2 的 112 组差分与 ASan+UBSan 均通过；
- r6 完整 2,000-step sandbox PASS，7 次抽检且 checksum 与既有本地基线相同。

r5 在目标 Windows/Clang64 的五组 40,000-step 交错结果中，中位吞吐从 7830.33 提升至 7990.93，约为 2.05%。匹配低频状态的 VTune v2 显示退休指令减少 3.21%、CPI 增加 2.69%、推算周期减少 0.61%；说明转换共享有效，但原调度的寄存器压力吃掉了大部分收益。

r6 严格 16-register AVX2 的 128×1024 prevalidated 九组交错预筛中，基线/直接加载 row-pair 中位比为 1.291；相同严格目标下 r5 row-pair 的对应中位比为 1.126。该结果只证明新调度值得进入目标机 A/B，不能替代 Windows 数据。默认值继续保持 `OFF`。

## Windows Clang64 验证

先构建 r4 算子基线：

```bash
cmake -E remove_directory build-r6-baseline

cmake -S . -B build-r6-baseline -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DMICIUS_REFERENCE_GEMV_AVX2=ON \
  -DMICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2=OFF \
  -DMICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES=ON \
  -DMICIUS_REFERENCE_ATTENTION_VALUE_AVX2=ON \
  -DMICIUS_ENABLE_FAST_MATH=OFF

cmake --build build-r6-baseline --parallel &&
ctest --test-dir build-r6-baseline --output-on-failure
```

再构建 row-pair：

```bash
cmake -E remove_directory build-r6-directload

cmake -S . -B build-r6-directload -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DMICIUS_REFERENCE_GEMV_AVX2=ON \
  -DMICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2=ON \
  -DMICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES=ON \
  -DMICIUS_REFERENCE_ATTENTION_VALUE_AVX2=ON \
  -DMICIUS_ENABLE_FAST_MATH=OFF

cmake --build build-r6-directload --parallel &&
ctest --test-dir build-r6-directload --output-on-failure
```

快速一致性检查：

```bash
for dir in build-r6-baseline build-r6-directload; do
  echo "build=$dir"
  ./"$dir"/bin/micius_sandbox.exe \
    --mode throughput --steps 2000 --warmup 128 \
    --verify-every 256 --threads 1 --cpu 4 |
    grep -E '^(gemv_backend|gemv_scale_validation|attention_value_backend|status|wall_seconds|steps_per_second|verifications|worker_start_cpu_0|checksum)='
done
```

性能验收仍使用五组、40,000 steps、奇偶轮换顺序。只有同时满足以下条件才考虑把开关晋级：

1. 两边全部 `status=PASS`；
2. `checksum` 全部相同；
3. `worker_start_cpu_0=4`；
4. row-pair 的五组中位吞吐至少提升 2%；
5. VTune 不出现新增 YMM spill、前端或 L1D 回退。

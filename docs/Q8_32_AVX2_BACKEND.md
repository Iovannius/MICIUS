# Q8_32 × F32 AVX2 实验后端

日期：2026-08-27

公开 ABI：`0x00060003`（未改变）

状态：实验、显式启用；scalar-double reference 仍为默认 oracle

## 1. 优化依据

Raptor Lake-P 的原始 VTune 采样显示：

- `ReferenceGEMV` 占 64.2% CPU Time；
- `isfinite` 两项合计 10.7%；
- P-core `FP Scalar=18.4%`、`FP Vector=0%`；
- P-core Port 0/1 利用率为 98.2%/86.6%；
- Memory Bound 约 0.3%–1.1%，`DRAM Bandwidth Bound=0%`。

因此当前 1.16 MB 合成模型受标量指令数量和执行端口限制，不受 DRAM 带宽限制。旧 `DraftGEMV_INT8_AVX2` 采用 activation 动态重数量化与 Q8_64 scale，不能满足公开 Q8_32 reference 契约，故没有复用。

## 2. 新内核契约

`GEMV_Q8_32_F32_AVX2` 接受：

- row-major `int8_t` 权重；
- 每行每 32 个权重一个 `QuantDescriptor`；
- FP32 input/output；
- symmetric Q8，即 `zero_point == 0`。

行为边界：

- activation 保持 FP32，不生成临时 int8 activation；
- packed-double 累加降低相对 scalar-double 的数值漂移；
- 输入地址和权重行不要求 32 字节对齐；
- K 不为 32 倍数时使用安全标量尾部；
- 支持 `INT8_MIN`、负 scale 和零 scale；
- 非有限 input、非有限 scale、非零 zero-point 和 FP32 输出溢出均返回明确状态。

默认 `ReferenceGEMV` 未删除或改写为优化实现。配置开关关闭时，执行路径仍是原有逐元素 scalar-double oracle。

## 3. 构建

建议使用两个独立目录，禁止把 scalar/AVX2 对象混在同一增量构建中：

```bash
cmake -S . -B build-scalar -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMICIUS_REFERENCE_GEMV_AVX2=OFF
cmake --build build-scalar --target micius_sandbox q8_32_gemv_tests

cmake -S . -B build-avx2 -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMICIUS_REFERENCE_GEMV_AVX2=ON
cmake --build build-avx2 --target micius_sandbox q8_32_gemv_tests
```

运行完整测试：

```bash
ctest --test-dir build-scalar --output-on-failure
ctest --test-dir build-avx2 --output-on-failure
```

AVX2 配置会发出 CMake warning，sandbox 输出应分别包含：

```text
gemv_backend=scalar_double_reference
gemv_backend=q8_32_f32_avx2_experimental
```

## 4. 已完成验证

严格编译参数：C++20、`-O3 -mavx2 -mfma`、`-Wall -Wextra -Wpedantic -Wconversion -Wshadow`。

| 验证项 | 结果 |
|---|---|
| Q8_32 内核差分 | PASS；64 组 shape，K=1…4096 |
| 最大差分误差 | 当前用例最终 FP32 输出为 0 absolute/relative error |
| 非对齐与尾部 | PASS |
| `INT8_MIN`、负/零 scale | PASS |
| 非有限值、zero-point、溢出拒绝 | PASS |
| 默认 scalar runtime contracts | PASS |
| AVX2 完整 runtime contracts | PASS |
| GGUF parser / tensor codec | PASS |
| ASan + UBSan（`detect_leaks=0`） | PASS |
| 20,000-step scalar/AVX2 checksum | 相同 |

Release CTest 额外强制取消 `NDEBUG`。三个历史测试使用 `assert()` 作为检查机制；如果断言仍被关闭，测试源码会在编译期报错，禁止产生“零检查但测试通过”的假阳性。

本地预筛平台为虚拟 AMD EPYC 9V74，不是目标 Raptor Lake-P。单 CPU 固定、每组 10,000 steps、scalar/AVX2 交错各 5 次的结果：

| Backend | Min | Median | Max | Mean |
|---|---:|---:|---:|---:|
| scalar-double | 1656.0 | 1704.6 | 1748.8 | 1697.7 steps/s |
| Q8_32 AVX2 | 3942.0 | 4439.1 | 4517.0 | 4330.3 steps/s |

median speedup 为 2.60×，mean speedup 为 2.55×；所有 10 次 checksum 均为 `0x631040919f4abd54`。这些数字只证明实现值得进入目标机 A/B，不能替代 Raptor Lake 实测。

## 5. Raptor Lake A/B 结果

Windows 11、Clang64 22.1.8、固定 P-core 逻辑 CPU 4、20,000 steps：

| 采集方式 | scalar | Q8_32 AVX2 | 加速比 |
|---|---:|---:|---:|
| 裸跑五组中位数 | 1161.7 | 2728.9 | 2.35× |
| HW Hotspots | 1003.5 | 2199.6 | 2.19× |
| Uarch Exploration | 1084.7 | 2733.6 | 2.52× |

所有运行均为 `PASS`、CPU 4、checksum `0xf5c202a8d9fb36ac`。Uarch 对比显示 instructions 由 170.33B 降至 59.62B（-65.0%），clockticks 由 40.24B 降至 16.69B（-58.5%）；P-core FP Vector 由 0% 升至 10.0%，Memory Bound 仍低于 1%。

以下命令保留为复测模板：

先选定一个已知 P-core 逻辑 CPU，以下以 `P_CPU_ID` 表示。两套命令必须使用同一个 ID、同一电源模式和相同参数：

```bash
./build-scalar/bin/micius_sandbox \
  --mode throughput --steps 20000 --warmup 256 \
  --verify-every 0 --threads 1 --cpu P_CPU_ID

./build-avx2/bin/micius_sandbox \
  --mode throughput --steps 20000 --warmup 256 \
  --verify-every 0 --threads 1 --cpu P_CPU_ID
```

确认两边均为 `status=PASS`、checksum 相同，并且 `worker_start_cpu_0=P_CPU_ID`。随后分别采集：

```bash
vtune -collect hotspots -knob sampling-mode=hw -knob sampling-interval=1 \
  -result-dir ./vtune/hw-scalar-p-01 -- \
  ./build-scalar/bin/micius_sandbox --mode throughput --steps 20000 \
  --warmup 256 --verify-every 0 --threads 1 --cpu P_CPU_ID

vtune -collect hotspots -knob sampling-mode=hw -knob sampling-interval=1 \
  -result-dir ./vtune/hw-avx2-p-01 -- \
  ./build-avx2/bin/micius_sandbox --mode throughput --steps 20000 \
  --warmup 256 --verify-every 0 --threads 1 --cpu P_CPU_ID

vtune -collect uarch-exploration -result-dir ./vtune/uarch-avx2-p-01 -- \
  ./build-avx2/bin/micius_sandbox --mode throughput --steps 20000 \
  --warmup 256 --verify-every 0 --threads 1 --cpu P_CPU_ID
```

验收重点不是只看 steps/s，还要检查：

- `GEMV_Q8_32_F32_AVX2` 是否成为可解析热点；
- retired instructions 是否明显下降；
- `FP Vector` 是否从 0 上升；
- Port 0/1 压力是否下降或转移；
- 新瓶颈是否转移到 attention、RMSNorm 或 softmax。

当前 `VTUNE_ROI_BEGIN/END` 仍是文本标记，不控制 VTune collection。构造和 warmup 仍可能进入采样；因此 A/B 必须保持完全相同的参数与构建配置，不能把不同 collector 下的绝对吞吐互相比较。

## 6. 尚未证明

- 真实 GGUF 逐层/logits parity；
- 更激进的 FP32 accumulator 或多行并行实现是否满足误差门槛；
- 大模型工作集下的 DRAM/LLC 行为。

在真实 logits oracle 通过前，该后端不应改为默认。

后续的 immutable scale 重检消除候选及其可信边界见
`docs/Q8_32_PREVALIDATED_SCALES.md`。该候选没有改变本文件所述公开 checked
内核，也没有把实验选项改为默认开启。

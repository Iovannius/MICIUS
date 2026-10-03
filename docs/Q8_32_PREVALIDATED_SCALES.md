# Q8_32 scale 预验证可信边界候选

日期：2026-08-28

公开 ABI：`0x00060003`（未改变）

状态：实验、默认关闭；只允许通过 V1 typed tensor binder 的 engine 实例进入快路径

## 1. 目标与上一次方案为何作废

VTune 已证明 Q8_32 AVX2 后端不再是唯一主热点，但 reference token 中仍会在每次 GEMV、每个输出行、每个 block 重复检查不可变 scale 的 `isfinite` 与 `zero_point`。这些 descriptor 在模型绑定后不会变化，因此可以把验证前移到绑定事务。

不能采用“只要编译开关打开，所有 `ReferenceGEMV` 都无条件相信 scale”的方案。旧 Core 曾公开接受无 type/shape/size 的裸 tensor 指针，调用者可绕过 V1 校验并让非法 `zero_point` 进入 unchecked 内核。该方案即使官方 C ABI 路径正确，Core 自身的状态不变量仍不成立，故作废。

## 2. 当前不可绕过的状态不变量

只有以下完整链路能令某个 engine 实例的 `reference_scales_prevalidated` 为真：

1. `Micius_BindModelTensorsV1` 校验 ABI、type、rank、shape、byte size、stride 与数据对齐；
2. 每个 F32 tensor 全量检查 finite；每个 Q8_32 descriptor 检查 finite scale、`zero_point == 0` 与零 padding；
3. 每个 scale tensor 被复制到 engine 候选所有权缓冲，不再借用调用者的 descriptor 内存；
4. 私有 `AIEngineFront::BindValidatedTensors` 按指针与元素数精确消费 `7 * num_layers + 1` 个 owned scale buffer；缺失、重复、尺寸错误或多余 ownership 均拒绝；
5. Q/K/V scale 在同一候选事务内复制到 packed QKV；其余 scale 指针只允许指向上述 owned buffer；
6. 新模型完成构建并成功 reset 后才提交；失败保留旧模型、旧 scale ownership 与旧预验证状态；
7. `BindTensors` 与 `LoadWeights` 不再是公开 Core mutator，调用者不能自行声明已验证状态；
8. 未携带实例级预验证状态的任何 `ReferenceGEMV` 调用继续进入 checked AVX2 内核。

候选 packed QKV 使用作用域守卫统一释放。普通拒绝、内存分配异常、reset 回滚和成功重绑定都不依赖分散的手工清理分支。

## 3. 快路径仍保留的检查

prevalidated 内核只省略以下两项不可变模型数据重检：

- `QuantDescriptor::scale` 是否 finite；
- `QuantDescriptor::zero_point` 是否为零。

以下运行时检查没有删除：

- weights/scales/input/output 空指针；
- 零维度与 size 乘法溢出；
- 每次 GEMV 的 activation finite；
- F64 累积 finite 与 FP32 输出范围。

公开 `GEMV_Q8_32_F32_AVX2` 始终是 checked 入口。

## 4. 构建开关

新增开关：

```text
MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES=OFF
```

默认关闭，且要求 `MICIUS_REFERENCE_GEMV_AVX2=ON`。它与
`MICIUS_ENABLE_FAST_MATH=ON` 不兼容：CMake 配置会直接失败，GNU/Clang
绕过 CMake 的直接构建也由编译期检查拒绝。attention 开关保持独立。

这是可信边界的必要条件，而不是性能偏好。`-ffast-math` 允许编译器假定
浮点输入有限，可能消除 V1 binder 中的 `std::isfinite`，从而令非法 scale
获得预验证状态。

实验构建的 sandbox 应显示：

```text
gemv_backend=q8_32_f32_avx2_experimental
gemv_scale_validation=private_v1_prevalidated_checked_fallback
attention_value_backend=f32_f64_avx2_order_preserving_experimental
```

checked 对照构建应显示：

```text
gemv_scale_validation=per_call_checked
```

## 5. 当前验证结果

| 验证项 | 结果 |
|---|---|
| 默认 scalar、checked AVX2、prevalidated AVX2 | 三配置各 5/5 测试通过 |
| checked 与 prevalidated GEMV | 64 组 shape 位级相同 |
| prevalidated 运行时参数/activation/输出范围检查 | PASS |
| 非有限 scale、非零 zero-point、非零 padding | V1 bind 拒绝 |
| F32/scale 错位指针 | V1 bind 拒绝且不解引用 |
| bind 后修改调用者原始 scale | 活跃模型不受影响 |
| 无效重绑定 | 旧模型仍可用 |
| 旧裸 `AIEngineFront::BindTensors` 探针 | 编译失败 |
| ASan + UBSan | 5/5 与压力用例通过（`detect_leaks=0`） |
| LeakSanitizer | 当前执行环境受 ptrace 阻断，未宣称通过 |
| 三配置 2,000-step 集成 checksum | 均为 `0x182dd19b1fb45fb6` |

r3 加固后重新从源码独立构建：default、checked、prevalidated 三配置仍各
5/5；三套 2,000-step 短压力运行均完成 7 次 verification，checksum 仍为
`0x182dd19b1fb45fb6`。prevalidated 的 ASan+UBSan 为 5/5，contracts 压力
通过。额外使用 `-ffast-math` 绕过 CMake 直接编译时，私有内核头文件按
设计产生编译错误，证明该组合不能生成可执行文件。

虚拟 EPYC 上 checked/prevalidated 交错各 5 次、每次 10,000 steps 的中位吞吐为 3411.9/3512.3 steps/s，预筛提升 2.94%，10 次 checksum 均为 `0x631040919f4abd54`。这只说明值得进入目标机 A/B，不是 Raptor Lake 结论。

## 6. Windows/Raptor Lake 验收

必须使用两个全新构建目录：

```bash
cmake -S . -B build-scale-checked -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DMICIUS_REFERENCE_GEMV_AVX2=ON \
  -DMICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES=OFF \
  -DMICIUS_REFERENCE_ATTENTION_VALUE_AVX2=ON

cmake -S . -B build-scale-prevalidated -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DMICIUS_REFERENCE_GEMV_AVX2=ON \
  -DMICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES=ON \
  -DMICIUS_REFERENCE_ATTENTION_VALUE_AVX2=ON

cmake --build build-scale-checked -j
cmake --build build-scale-prevalidated -j
ctest --test-dir build-scale-checked --output-on-failure
ctest --test-dir build-scale-prevalidated --output-on-failure
```

先做带验证的短运行：

```bash
for dir in build-scale-checked build-scale-prevalidated; do
  echo "build=$dir"
  ./"$dir"/bin/micius_sandbox.exe \
    --mode throughput --steps 2000 --warmup 128 \
    --verify-every 256 --threads 1 --cpu 4 |
    grep -E '^(gemv_backend|gemv_scale_validation|attention_value_backend|status|steps_per_second|verifications|worker_start_cpu_0|checksum)='
done
```

短运行的 `status`、7 次 verification、CPU 4 与 checksum 必须一致。随后沿用前一阶段的奇偶交错五组 20,000-step A/B，只把目录替换为上述两套构建。

进入 VTune 的最低条件：

- 两套 CTest 均为 5/5；
- 所有 A/B checksum 相同；
- prevalidated 五组中位数没有回退，且建议至少提升 2%；
- Hotspots 中 `isfinite`/scale validation 时间下降，并且没有新异常热点。

若吞吐信号低于噪声或 Windows 编译/契约测试失败，本开关继续保持 OFF，不在该候选之上规划下一项优化。

## 7. 尚未证明

- Windows Clang64 的两套全新构建与 LeakSanitizer 等价检查；
- 固定 Raptor Lake-P CPU 4 的五组中位收益与 VTune 行级归因；
- 真实 GGUF 的逐层/logits/top-k parity；
- 其他调用约定或外部 C++ Core 用户的源码兼容性。

# MICIUS v0.6-rebuild 第一检查点变更清单

日期：2026-08-24

## 工程与 ABI

- 新增 Make/CMake 构建入口和三个测试程序。
- 新增 `Micius_CreateEngineV1`、`Micius_InitConfig` 与 ABI 版本校验。
- 新增结构化错误码、线程局部错误字符串与 C ABI 异常边界。
- 新增 `MiciusTensorEntryV1`，携带 type/rank/shape/bytes/stride/flags。
- 旧 tensor/raw-weight ABI 与 speculative ABI 保留符号但 fail closed。

## Reference 引擎

- 新增权威单线程 prefill/decode 路径。
- 新增标量 Q8 GEMV、RMSNorm、RoPE、GQA attention、SwiGLU 与 LM head。
- KV 改为逐层逐位置存储，scratch/residual 与 KV 分离。
- 新增 reset、sequence length、logits copy 和只读 KV slot API。
- context-full、非法 token、未绑定模型和数值错误均返回明确状态。

## GGUF 与量化

- Parser 限定 GGUF v2/v3 和 Llama 架构，校验 metadata/tensor 全部边界。
- 重复 metadata/tensor name、过深嵌套、未知 type、越界/重叠 range 均拒绝。
- 配置提取改为事务式，失败不修改调用方配置。
- 新增独立 F16/Q8_0 codec；Q8_0 按矩阵行转换，不跨行合并 block。
- 修复负 scale 产生 `+128` 后直接转 `int8_t` 的实现相关行为，改为显式饱和。
- V1 binder 拒绝非有限 F32、非有限 scale、非零 zero point 和非零 padding。

## Shell 与生命周期

- 移除 hash/空白伪 tokenizer、词表扫描伪 decoder 和伪 fallback token。
- 只接受/显示 token ID，避免把未完成路径伪装成语言输出。
- Engine 在 mmap/权重缓冲之前销毁；内部 scale 副本的析构顺序也已固定。
- reference 模式不再分配不可达 speculative dummy 权重。
- REPL 命令边界捕获异常并清理半建模型状态。

## 第二逻辑迭代（ABI 0x00060003）

- 删除 GGUF Q/K 进入 Core 后的第二次 RoPE 行置换；V1 契约明确 Q/K 已采用 GGUF adjacent-pair 行序。
- 将公开和 reference 内部量化块从有损 Q8_64 改为 Q8_32；GGUF Q8_0 的每个 FP16 scale 与 32 个 int8 值逐块原样保留。
- V1 `entry_size` 现在按字节步长读取并复制公共前缀，真正支持未来扩展条目；乘法溢出显式拒绝。
- Core tensor bind 改为候选模型完整构建后再交换；验证/分配失败不销毁最后一次有效模型。
- `CONTEXT_FULL` 等未开始计算的失败不再抹除最后一次成功 logits。
- GGUF UINT64 元数据不再统一压缩至 uint32；只有实际映射到运行配置的字段才做范围检查。
- 明确拒绝非零 sliding-window attention 及不匹配的 key/value length，避免把不同模型子集当作普通 LLaMA 执行。
- Shell 先输出最后一次 prompt decode 已产生的 T1，再以 T1 推进 T2；所有非 YIELD 状态均终止同步生成循环。
- Shell 预留全部桥接/所有权容器容量，失败加载立即清空旧模型标签；Windows mmap 构造失败补齐 handle 清理。
- 新增非对称 Q/K、位置 1 RoPE、2Q:1KV GQA、两位置 softmax、O 投影、SwiGLU 和 logits 的独立标量 oracle。

## CMake 行为重构

- 以原始 CMake 为基线恢复 `ARCH`、`ALIGN`、递归源码发现、`bin/lib` 输出、静态核心库、CLI、sandbox、LLD、静态运行库以及 Windows Clang CodeView/PDB 行为。
- 当前源码使用 `std::construct_at/std::destroy_at`，因此 CMake 明确要求 C++20；实测 C++17 无法编译，不再提供虚假的兼容声明。
- ThinLTO/IPO、LLD、静态 GNU 运行库改为配置期能力探测；可用时保持原路径，不可用时给出明确诊断并使用工具链默认路径。
- `-ffast-math` 仍由 `MICIUS_ENABLE_FAST_MATH` 保留，但默认关闭：实测开启后非有限 F32 权重检查失效并导致契约测试失败。
- 原始 v0.6 快照没有可用的 `AIEmuSandbox.cpp`；现已重构当前 V1 reference ABI 沙箱，同时保留 `MICIUS_SANDBOX_SOURCE` 覆盖入口。

## Reference 压力沙箱重构

- 删除对 fail-closed 的 v0.5 raw-weight、Verifier、DraftTree、Oracle audit 与 Q8_64 接口依赖；沙箱仅使用公开 ABI `0x00060003`。
- 使用确定性、非零、完整覆盖 Q/K/V/O 与 SwiGLU 的合成 Q8_32 模型；相同参数输出稳定 checksum。
- 增加 `throughput`、`churn`、`contracts` 三类独立负载，避免把 decode 热点、全状态 reset 和事务绑定分配混在一份 VTune 数据中。
- `--threads` 为共享只读模型的独立 Context 并行负载；不对单个非线程安全 Context 做未定义的并发调用。
- 定时循环无模型/context 构造，支持 warmup、reset 周期、logits/KV 抽检和文本 ROI 边界；错误返回非零退出码。
- Make 的 `all/check` 现在编译 `micius_sandbox`，`check` 不自动运行长压力负载；CMake 保持原目标名与 `bin/` 输出。

## Windows Clang64 构建修正（2026-08-25）

- 测试源码不再使用 `src/...` 形式硬编码头文件物理位置；改由目标公开的 include roots 解析裸头文件名，兼容平铺源码树与 `include/AI`、`include/app` 历史布局。
- 内部 `ReferenceStepResult::CONTEXT_FULL` 改名为 `SEQUENCE_LIMIT_REACHED`，避免被 Windows SDK 的 `CONTEXT_FULL` 宏在预处理阶段替换。
- 公共 C ABI 的 `MICIUS_STATE_CONTEXT_FULL` 与 `MICIUS_ERROR_CONTEXT_FULL` 保持不变，没有 ABI 变更。
- Linux 严格全量构建和三套契约测试重新通过；额外以人为定义的 `CONTEXT_FULL` 宏完成语法探针。Windows Clang64 完整重建仍以修正后的实机复测为准。

## Q8_32 AVX2 实验后端（2026-08-27）

- 新增 `GEMV_Q8_32_F32_AVX2`：保持 FP32 activation，不进行旧 Q8_64 路径的动态重数量化；每 32 个 int8 权重原样使用一个 `QuantDescriptor`。
- 内核以 AVX2 packed-double 执行符号扩展、F32→F64 转换和累加，提供非对齐加载与任意 K 尾部；`INT8_MIN`、负/零 scale 均受支持。
- activation 有限性从“每个输出行重复扫描”提升为每次 GEMV 只扫描一次；scale/zero-point 和输出范围仍 fail closed。
- 默认 scalar-double reference 完全保留。只有配置 `MICIUS_REFERENCE_GEMV_AVX2=ON` 时，完整 reference token 路径才转接实验内核；sandbox 会打印实际 backend。
- 新增 `q8_32_gemv_tests`，覆盖 64 组差分 shape、非对齐地址、非 32 倍数尾部、异常输入/scale/zero-point 和 FP32 溢出。
- sandbox 新增单线程 `--cpu N` 固核选项，并打印 `worker_start_cpu_0`，用于避免混合 P/E 核采样。
- Linux GCC 严格构建、default/AVX2 runtime contracts、GGUF/codec tests 及 ASan+UBSan 均通过。
- Windows Clang64 Release 的 scalar/AVX2 构建与有效 CTest 均通过；固定 Raptor Lake-P CPU 4 的五组裸跑中位吞吐由 1161.7 提升至 2728.9 steps/s（2.35×），checksum 相同。
- 固核 VTune A/B 显示 retired instructions 下降 65.0%、clockticks 下降 58.5%、FP Vector 从 0% 升至 10.0%，Memory Bound 仍低于 1%；热点已从 GEMV 转移至 attention。
- 修复 Release CTest 假阳性：三个历史测试依赖 `assert()`，现在测试目标显式取消 `NDEBUG`，测试源码也在断言被关闭时直接拒绝编译。

## Attention V 聚合 AVX2 实验后端（2026-08-28）

- 新增 `AccumulateAttentionValuesF32_AVX2`，沿时间维保持严格串行，沿独立 head dimension 以四路 packed-double FMA 向量化。
- 每个时间步仍将累积结果转换并回写 FP32，有限性和 FP32 范围检查仍逐步执行；非 4 倍数 head dimension 使用标量 FMA 尾部。
- `sqrt(head_dim)` 分母从 QK 的 time 循环外提至每 token 一次，除法和 QK 累加顺序不变。
- 新增 `MICIUS_REFERENCE_ATTENTION_VALUE_AVX2`，默认关闭且与 GEMV 开关独立；sandbox 输出实际 attention-value backend。
- 新增 `attention_value_avx2_tests`，覆盖 96 组 sequence/head shape、非对齐、stride padding、1/3/5 等尾部维度、bitwise 差分及非有限/溢出拒绝。
- Linux GCC 五个测试目标、完整 runtime contract、ASan+UBSan 通过；完整 sandbox 与仅 GEMV AVX2 基线 checksum 相同。
- 本地虚拟 EPYC 预筛的五组配对中位提升为 1.31×；该数字存在明显频率漂移，只用于决定进入目标 Windows A/B，不作为最终性能结论。

## Q8_32 scale 预验证可信边界候选（2026-08-28）

- 否决“编译时全局无条件跳过 scale 检查”的旧方案：它无法阻止裸 Core tensor 入口绕过 typed V1 binder。
- 新增 `MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES`，默认关闭且依赖 Q8_32 AVX2；快路径由 engine 实例状态选择，不由调用者参数选择。
- `Micius_BindModelTensorsV1` 在完整验证后复制并移交每个 scale tensor；Core 按指针与元素数精确消费 `7 * num_layers + 1` 个 owned buffer。
- `AIEngineFront::BindValidatedTensors` 改为私有 friend 边界；公开裸 `BindTensors` 与历史 `LoadWeights` mutator 移除，未验证状态只能走 checked GEMV。
- 候选 QKV packed storage 使用作用域守卫，覆盖普通拒绝、异常、reset 回滚和成功重绑定；scale ownership 与模型交换保持同一事务。
- 新增 checked/prevalidated 位级差分、快路径剩余运行时检查、NaN/zero-point/padding、错位指针、调用者 scale 修改、重绑定保留和 Core API 编译期负向测试。
- Linux GCC 默认/checked/prevalidated 三配置各 5/5；三者 2,000-step checksum 相同；ASan+UBSan 通过。LeakSanitizer 因执行环境 ptrace 限制未计为通过。
- 虚拟 EPYC 五组交错预筛中位提升 2.94%；Windows Clang64、Raptor Lake 五组 A/B、VTune 与真实 GGUF parity 仍待验收。
- r3 加固禁止 `MICIUS_ENABLE_FAST_MATH` 与预验证 scale 路径组合；GNU/Clang 直接构建另有编译期拒绝，避免非有限值检查被优化掉后仍授予可信状态。
- unchecked GEMV 声明移入 `src/AI` 私有头文件；公开 checked 入口保持不变。
- packed QKV 分配失败改为抛出 `std::bad_alloc`，由 V1 ABI 准确映射为 `MICIUS_ERROR_OUT_OF_MEMORY`，作用域守卫继续负责回收候选存储。
- r4 修正 `q8_32_gemv_tests` 对私有头文件的 CMake include-root 路径；Windows Clang64 checked/prevalidated 均完成 5/5 CTest，2,000-step 验证均 PASS、7 次抽检且 checksum 同为 `0xb1754270c8a88bcb`。

## Q8_32 双行共享输入 AVX2 候选（2026-08-30）

- 从旧四行 GEMV 只保留“跨输出行调度”和“每行独立累加链”；拒绝动态 INT8 activation 重量化、`abs/sign + maddubs`、主动预取及未声明的对齐假设。
- 新增双行 tile：每 32 列只做 8 次 F32→F64 activation 转换并供两行复用；每行仍保留四条 F64 累加链、原规约树和原标量尾部顺序。
- 没有采用四行 tile：四行 × 四条累加链会占满 AVX2 的 16 个 YMM 寄存器，无法为 scale、权重和 activation 临时值留寄存器。
- 新增 `MICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2`，默认关闭且要求 `MICIUS_REFERENCE_GEMV_AVX2=ON`；sandbox 会报告 `q8_32_f32_avx2_row_pair_experimental`。
- `q8_32_gemv_tests` 扩展为 112 组 shape，覆盖偶/奇行、非对齐、块边界、`-128/127`、负 activation、checked/prevalidated 四路径；新旧结果全部位级一致。
- r5 的 `-march=native` 汇编检查实际使用了 AVX-512 扩展寄存器，不能证明 16-register AVX2 无 spill；严格 AVX2 重编译后确认原双行循环会将一个累加器放到栈上。
- 本地算子微基准中位数提升 15.3%；完整 sandbox 五组交错中位数为 -0.69%，未达到 2% 晋级线，因此候选保持默认关闭，等待 Windows Clang64/Raptor Lake 固核复测。

## Q8_32 双行直接权重加载修订（2026-08-30）

- Windows r5 五组 40,000-step 交错中位吞吐提升约 2.05%；匹配低频 VTune v2 显示整体退休指令减少 3.21%、CPI 增加 2.69%、推算周期减少 0.61%。
- 每四个权重改为边界安全的直接加载；Clang/GCC 可合成为 `vpmovsxbd xmm, m32`，不再保持并移位四个 16-byte packed-weight 值。
- 严格 `-march=haswell -mno-avx512f` 汇编中，双行主热循环不再访问栈上的 YMM；prevalidated 函数体由 2085 bytes 缩至 1968 bytes。
- 112 组位级差分、ASan+UBSan 和完整 2,000-step sandbox 均通过；目标 Windows/Clang64 性能仍待 A/B，因此开关继续默认关闭。

## Attention 验证批处理否决与 query F64 cache 候选（2026-08-31）

- 上传源码确认 `AIDataFront` 的两套 compact、`GlobalBusRectify` 和 iGPU proposal 装填均无生产调用者；公开 ABI 当前关闭 speculative workers，因此没有实际并行写共享路径可验证 90° AoSoA/伪共享假设，本阶段不改 KV 布局。
- 跨完整 sequence 的粘滞验证中位数慢 3.29%；缩短到每个 time row 后，十一轮长测中位数仍慢 1.86% 且九轮落后。两版代码和开关均从交付源码移除，只保留负结果。
- 增加“F64 已略超 `FLT_MAX`、但转换会舍入回有限 `FLT_MAX`”的向量及标量尾部负向测试，防止错误地退化成最终 finite-only 检查。
- 新增默认关闭的 `MICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE`：每个 query head 只做一次精确 FP32→F64 准备，历史 key 扫描仍保持逐维 F64 乘加及原规约顺序；fast-math 组合被配置期和直接编译期双重拒绝。
- 短上下文交叉测试显示 1–3 行仍为负、4 行仅 +0.93%、8 行为 +4.96%；因此前 7 行保留原逐点转换，从第 8 行起才启用 query cache，选择分支位于 key 循环外。
- 208 组非对齐/多维 shape 位级差分覆盖准备值与 dot 结果。严格 AVX2 的 512×32 算子预筛十一轮全部领先，中位数由 90,979 提至 118,389 scans/s（+30.13%），hash/checksum 相同。
- 加入“前 7 行保留原路径、从第 8 行启用缓存”的短上下文阈值后，本地严格 AVX2 完整 sandbox 五轮 20,000-step 交错中位数由 7,285.023 提至 7,777.866 steps/s（+6.765%），五轮全胜且 checksum 相同；六个测试目标与 2,000-step ASan+UBSan 全量运行通过。Windows Clang64/Raptor Lake 固核 A/B 仍是正式晋级门槛。
- 使用完全相同的严格 AVX2/GCC 参数审计 r6 与 r7-cache-OFF：`RunReferenceToken` 均为 `0x11ed` 字节，地址归一化后的符号反汇编 SHA-256 完全相同；确认默认关闭时未改变基线热函数指令序列。
- r6 目标 Windows 五组裸跑中位吞吐提升 4.88%；匹配 VTune 显示退休指令减少 9.20%、CPI 增加 5.67%、推算周期减少约 4.1%，r6 取代 r5 成为当前实验 GEMV 候选。

## Attention query cache 目标结论与 time-quad 候选（2026-08-31）

- r7 在 Windows Clang 22.1.8 / Raptor Lake-P CPU 4 上完成双配置 6/6 CTest、2,000-step 验证和五组 40,000-step 交错裸跑；checksum 全部一致，候选五轮全胜，但独立中位提升仅 1.422%、配对中位提升 1.976%。
- 逐行 VTune 解释了收益受限原因：基线最热点约 12.081B 指令、CPI 0.395；F64 cache 主循环行约 7.180B 指令、CPI 0.560，另有循环行 0.924B 指令、CPI 0.547。整体指令下降 7.43%，CPI 上升 5.07%，八字节查询缓存加载抵消了部分转换消除收益；r7 保留为默认关闭的对照实验，不晋升默认路径。
- 新增默认关闭且与 query cache 互斥的 `MICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD`。每个维度只加载并转换一次 FP32 query，同时推进四个历史 key 行的独立 F64 累加器；四个 dot 完成后仍按时间顺序除法、检查、写 score 和更新 maximum，1–3 行尾部走原标量循环。
- fast-math 组合在 CMake 与直接 GNU/Clang 编译期双重拒绝；sandbox 新增 `attention_score_time_tile=four_rows_shared_query_experimental`，且明确报告没有 F64 query cache。
- 新增 312 组四行 dot 位级差分与 119 组 1–17 行完整 scan/tail 差分；baseline/time-quad 各 7/7 合约测试通过，候选的 ASan+UBSan 专项测试及 600-step 验证通过。
- 生产汇编确认分组热循环每维为一次 query 转换、四次 key 转换和四条独立 FMA 链。r7 与 r8 关闭新开关后的 `AICoreEngine.cpp` 对象逐字节相同，默认基线未改变。
- 本地 512×32 算子七轮全部领先，中位提升 102.98%；完整 sandbox 五组交错中位由 2,263.143 提至 2,629.456 steps/s（+16.186%），配对中位 +15.579%，4/5 获胜且 checksum 相同。由于本地频率/负载漂移明显，这些结果只授予 Windows 目标机 A/B 资格。

## 私有紧凑 Q8_32 scale 实验（2026-09-03）

- 新增默认关闭的 `MICIUS_REFERENCE_GEMV_COMPACT_SCALES`，强制依赖 AVX2 与私有 V1 prevalidated-scales 边界；公共 ABI 仍为 `0x00060003`，16-byte descriptor 不变。
- V1 binder 先完整验证公共 descriptor，再将每个 scale 的原始四字节位模式复制到模型私有连续 `float` 数组；成功模型不保留另一份私有 16-byte-per-block descriptor 数组。
- packed QKV 与其余矩阵统一使用紧凑 scale；热数据由 48 B/block 降为 36 B/block。GEMV 直接读取 F32 scale，不在热循环重建 descriptor 或执行逐 block scale 检查。
- 事务绑定、失败重绑回滚与 checked fallback 保持原契约；sandbox 新增 `gemv_scale_storage=private_f32_per_block_experimental`。
- 默认、R8+compact Release、candidate ASan+UBSan 三套 clean build 均为 7/7 CTest；LeakSanizer 未运行。
- 既有 SmolLM2 32-step harness 中，R8/candidate token 32/32 一致，全部 logits 逐字节一致，轨迹 SHA-256 仍为 `d3da3fad66f19ddb87ae7ae551e925992a21559358c8d1bffb66e797ecbb08b3`。
- 本轮未重新生成外部 oracle，也未作本地性能结论。
- Windows 11、Clang 22.1.8、Raptor Lake-P CPU 4 上，R8 baseline 与 R9
  candidate 的 clean build 均为 7/7 CTest；2,000-step smoke 均 PASS、七次
  抽检且 checksum 同为 `0xb1754270c8a88bcb`。
- 五组交错 40,000-step 中，candidate 仅胜 2/5；baseline/candidate 普通中位数
  分别为 4385.762023/4366.281219 steps/s（-0.4442%），配对变化中位数
  -0.0866%，全部长测 checksum 同为 `0x4b60bb23cf866c7e`。
- 截至 2026-09-04，R9 未达到预设的 4/5 胜场与两个 >2% 中位数门槛；synthetic performance
  gate 失败，未采 VTune，默认开启被拒绝。实验代码继续 default-off；真实
  SmolLM2 吞吐 A/B 尚未运行，不能用既有 parity 结果替代。
- synthetic 结果提交后，CLI 新增独立 `--model-benchmark` 测量入口。GGUF
  映射、转换、V1 binding、reset、prompt、warmup、token/logits hash 均在
  ROI 外；计时区只含固定次数的 reference decode 与预分配 token 写入。
- 当前跟进容器缺少 CMake/Ninja，因此新增入口的 CMake/CTest 标记为
  BLOCKED。GCC 13.3.0 直接编译的 R8/R9 两配置均成功，并各自完成真实
  SmolLM2 `4 warmup + 32 timed` 冒烟；final token、完整 timed-token hash
  与最后 logits 位模式 hash 三者一致。本地未固核数据不作性能结论。
- 文档已预注册 Windows CPU 4 的 clean build/CTest、32-step smoke 与五组
  128-step 交错真实模型 A/B；在该 gate 完成前不写入真实模型性能结论，
  也不采 VTune。

## R9 Windows 真实模型性能实验结案（2026-09-07）

- 数据来自用户提交的完整目标机终端输出；没有在本地重跑或生成新 oracle。
- 含计时入口的 R8/R9 clean build 与 CLI 链接均成功，CTest 均 7/7 PASS；
  10.43/11.02 s 为测试套件耗时，不作为性能证据。
- 第二批 synthetic 40,000-step 五组交错：3/5 胜，中位数
  4364.538522 -> 4390.604896 steps/s（+0.5972%），配对中位 +1.6187%；
  仍未过门槛。第一批 2/5、-0.4442%/-0.0866% 的负结果单独保留。
- 真实模型位于 `models/smollm2-360m-instruct-q8_0.gguf`。复用
  `build-r9-ab-r8-baseline` 与 `build-r9-ab-r9-candidate`，仅切换 compact scales。
- CPU 4、token ID 1、4 warmup + 32 timed 的 smoke 均 PASS；final token
  41530，token hash `0x211e5b382d5e4752`，last-logits hash
  `0xf251fe102e07e8f1`。这是本次 checksum smoke，不是全部 logits parity。
- CPU 4、8 warmup + 128 timed、奇数 R8 先/偶数 R9 先的五组 A/B：
  R8 样本 9.036225/9.265652/9.393939/9.792609/9.692460 steps/s；
  R9 样本 9.156251/9.014302/9.468757/9.994913/9.822512 steps/s。
- 十次均 PASS、完成 128 步，CPU 起止均为 4；final token 30，token hash
  `0x4f9cb9a464361ddf`，last-logits hash `0x796b3ffb22c87b08`，全部一致。
  目标机原始汇总路径为
  `r9-real-ab.JNVf2t/r9-compact-scales-smollm2-cpu4-ab.txt`。
- R9 胜 4/5；普通中位 9.393939 -> 9.468757 steps/s（+0.7964%），
  配对提升中位 +1.3283%。胜场达到门槛，但两个中位提升均未超过 2%，
  因此真实模型性能门 FAIL。两侧首尾均有约 +7.3% 同向漂移；原因无遥测证据。
- R9 性能实验到此结束，不追加跑数、不采 VTune、不默认开启。保留
  default-off 代码，R8（compact OFF）继续作为性能基线。
- 本检查点只修改两份文档并制作源码归档；没有更改执行代码、重建或重跑测试。
  详细原始样本、门槛判定、证据边界与修正后的复现命令见
  `docs/Q8_32_COMPACT_SCALES.md`。

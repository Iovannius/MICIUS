# MICIUS reference 压力沙箱验证记录

日期：2026-08-24  
公开 ABI：`0x00060003`

## 设计边界

旧沙箱依赖 v0.5 raw-weight、Q8_64、Verifier/DraftTree 和 Oracle audit 路径；这些接口在当前版本均已 fail closed，因此不能通过重命名函数继续使用。新沙箱只测已启用的 reference 路径，并将负载拆为：

- `throughput`：完整 reference decode、逐步增长的 attention 上下文与低频 reset；
- `churn`：按指定周期清空 KV/scratch/logits，再从空上下文继续 decode；
- `contracts`：交替提交错误 entry stride/byte size、验证失败绑定事务性、重新绑定有效模型并检查确定输出，同时在预检中覆盖 context-full 与 last-logits 保留。

多线程模式为每个线程创建独立引擎 Context；合成权重只读共享。它用于检查跨 Context 隔离和主机级并行扩展，不测试当前禁用的 speculative worker 阵列。

## 已执行验证

| 项目 | 结果 | 范围 |
|---|---|---|
| 严格编译 | PASS | C++20、AVX2/FMA、`-Wall -Wextra -Wpedantic -Wconversion -Wshadow` 零警告 |
| 全工程 `make check` | PASS | 三套既有测试通过，沙箱目标成功构建 |
| 默认沙箱 | PASS | 4096 steps；16 次 logits/KV 抽检；确定 checksum |
| throughput/churn/contracts | PASS | 单线程、双线程和四线程独立 Context |
| 非默认 GQA/shape | PASS | hidden=96、4Q:1KV、3 layers、intermediate=384、vocab=257 |
| 重复并行执行 | PASS | 优化构建连续 12 次四线程 churn，checksum 完全一致 |
| GCC 静态分析 | PASS | 沙箱翻译单元 `-fanalyzer -fsyntax-only` 无诊断 |
| O3/IPO 等价配置 | PASS | `-O3 -march=native -flto=auto` 与静态 GNU runtime |
| ASan + UBSan | PASS | 三模式、并行 Context，无 sanitizer 报告 |
| TSan | PASS（有限） | churn/contracts 双线程无竞争报告；GCC 提示其不能建模休眠 Chase-Lev 路径的 atomic fence |

## 结果解释

`steps_per_second` 是合成 reference forward 的进程级完成率。它受 hidden/vocab/layers/context、线程数、reset 周期与 `verify_every` 直接影响，不能与真实模型 token/s 混用。`contracts` 的主要指标是 cycles/s，每 cycle 包含两次失败封闭/有效绑定组合和两次确定性 decode，不是纯推理吞吐。

相同二进制、参数和 CPU 条件下 checksum 应保持一致。若 VTune 运行出现 `status=FAIL`、退出码非零、不同线程 checksum 分歧，或相同参数在重复运行中 checksum 变化，应保留完整命令行与输出再定位。

## 未验证边界

- 当前环境没有 CMake，故没有实际执行 CMake 生成的 sandbox 目标；Make 与 CMake 等价旗标/源码分区已验证。
- Windows/MSVC、Windows Clang PDB 和真实 VTune 采样尚未执行。
- speculative 调度器仍然禁用，本沙箱不对其稳定性作任何结论。

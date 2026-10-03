# MICIUS v0.6 修复与重建大纲

状态：第二逻辑迭代完成；真实模型与 tokenizer 对齐阶段待输入  
依据：`MICIUS_v0.6_INCOMPLETE_SOURCE_AUDIT.md`、`MICIUS_v0.6_AUDIT_VALIDATION.md`  
原则：先建立可证明正确的单线程基线，再恢复投机和并发；任何未支持的模型、张量或 ABI 必须显式失败，禁止 silent dummy。

| 阶段 | 代码状态 | 当前证据边界 |
|---|---|---|
| Phase 0 工程/ABI/生命周期 | 已完成 | 严格构建、错误路径与生命周期测试通过 |
| Phase 1 单线程 reference | 已完成强化基线 | 非对称多头/GQA/SwiGLU 独立数学 oracle 通过；尚无目标 GGUF logits oracle |
| Phase 2 KV/状态模型 | 已完成 reference 路径 | 直接读取两层同位置 K/V，证明 layer/position 隔离；context-full 通过 |
| Phase 3 GGUF/张量/量化 | Q8_32 精确契约完成，集成验收待定 | 畸形 GGUF、扩展 V1 entry stride、F16/Q8_0 精确 codec 测试通过；未挂载用户目标模型 |
| Phase 4 Tokenizer/Decoder | 阻塞 | 缺目标 GGUF tokenizer metadata 与可信 token 向量 |
| Phase 5 Speculative | 未开始且强制禁用 | 创建、verifier、tree 与旧无尺寸 ABI 均 fail closed |

## 1. 工作基线

- 用户上传的 20 个 v0.6 文件保持不变。
- 修改只发生在 `micius-v0.6-rebuild/` 工作副本。
- 工作副本初始文件与上传快照 20/20 字节一致。
- 每个阶段必须同时满足：可构建、失败路径可解释、测试有明确 oracle、没有把下一阶段的占位伪装成已支持功能。

## 2. 支持边界

### 2.1 第一条可用路径

首个 correctness target 定义为：

- GGUF v2/v3；
- `general.architecture == "llama"`；
- decoder-only、RMSNorm、RoPE、GQA/SwiGLU；
- 矩阵权重先支持 Q8_0；
- norm 支持 F32/F16；
- embedding 支持 F32/F16/Q8_0；
- reference mode 单线程、一次只提交一个真实 Token；
- speculative mode 默认关闭。

具体 tokenizer 仍必须由目标 GGUF 的 `tokenizer.ggml.model`、token types、scores/merges 和 special-token metadata 决定。在收到目标模型名称或 metadata dump 之前，Phase 4 不得标记完成。

### 2.2 明确拒绝

以下情况在创建或加载阶段返回错误，不做零权重回退：

- 未知 architecture 或 GGUF version；
- 未支持 GGML type；
- 缺失必需 tensor；
- tensor rank/shape/bytes 与配置不一致；
- `hidden_dim != num_heads * head_dim`；
- `num_heads % num_kv_heads != 0`；
- 0 尺寸、乘法溢出、文件越界；
- 当前 SIMD 路径不支持且没有标量尾部的维度。

## 3. Phase 0：工程与生命周期闭合

### 3.1 构建

- 新增 `CMakeLists.txt` 和本地可用 `Makefile`；
- C++20、AVX2、FMA、pthread；
- 修复多余类限定符、错误方法名、缺失 include/intrinsic；
- 删除未实现的 `sandbox` 路由，清理未实现旧接口；
- banner 与 ABI 版本统一为 v0.6-rebuild。

验收：所有翻译单元在 `-Wall -Wextra -Wpedantic` 下完成编译与链接；warning 单独登记，不能靠 `-fpermissive`。

### 3.2 C ABI

- 保留 legacy `Micius_CreateEngine`，只读取旧结构前 5 字段并填入保守默认值；
- 新增带 `config_size + abi_version` 的版本化创建入口；
- 所有 `extern "C"` 入口捕获异常并验证空指针、token 范围和输出指针；
- 增加可查询的错误码/错误字符串；
- `BindTensors` 返回真实结果，不再无条件 true。

验收：旧 5 字段调用不会越界读取；错误配置稳定失败；异常不越过 C ABI。

### 3.3 初始化、分配与销毁

- 初始化 callback、wrapper、TLAB、pipeline 和 queue 全部成员；
- 分配并初始化 `safe_dummy_fp32`，但生产绑定仍禁止必需 tensor 落入 dummy；
- 检查 `_mm_malloc`、`posix_memalign`、mmap 和尺寸乘法；
- raw proposal 显式 placement-new / destroy；
- 全部 WorkerContext 构造完毕后再启动线程；
- `tasks_processed/tasks_stolen` 改为 atomic；
- Shell 先停止/销毁 Engine，再释放权重与 mmap；
- logger 释放 DaemonState，限制 placeholder 数并修复整数存储。

验收：构造任意一步失败时不泄漏已建立对象；析构顺序不允许 Worker 访问已释放模型。

## 4. Phase 1：单线程非投机 reference path

- 新增 reference execution mode，并设为默认；
- `PrefillToken(token)` 只提交该真实 Token；
- `DecodeOne()` 每次只产生一个 logits 向量和一个 Token；
- 禁用双 proposal、work stealing、golden path、Epoch GC 和 verifier；
- 单独保存 residual/scratch，禁止与 KV 共槽；
- Reset 清空 sequence position、KV valid length 和输出队列。

验收：固定小模型/合成权重下，连续两次相同输入得到相同逐层结果；Prefill 长度严格等于输入 Token 数。

## 5. Phase 2：重建 KV 与状态模型

### 5.1 存储布局

采用逻辑布局：

```text
K[layer][position][kv_head][head_dim]
V[layer][position][kv_head][head_dim]
```

并单独分配：

```text
residual[hidden_dim]
normed[hidden_dim]
qkv[q_dim + 2 * kv_dim]
attention[hidden_dim]
ffn[2 * intermediate_dim]
```

- `seq_len` 只按 Token/position 增长；
- layer loop 不改变时间长度；
- 当前 Token 的 K/V 只写对应 layer/position；
- 超过 max_seq_len 明确返回 context-full，不钳制 RoPE 到最后位置。

### 5.2 内核边界

- GEMV 权重行改用合法 stride 或 unaligned load；
- 处理 INT8_MIN；
- 明确只支持 symmetric Q8，拒绝非零 zero_point；
- 所有 SIMD 循环提供标量尾部；
- 去除被调用方违反的 `restrict` 契约；
- RoPE 校验并处理 0、odd 和非 8 倍数 head_dim；
- overlap 搬运改为 `memmove` 或显式拒绝。

验收：每层 KV 地址不重叠；layer/position 索引测试覆盖第一项、最后一项和 context-full。

## 6. Phase 3：GGUF、量化与张量契约

### 6.1 Parser

- 严格拒绝未知 version；
- metadata key 使用 architecture 前缀精确匹配；
- `n_dims <= 4`；
- 所有 `cursor/len/count/product` 使用防溢出检查；
- alignment 必须非零且为 2 的幂；
- tensor offset + computed bytes 必须在映射内；
- ParseHeaders 可重复调用时先清空状态；
- 所有异常在 Shell 命令边界转换成错误消息。

### 6.2 Tensor ABI

版本化 tensor entry 至少包含：

- name；
- data；
- byte size；
- scalar/block type；
- rank 和 dimensions；
- row stride；
- ownership/lifetime 标志。

Core 按 tensor contract 验证每个必需张量，返回缺失/shape/type 错误清单。

### 6.3 Q8_0/F16

- 实现完整 IEEE-754 half 转换；
- Q8_0 转换逐矩阵行进行；
- reference 桥接块宽固定为 32，与 GGUF Q8_0 一一对应；
- scale 数量使用 `rows * (cols/32)`，不合并相邻 scale，不重新量化 int8；
- 禁止全张量 flatten 后跨行合并 block；
- dynamic tensor names 在建 bridge 前一次性拥有并固定地址。

验收：K=32/64/96/128 和多行 tensor 为正向用例；非 32 整倍数、非有限 scale、容量不足与 source byte mismatch 均有负向用例；负/零 scale 与 `INT8_MIN` 必须逐位保留；任何写入都在调用方声明容量内。

## 7. Phase 4：Tokenizer、Prefill 与 Decoder

- 读取 tokenizer model、tokens、scores、types、merges、BOS/EOS/UNK 和 chat template；
- 按目标模型实现 SentencePiece unigram 或 BPE，不以词表字符串扫描代替算法；
- 实现 byte fallback/byte decoder 与跨 Token UTF-8 缓冲；
- Prefill 只写真实 Prompt Token；
- `TREE_REJECTED` 不允许被忽略；reference mode 不返回该状态；
- EOS/special token 按 metadata 停止或过滤。

验收：至少 20 个固定文本向量的 token IDs 与可信参考实现逐项一致；decode(encode(text)) 对普通 UTF-8 文本可逆。

## 8. Phase 5：重新接入 speculative execution

### 8.1 Proposal 协议

- 节点状态：ALLOCATED -> COMPUTING -> READY -> PUBLISHED；
- `commit_counter` 只统计 READY 节点；
- 每个 task slot 有明确 owner、epoch 和回滚路径；
- queue push 失败必须取消 descriptor 并归还 slot；
- proposal counter 使用有界 CAS，不允许超过容量。

### 8.2 Verifier ABI

公开只读访问：

- node count；
- token；
- parent；
- depth/path；
- ready 状态。

验证结果必须返回：

- accepted prefix/leaf；
- first rejected position 的 target correction Token；
- 可选 bonus Token。

所有 index 在提交前验证。

### 8.3 Epoch、history 与 GC

- Epoch 仅负责逻辑失效，不再充当 quiescence；
- per-epoch in-flight 计数归零后才回收；
- accepted path 全部祖先转为 authoritative prefix，在离开上下文前不回收；
- next tree 只有在 accepted prefix 已发布后启动，或显式携带可验证的预测 prefix snapshot；
- TLAB 在 reset/worker exit 时归还；
- history 与 stats 使用原子/锁/不可变 snapshot。

验收：重复 reset、partial accept、full reject、queue full、worker=0/1/N、context-full 后槽计数回到定义值；不存在 committed-but-unready 节点。

## 9. 实施顺序与停线条件

执行顺序固定为：

1. Phase 0 完成并严格构建；
2. Phase 1/2 获得逐层 reference；
3. Phase 3 让 loader contract 可验证；
4. Phase 4 与指定 tokenizer 对齐；
5. 只有前四阶段全部通过，才开始 Phase 5。

遇到以下任一情况必须停线，而不是继续堆补丁：

- 无法确定目标 GGUF architecture/tokenizer；
- tensor shape 与假定布局冲突；
- 中间层不能与 reference 对齐；
- 必须依赖 silent fallback 才能运行；
- 并发路径没有可陈述的 ownership/happens-before 证明。

## 10. 第一重建检查点实装结果

本检查点已经落地：

1. 建立 Makefile/CMakeLists，并采用临时目标文件后原子替换，避免并行构建留下零字节 `.o`；
2. 建立版本化 config/tensor ABI、线程局部错误码和 C 边界异常防火墙；
3. 旧 tensor/raw-weight ABI 因缺少 byte/type/shape 信息而保留符号但拒绝执行；
4. 实现独立单线程 Llama reference forward：embedding、RMSNorm、Q/K/V、RoPE、GQA attention、SwiGLU、LM head 与 greedy argmax；
5. KV 采用 `[layer][position][kv_head][head_dim]`，并提供只读测试接口验证缓存归属；
6. Reset、prefill、decode、logits copy、context-full 和 runtime stats 形成一致状态契约；
7. GGUF v2/v3 parser 加入版本、计数、rank、alignment、range、overlap、重复名称、嵌套深度和重复解析防护；
8. V1 tensor binder 精确核验名称、类型、shape、bytes、row stride、F32 有限值与对称 Q8 descriptor，并拥有内部 scale 副本；
9. F16/Q8_0 codec 独立成可测试模块，按行合并 32-block，处理负 scale 的饱和舍入并验证所有容量；
10. Shell 移除 hash/空白伪 tokenizer 与词表字符串伪 decoder，只接受显式 token ID，并在所有命令异常时 fail closed；
11. speculative 创建、verifier 和 tree ABI 强制返回不支持，旧并发代码不进入 reference 正常路径；
12. 三套测试覆盖 runtime contract、GGUF parser 与 tensor codec，并完成常规、并行及 ASan/UBSan 回归。

## 11. 下一检查点的输入与准入条件

继续到“可输出自然语言”的最短路径需要：

1. 用户实际使用的目标 GGUF（或完整 metadata/tensor manifest）；
2. `tokenizer.ggml.*` 全量 metadata，尤其 model、tokens、scores/merges、token types、BOS/EOS/UNK 和 chat template；
3. 同一模型由可信实现产生的固定 prompt token IDs、至少两个位置的逐层 K/V 或 hidden-state、最终 logits/top-k；
4. 明确目标上下文长度与 RoPE scaling 类型。

只有 tokenizer 向量和 logits/KV 对齐后，Phase 4 才能完成；只有 Phase 0–4 全部成立后，才允许恢复 Phase 5。当前检查点不能据此声称“真实模型已经能生成正常语言”。

## 12. 第二逻辑迭代闭合项

第一检查点的测试虽然通过，但其对称构造掩盖了两个数学错误。本轮按反例驱动完成以下闭合：

1. Q/K 使用非对称逐行权重，并在 position 0/1 分别验证原始行序与 adjacent-pair RoPE，证明 Core 不应再次置换 GGUF 行；
2. Q8_0 两个相邻 block 使用不同 scale，证明合并到 Q8_64 会改变反量化值，因此 ABI 升至 `0x00060003` 并改为精确 Q8_32；
3. 2 个 query head 共享 1 个 KV head 的两位置 attention 由测试侧独立计算 softmax、O projection、SwiGLU、final norm 与 logits；
4. 扩展 entry stride、失败重绑、context-full logits、UINT64 metadata 与 LLaMA 子集拒绝均建立回归测试；
5. Shell 自回归时序按 `emit(T1) -> decode(T1) -> T2` 修复，并把 context-full/numeric/OOM 等状态统一视为同步命令终态。

这些结果证明合成 reference 路径的内部契约已显著收紧，但不能替代真实 GGUF 的逐层 parity、tokenizer/chat template 或 speculative 并发证明。

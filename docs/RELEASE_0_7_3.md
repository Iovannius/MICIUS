# MICIUS v0.7.3 发布说明

整理日期：2026-11-22。v0.7.3 基于 v0.7.2，纳入已通过固定功能验收的 M4-P1，维持单线程和模块化发布。本次封包保持其 `src/`、`include/`、`tests/`、`cmake/`、`examples/`、`validation/`、CMake 配置和数值参数逐字节不变；只更新发布说明、证据、清单、源码校验器的版本提示及 Git 文件规则。

## 新增能力与修复

| 项目 | 交付行为 |
|---|---|
| 稠密架构 | 独立 `parser_qwen3` 支持 Q/K 逐头 RMSNorm、独立查询宽度和分半 RoPE；Llama 3.x 支持 GGUF 频率因子；Qwen2/2.5 和原 Llama 路径保留 |
| 混合权重 | 新增同层级 `MICIUS::quant`，按张量读取 F32/F16/Q8_0/Q4_K/Q6_K；Q4_K_M 作为混合格式处理，不转为 Q8 |
| 消费方式 | 压缩权重在算子消费时逐块反量化为 F32、有序 F64 累加，不保存整模型 F32 副本；扩展路径不量化激活 |
| 兼容性 | 按结构、元数据及执行语义判断，不限制第三方发布者；不支持任意架构或任意聊天模板 |
| 会话 | 修复 Windows 已存快照的列表路径；保留模型/会话隔离、配置、存盘与跨进程恢复 |
| 其他边界修复 | 共享 F16/F32 embedding 的输出权重消费；验收工具的长输入/程序路径、MSYS2 扩展路径及独立头宽度证据解析 |

原 Q8_0 数值路径保持既有契约。混合权重/扩展架构报告 `gguf_block_dequant_ordered_f64`，即使使用 INT8 构建也不代表 W8A8 的旧性能或质量证据适用于它。三个 Parser 可以独立或组合链接；runtime V1 保持 `0x00060003`，既有公开结构布局不变。[设计与支持边界](M4-DENSE-GGUF-AND-WEIGHTS.md)

稠密 `llama`、`qwen2`、`qwen3` 为本轮范围。MoE、混合注意力、滑窗、未知缩放及额外执行语义明确拒绝。聊天仅接受已验证的 byte-level BPE 词表与精确模板身份；Qwen3 限非思考、无工具文本模式。加载成功不自动意味着 `chat` 可用。

## 已完成的验收

合成与契约：INT8 **33/33**、F64 **26/26**，包含独立量化解码与稠密前向参考、冻结旧 Q8 输出、错误绑定/生命周期、文本模板、采样、快照与中断拒绝。三个 Parser 的单独/组合链接、旧 v0.7 C 对象重链接、独立 sampling 组件及合成跨进程续聊已通过。最终发布包的独立解包重建与 SDK 检查另记在包外交付核验报告。

用户于 2026-11-21 执行 `m4-20261121-175700-04111f58`，状态 **M4_SELECTED_MODELS_FUNCTIONAL_PASS**，指定协议 **M4-P1-SELECTED**，随后独立复核通过：

- 原 SmolLM2-360M / Qwen2.5-1.5B Q8_0，INT8/F64、贪心/固定种子：**8** 项回归，其中 **4** 项贪心复现 M2 封存回复。
- Llama-3.2-1B-Instruct / Qwen3-0.6B Q4_K_M，两套构建：**4** 项前向与 **8** 项文字续聊。
- 共 **72** 次正式原生调用，另有 **12** 次元数据预检。连续运行与退出后恢复的回复、有效 KV、logits、历史、配置、随机状态一致；reset 重放通过，累计生成诊断计数按原语义单独核对。
- **1,000** 项原始证据通过散列检查；REVIEW **874** 项、FULL **1,002** 项文件均核对完整集合及解压字节。

context 固定 512，每轮回复最多 32 token；该验收不外推为任意模型或长对话验收。Llama 的两轮回复为 `No` / `blue`，Qwen3 为 `blue` / `blue`，原文保留，未将必须回答 blue 作为功能正确性的替代标准。[实验方案](M4-P1-SELECTED-EXPERIMENT.md) · [原始结果](evidence/m4-functional-result.json) · [独立复核](evidence/m4-independent-review.json)

| 原始对象 | SHA-256 |
|---|---|
| M4-P1 源码 ZIP | `96960284818e5a61604eeb7b36658aa4db375aa3dec3c54ede8fce6874d70a76` |
| M4-P1 result.json | `20d3fc650fff0b4fd5715f79628916c5f2cbc16590d3c3a790953ebf93e45fd6` |
| REVIEW ZIP | `fc583e73a524cd1af5f8c3c39d073e28758f93ea2f52b023f106176cdbe942bd` |
| FULL ZIP | `3f911e07c32c965b14fd8e4b5cf85116cb2059af5245f49a418d9f23c07e23fd` |

[M4-P1 原清单](evidence/M4-P1-MANIFEST.json) 保留交付时的历史身份，其中的待验收状态由本次实机结果补充，不是当前发布状态。

## D1 独立诊断与未覆盖范围

针对 Llama 首轮 `No`，用户执行 `d1-20261121-233647-a3435e15`。同一 GGUF、同样的 42 个模板提示 token 下，MICIUS 与冻结的 llama.cpp 都选择 **2822 = No**，状态 **FIRST_TOKEN_MATCH**。

这支持该固定输入下的首词行为也存在于参考实现，不能唯一归因于模型太小，也不能证明所有计算完全相同。完整 logits 存在差异，42 个提示前缀有 1 处 argmax 不同；最后一行 RMSE 0.156452、最大绝对差 0.791764。差异来源尚未归因，不宣称是已确认的正常浮点误差。完整数值等价、完整回复对照、广泛质量、长上下文和新路径性能均不作为本版通过项。[完整诊断](M4-D1-FIRST-TOKEN.md)

## 封包、构建和升级

源码包按 v0.7 / v0.7.1 / v0.7.2 的目录层次组织，保留研究文档；不含模型、用户对话、构建缓存、可执行文件或 Git 元数据。Windows Clang64 SDK 提供 INT8/F64 独立前缀、9 个静态组件、6 个 C 头文件、CMake 导出、示例、CLI、PCRE2 DLL 和许可证。外部 text/session 开发仍需 PCRE2 开发依赖，预编译程序面向本次构建机 ISA。

将源码顶层目录改名或覆盖到 `MICIUS/` 后，在 Git Bash 运行：

```bash
python -B scripts/verify_release.py
bash scripts/build-cli.sh
./micius_cli/micius_cli.exe
```

`python` 需要已配置 Clang64 PATH；构建脚本自行补齐默认工具链路径。脚本先完整构建与测试，再生成与 EXE 同名的 `micius_cli/` 目录。中间目录为 `build-cli-int8/`。已有程序/构建目录不会被覆盖，应先保存或改名；`.micius-cache/` 和模型目录保留。源码、产物和会话互相分离。

快照继续核对模型内容、容量、后端及数值构建指纹，不保证 v0.7.2 快照直接迁移，也不保证任意自行编译程序与 SDK 的快照互换。原程序和会话应保留。发布整理不执行 Git 操作；用户按自己的流程提交、推送和打标签。

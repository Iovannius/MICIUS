# M4 · 稠密 GGUF 与混合权重（v0.7.3）

本工作单元基于 v0.7.2，修复已存会话列表的 Windows 路径问题，扩展稠密模型的解析与计算语义，并接入 Q4_K_M 常见的混合权重。推理保持单线程。M4-P1 的指定实机功能协议已完成，2026-11-22 纳入 v0.7.3 发布包；没有新增吞吐或模型质量结论。[实际验收与证据](RELEASE_0_7_3.md)

## 支持边界

| 架构模块 | 计算语义与边界 |
|---|---|
| `parser_llama` | 原 Llama 路径；Llama 3.x 的相邻配对 RoPE，以及 GGUF `rope_freqs.weight` 频率因子。声明 `llama3` 缩放时必须提供因子张量 |
| `parser_qwen2` | Qwen2/2.5 稠密模型；Q/K/V bias 在投影之后、RoPE 之前相加；分半 RoPE |
| `parser_qwen3` | Qwen3 稠密模型；逐头 Q/K RMSNorm、分半 RoPE、独立的查询投影宽度，允许 `head_count × head_dim` 不等于 hidden |

按 `general.architecture`、元数据、张量类型、形状和执行语义判断兼容性，不限制作者、组织或文件名。第三方模型若沿用这些稠密结构与张量约定，可以使用对应模块；这不等于支持任意 GGUF。未知架构、MoE、混合注意力、滑窗、不同 K/V 头维度、部分 RoPE、未知缩放和不支持的额外张量会被拒绝。非中性的额外 logits/embedding/residual 缩放、softcapping 与 clamp 也不被静默忽略。YaRN 等其他缩放机制不在本轮范围内。

模型加载和文字对话是两个验收层。分词器仍限定于已支持的 byte-level BPE 词表；聊天模板采用精确身份匹配。本轮新增 `tests/text/dense-templates.json` 中的 Qwen3 非思考/无工具模板和 Llama 3.1/3.2/3.3 文本模板。相同计算架构但模板不同的模型可能可以加载、执行 token 前向，却拒绝 `chat`。不执行任意 Jinja，也不默认为未知模板套用现有提示格式。

## 两条权重消费路径

Q4_K_M 是混合量化的模型配置名，不是单一张量编码。读取每个张量的 GGML 类型，支持 F32、F16、Q8_0、Q4_K、Q6_K；不根据文件名猜测类型。Q4_K/Q6_K 每块 256 个值，分别占 144/210 字节；Q8_0 每块 32 个值占 34 字节。行长度必须符合块大小，范围、溢出、非有限值与维度均在绑定前检查。Norm、bias 与 RoPE 因子向量支持 F32/F16。

| 路径 | 行为 |
|---|---|
| 原 Q8_0 模型，且可由原结构承载 | 保留 v0.7.2 的 Q8_32 桥接与 INT8 四行面板/回退，或独立 F64 参考 |
| 混合权重、Qwen3、独立查询宽度或 RoPE 因子等扩展 | 保留 GGUF 压缩权重，在算子消费时逐块反量化为 F32，按序 F64 累加；激活不再量化 |

扩展路径的诊断名为 `gguf_block_dequant_ordered_f64`。即使链接 INT8 配置，这条新路径也不是已验收的 W8A8 内核。没有将 Q4 重量化成 Q8，没有整模型 F32 权重副本，也没有借此声称 Q4 性能优于 Q8。Attention 仍用既有单线程执行路径。

模块 `MICIUS::quant` 与 `MICIUS::core`、`MICIUS::gguf`、三个 Parser 一同导出。它承载权重格式实现，当前不发布独立的外部量化插件 ABI。Parser 的现有 C 接口可独立链接；增加 Qwen3 工厂、特征标志和 `Micius_FileWeightPathV1` 查询，既有结构布局和 runtime V1 `0x00060003` 不变。直接 V1 张量绑定仍遵守原 Q8_32 契约，新文件权重通过文件会话接口接入。

## 所有权、状态与修复

文件会话持有只读 GGUF 映射；扩展矩阵借用该映射，F16/F32 向量按需要复制。销毁顺序为引擎、辅助缓冲、文件映射。新的文件模型完整加载成功后才替换 Shell 的旧模型；失败加载保留原会话。缺省输出矩阵可与 embedding 共享，包含 F16/F32 的共享情况。不同已打开模型拥有独立的引擎与生命周期。

修复 `Session::List` 在 Windows 扩展路径中把 `kv/chat.mcs` 作为单一组件拼接的问题，现在逐级拼接 `kv` 与 `chat.mcs`，正确识别已保存快照。快照仍恢复有效 KV、logits、历史、配置和 RNG；原 V1 累计生成诊断计数保留累计语义。快照要求相同模型与数值构建指纹；v0.7.2 快照不承诺直接跨构建迁移，原目录与文件不会自动删除。

## 交付前验证与证据边界

两套完整 CTest 计划分别为 INT8 33 项、F64 26 项。除原契约外，新增独立量化/稠密前向、已知模板/续聊和 M4 验收流程三组测试。独立 Python 参考从逻辑量化值生成 GGUF，而非复用 C++ 解码：五种编码逐值核对，Qwen3 Q/K norm 与独立头宽度、Llama RoPE 因子、混合权重和 tied embedding 前向使用完整 logits 对照。故意省去 Q/K norm 或频率因子必须产生可检测差异。

同时检查混合权重多模型生命周期、失败加载保留旧模型、错误元数据拒绝、原 Q8 冻结输出、跨进程续聊、截断记录、篡改程序、异常退出及中断不放行。独立解包后通过 Bash 的 `prepare` 构建、CTest、安装前缀移动、单独/组合 Parser 链接和 v0.7 C 对象重链接。实际通过数量和归档身份以包外最终核验报告为准。

独立解包检查还覆盖含空格和超过 Windows `MAX_PATH` 的证据目录。验证程序以 UTF-8 转宽字符的扩展路径读写 token/logits 文件；启动器在 Windows 工作目录长度受限时使用较短的已有父目录，所有模型和证据参数仍使用绝对路径，并记录实际启动目录。

独立快照证据解析器同步识别 M4 的文件权重、Q/K norm 和 RoPE 因子标志；只有文件权重路径可使用独立查询宽度。验收流程回归实际执行新模型的贪心/固定种子文字续聊，避免沿用旧的 `hidden == heads × head_dim` 校验误拒绝 Qwen3，同时继续拒绝未知或互不兼容的标志。

合成数值验证不能替代新的真实模型数值或质量验收。M1–M3 的旧实机证据仍属于原两款 Q8 模型；准备阶段不运行真实模型；用户随后完成的 M4-P1 功能记录与 D1 首词诊断见发布说明，不将固定重放等同于全面数值或质量验收。

## 在用户终端运行

将源码包解压为 `C:/Projects/MICIUS-M4`，从核心目录运行：

```bash
cd /c/Projects/MICIUS
bash ../MICIUS-M4/scripts/run_m4.sh all
```

默认复核现有 SmolLM2/Qwen2.5 Q8 的 8 项对话/快照用例。成功状态为 `M4_Q8_REGRESSION_PASS_NEW_MODELS_PENDING`，其中 NEW_MODELS_PENDING 必须保留。需要加入新格式模型时，在同一命令追加一个或多个参数：

```bash
bash ../MICIUS-M4/scripts/run_m4.sh all \
  --dense-model 'C:/path/to/dense-q4_k_m.gguf' --chat-dense
```

替换为实际文件路径。`--chat-dense` 要求全部追加模型具有已支持模板；不指定时只运行固定 token 前向、reset 和跨进程重放。模型身份运行前后均核对；成功状态 `M4_SELECTED_MODELS_FUNCTIONAL_PASS` 仅表示所选模型功能重放通过，尚不是独立数值/质量或性能通过。未知模板导致失败时保留证据，不能跳过后改判成功。

`prepare` 只构建与运行合成检查。`run <result-directory>` 只允许尚未采样的准备目录；中断后保留文件并重新 `all`，不拼接旧样本。`MICIUS_MODELS` 可覆盖默认的当前目录 `models/`。完整协议见 [protocol.json](../validation/m4/protocol.json)。

## 格式与模板来源

实现对照 GGML 的块布局、llama.cpp 的量化解码、GGUF 转换器及 Llama/Qwen3 模型构图。该工作区的 `llama.cpp-0.6.0` 源码是格式参考，不被链接或打入 MICIUS SDK。独立参考与模板字节保存在 `tests/reference/dense_fixture.py` 和 `tests/text/dense-templates.json`；它们与交付清单一同封存。模板来源分别为 Llama-3.1/3.2/3.3-Instruct 和 Qwen3 的文本对话模板，支持的是这些固定模板字节与无工具文本语义。

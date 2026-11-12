# MICIUS v0.7.2 发布说明

v0.7.2 在 v0.7.1 上纳入 M3 的持久化对话和采样功能，保持单线程推理与模块化发布。INT8/F64 数值内核、分词规则和聊天模板沿用已验证实现。发布源码中的 `src/`、`include/`、`tests/`、`cmake/`、`examples/` 与 `CMakeLists.txt` 均与已验收 M3 相同。版本整理更新文档、证据和构建辅助脚本，不扩展新的推理优化。

## 新增能力

| 功能 | 行为 |
| --- | --- |
| KV 与上下文存盘 | 保存已消费的有效 KV、最后 logits、token 历史、轮数、采样配置和随机状态；完成整轮后 CLI 自动保存。 |
| 模型与会话隔离 | `.micius-cache/<GGUF-sha256>/contexts/<id>/`；同模型不同对话有独立配置和快照，同一目录由文件锁保护。 |
| 采样 | 会话内的 `sampling.conf` 控制温度、top-k、top-p、重复惩罚、种子和回答长度。温度 0 保留原有贪心行为。 |
| 恢复 | `chat list` / `chat resume ID`，或 `--chat MODEL --session ID --context N`；恢复前校验完整文件、模型、容量和数值构建身份。 |
| 模块 | 新增 `MICIUS::sampling`、`MICIUS::session` 及 `micius_state.h`。原 V1 runtime ABI 仍为 `0x00060003`；采样可独立链接。 |
| CLI 构建 | `bash scripts/build-cli.sh` 在源码根目录生成 `micius_cli/micius_cli.exe`，附带所需运行库和许可证。 |

使用者可以将压缩包的顶层目录改名为 `MICIUS/`，再运行上述脚本。实际 EXE 名为 `micius_cli.exe`，所以文件夹使用下划线。生成目录不放模型或 KV；相对模型路径与默认缓存仍以启动时的工作目录为准。脚本先验证源码、构建并运行全部测试，通过后才发布程序目录；现存构建和产物目录不会被覆盖。[构建说明](BUILD_AND_CONFIGURATION.md)

`sampling.conf` 和 `kv/chat.mcs` 位于同一会话目录内。配置在每轮前重新读取，非法配置保留原状态；改变 seed 从新种子开始随机序列，`:reset` 清空对话并恢复种子。坏快照或不兼容快照在完整校验后拒绝，不替换当前状态。原 V1 累计生成计数不会因 reset 清零，这是诊断语义，不是 KV 恢复差异。[详细契约](M3-PERSISTENCE-AND-SAMPLING.md)

## 已完成的实机验收

运行号：`m3-20261111-041752-5d9d7b63`；2026-11-11 用户执行。状态 `PERSISTENT_CHAT_FUNCTIONAL_PASS`，随后独立复核通过。

- SmolLM2-360M-Instruct Q8_0 与 Qwen2.5-1.5B-Instruct Q8_0，INT8/F64，贪心/固定种子，共 **8/8** 项用例。
- 每项连续两轮、独立首轮、退出后恢复第二轮及 reset 重放，共 **32** 次正常退出的原生调用。
- 连续运行与跨进程恢复的 **8/8 份完整快照逐字节一致**，包含有效 KV、logits、历史、配置、随机状态与数值构建身份。
- reset 重放恢复首轮数值状态及回复；累计诊断计数另行核对。**4/4** 贪心用例匹配 M2 原回复。
- **461** 项原始证据完整；REVIEW 的 **399** 项、FULL 的 **463** 项文件均核对集合、散列与解压流。

协议为 context 512、两轮各最多 32 个回复 token，实际短回复均提前结束。它验证指定模型与构建下的功能一致性；不更新性能结论，不外推长对话质量、跨版本/跨后端兼容或断电持久性。损坏快照、保存失败保护、锁和错误路径由合成契约检查覆盖。

| 原始对象 | SHA-256 |
| --- | --- |
| M3 源码 ZIP | `52a30633887ec5408f153f389d6df6bb5f63ac872f5dcaccd43d33f9ca9f28f7` |
| result.json | `5e1406df8405d5cf5e015b1168648f2b192bf02fdf2d3172109c9e07ecb638ff` |
| REVIEW ZIP | `592f2e1b659a7d77990e2862ac739232d96f0bed6fe9ebfb60039de4e523f5e6` |
| FULL ZIP | `396fac0c06984136723951401eb5429ec8babe07fbed91d457ea1fa4abf85dd0` |

[原始实机结果](evidence/m3-functional-result.json) 保留原字节；[独立复核记录](evidence/m3-independent-review.json) 的本机路径规范为相对运行号/文件名，原始散列与测量保留。[M3 原清单](evidence/M3-MANIFEST.json) 保留交付时的历史身份，其中的“待验收”由本次记录补充，不能单独作为当前状态。

## 发布与使用边界

源码包按 v0.7/v0.7.1 的层次组织，包含公开接口、模块源码、测试、示例、构建脚本和研究文档；不包含真实模型、用户会话、构建缓存或 Git 元数据。Windows Clang64 SDK 为 INT8/F64 两个独立安装前缀，运行库随 CLI 分发，PCRE2 开发依赖仍由外部 C 调用方的构建环境提供。

验证入口为 `python -B scripts/verify_release.py`；INT8 与 F64 完整测试分别为 **30** 和 **23** 项。`scripts/run_m3.sh prepare` 只构建并检查合成模型；`all` 另行执行固定实机协议，需模型文件。发布整理使用已封存实机证据及独立重建，不重新运行目标模型。

旧 v0.7.1 没有可恢复的 KV 存盘，历史空目录不能恢复已退出的对话。v0.7.2 的快照仅在匹配数值构建、后端、模型内容和容量下恢复。发布 SDK 和自行构建的程序不要假定可交换快照；程序将按内嵌身份检查。升级前保存原程序和会话目录；源码手动覆盖不会迁移用户数据。发布包不执行 Git 操作。

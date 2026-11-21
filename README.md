# MICIUS v0.7.3

MICIUS is a single-threaded GGUF inference runtime with modular C interfaces, UTF-8 text chat, persistent KV/context snapshots and per-conversation sampling. v0.7.3 adds dense Qwen3, Llama 3.x frequency factors and native Q4_K/Q6_K mixed-weight decoding while retaining the established Q8_0 path. Runtime V1 remains `0x00060003`.

[Release notes / 中文](docs/RELEASE_0_7_3.md) · [Current status](docs/CURRENT_STATUS.md) · [Documentation](docs/README.md)

## Build and start

Extract the source ZIP; the top-level directory may be renamed to `MICIUS/`. In Git Bash, from that directory:

```bash
bash scripts/build-cli.sh
./micius_cli/micius_cli.exe
```

The script verifies source identity, builds INT8 and runs all **33** synthetic/contract tests, then creates `micius_cli/` containing the executable, required runtime DLL, licenses and a build receipt. Existing `build-cli-int8/` and `micius_cli/` directories are preserved; save or rename them before a fresh build. Models and conversations stay in their own directories. Build outputs and the local `micius_cli.zip` demo are ignored by Git.

Requirements: Windows x64, MSYS2 Clang64, CMake ≥ 3.21, Ninja, Python 3.8+, pkg-config and PCRE2-8. Scripts default to `/c/msys64/clang64/bin`; override with `MICIUS_CLANG64_BIN`. Binaries target the build machine's CPU. The separate F64 reference uses `bash scripts/build-f64-reference-windows.sh` and has **26** tests. [Build and SDK instructions](docs/BUILD_AND_CONFIGURATION.md)

## Models and conversations

```text
load "models/qwen2.5-1.5b-instruct-q8_0.gguf"
chat
```

Enter text at `You:`; replies follow `Assistant:`. `:config`, `:reload`, `:save`, `:reset` and `:exit` retain their v0.7.2 meanings. Use `chat list` and `chat resume ID` to reopen saved conversations. `:log start "session.txt"` / `:log stop` control a separate readable transcript.

Model content hashes select `.micius-cache/<model-sha256>/contexts/<id>/`, containing `sampling.conf` and `kv/chat.mcs`. Relative paths use the process working directory; set `MICIUS_CACHE_ROOT` when needed. Restore requires matching model, capacity and numerical build/backend. Cross-version or arbitrary-build snapshot migration is not promised; retain the previous executable and session data when upgrading.

Parsers accept structurally compatible dense `llama`, `qwen2` and `qwen3` models independently of publisher. Q4_K_M files are interpreted per tensor type: F32, F16, Q8_0, Q4_K and Q6_K. Mixed weights and extended architectures use block dequantization with ordered F64 accumulation, **not Q4-to-Q8 requantization**. Original supported Q8 models retain their established numerical paths. MoE, hybrid attention and unsupported execution semantics are rejected. A loadable model may still have an unsupported tokenizer or chat template. [Exact support boundaries](docs/M4-DENSE-GGUF-AND-WEIGHTS.md)

## Validation and modules

The audited M4-P1 run passed 8 old-Q8 regression cases, 4 new-model forward cases and 8 new-model persistent-chat cases using Llama-3.2-1B-Instruct / Qwen3-0.6B Q4_K_M. The separate D1 comparison reproduced Llama's first token `No` with a frozen llama.cpp reference and the same 42 prompt tokens. Logits differed; this is a fixed first-token diagnostic, not numerical equivalence, response-quality or performance acceptance. [Evidence and limitations](docs/RELEASE_0_7_3.md)

The source includes repeatable synthetic tests and operator-run protocols. `bash scripts/run_m4_selected.sh all` runs the fixed four-model experiment and requires the corresponding model files. It is optional when building this release; source/SDK delivery checks use synthetic models and the archived real-model evidence.

CMake exports `MICIUS::core`, `MICIUS::quant`, `MICIUS::gguf`, `MICIUS::parser_llama`, `MICIUS::parser_qwen2`, `MICIUS::parser_qwen3`, `MICIUS::text`, `MICIUS::sampling` and `MICIUS::session`. Parsers are independently linkable; sampling does not depend on the inference core. [File client](examples/file-consumer/model_probe.c) · [Session client](examples/session-consumer/session_probe.c)

[v0.7.2 history](docs/RELEASE_0_7_2.md) · [INT8 research ledger](docs/INT8_EXPERIMENT_LEDGER.md) · [License](LICENSE)

# MICIUS v0.7.1

MICIUS is a single-threaded GGUF inference runtime. This release adds UTF-8 multi-turn chat to the v0.7 INT8 and F64 backends. Its file loader, Llama and Qwen2 parsers, tokenizer and inference core can be linked separately. The runtime V1 C ABI is `0x00060003`.

## Build and chat

Open Git Bash in the extracted source directory or in your MICIUS checkout after copying the release files. The verified environment is Windows x64 with MSYS2 Clang64, CMake, Ninja, Python, pkg-config and PCRE2-8. Keep the model GGUF in `models/` or pass its absolute path.

```bash
export PATH="/c/msys64/clang64/bin:$PATH"
bash scripts/build-int8-release-windows.sh
./build-int8-release/bin/micius_cli.exe --version
./build-int8-release/bin/micius_cli.exe --chat "models/qwen2.5-1.5b-instruct-q8_0.gguf" --context 1024
```

At `You:`, type a message and press Enter. Use `:reset` for a fresh conversation and `:exit` to leave chat. Use `:log start "session.txt"` and `:log stop` to record readable text. The interactive command shell also accepts `load "path/to/model.gguf"` followed by `chat`. For the F64 backend, run `bash scripts/build-f64-reference-windows.sh` and use the CLI in `build-f64-reference/bin/`.

## Supported use

- SmolLM2-360M-Instruct Q8_0 (`llama`) and Qwen2.5-1.5B-Instruct Q8_0 (`qwen2`) passed the fixed two-turn conversation and reset replay in both backends. Chat requires a supported tokenizer and chat template. [Chat and logging](docs/M2-TEXT-CHAT.md)
- Completed turns reuse the live session's KV. The loader can create a separate context directory for each file session. Persistent KV snapshots and reopening a previous conversation are outside v0.7.1. [Session lifetime](docs/M1-MODULES-AND-ABI.md)
- Install a built SDK with `cmake --install build-int8-release --prefix <prefix>`. A CMake consumer can select `MICIUS::core`, `MICIUS::gguf`, either architecture parser and `MICIUS::text`. [Public C file example](examples/file-consumer/model_probe.c) · [Tokenizer example](examples/text-consumer/token_probe.c)

[Release notes and verification / 中文](docs/RELEASE_0_7_1.md) · [Documentation index](docs/README.md) · [INT8 research ledger](docs/INT8_EXPERIMENT_LEDGER.md) · [License](LICENSE)

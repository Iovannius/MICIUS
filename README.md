# MICIUS v0.7.2

MICIUS is a single-threaded GGUF inference runtime with modular C interfaces, an AVX2 INT8 backend and an independent F64 reference. v0.7.2 adds persistent KV/context snapshots and per-conversation sampling configuration to v0.7.1. It incorporates M3's verified implementation; the runtime V1 C ABI remains `0x00060003`.

[Release notes / 中文](docs/RELEASE_0_7_2.md) · [Persistence and sampling](docs/M3-PERSISTENCE-AND-SAMPLING.md) · [Documentation](docs/README.md)

## Build and start

In Git Bash, open the extracted source directory (it may be renamed to `MICIUS/`):

```bash
bash scripts/build-cli.sh
./micius_cli/micius_cli.exe
```

The script verifies the source, builds the INT8 release and runs all 30 synthetic/contract tests. Only after success does it publish `micius_cli/micius_cli.exe` with its PCRE2 runtime DLL and licenses. The directory uses the executable's actual base name, including the underscore. Build files and logs remain in `build-cli-int8/`. Existing build/output directories are preserved; rename them before a new clean build. The script never runs a real model or modifies Git.

Requirements: Windows x64, MSYS2 Clang64, CMake ≥ 3.21, Ninja, Python 3.8+, pkg-config and PCRE2-8. The script uses `/c/msys64/clang64/bin`; set `MICIUS_CLANG64_BIN` for another location. It compiles for the build machine's CPU. [Full build and SDK instructions](docs/BUILD_AND_CONFIGURATION.md)

## Chat, save and resume

From the source directory, start the CLI and enter:

```text
load "models/qwen2.5-1.5b-instruct-q8_0.gguf"
chat
```

Type at `You:`; replies appear after `Assistant:`. Completed turns are saved automatically. `:config` displays sampling parameters, `:reload` reloads the configuration, `:save` saves explicitly, `:reset` starts this conversation over, and `:exit` returns to the command shell. Use `chat list` and `chat resume ID` to reopen a saved conversation. `:log start "session.txt"` and `:log stop` control a separate readable transcript.

Each model's content hash selects `.micius-cache/<model-sha256>/`, with independent `contexts/<id>/sampling.conf` and `contexts/<id>/kv/chat.mcs`. Set `MICIUS_CACHE_ROOT` to change the root. This path is relative to the working directory, not the executable directory: keep launching from the same source root to find the same sessions. Snapshots restore active KV, logits, token history, sampling parameters and RNG state. Temperature 0 preserves greedy behavior; positive temperatures support top-k, top-p and repetition penalties. Restore requires matching model content, context capacity and numerical build/backend.

## Validation and modules

SmolLM2-360M-Instruct Q8_0 (`llama`) and Qwen2.5-1.5B-Instruct Q8_0 (`qwen2`) passed M3's eight fixed functional cases: both backends, greedy/seeded generation, two-turn continuation in a new process and reset replay. The four greedy cases also matched M2. This is short-conversation functional evidence, not a broad quality or throughput claim. [Evidence and limits](docs/RELEASE_0_7_2.md)

For the independent F64 build use `bash scripts/build-f64-reference-windows.sh`; its CLI is in `build-f64-reference/bin/`. Original INT8 and F64 build scripts remain available. CMake consumers can select `MICIUS::core`, `MICIUS::gguf`, either architecture parser, `MICIUS::text`, `MICIUS::sampling` or `MICIUS::session`. The sampler has no inference-core dependency. [Session client](examples/session-consumer/session_probe.c) · [Sampler client](examples/sampling-consumer/sampling_probe.c)

To independently reproduce the real-model functional protocol, use `bash scripts/run_m3.sh all` from the extracted source root and set `MICIUS_MODELS` to the directory containing both GGUFs. `prepare` only builds and uses synthetic fixtures. Keep original evidence archives; repeated runs produce separate results.

[v0.7.1 history](docs/RELEASE_0_7_1.md) · [INT8 research ledger](docs/INT8_EXPERIMENT_LEDGER.md) · [License](LICENSE)

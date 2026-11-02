# MICIUS v0.7 · INT8

MICIUS (Multicore Inference Concurrency for Integrated UMA Systems) is a C++20 CPU inference runtime. Version 0.7 ships the first INT8 release. It loads supported Llama-family Q8_0 GGUF files, accepts token IDs, and exposes prefill, greedy decode, logits, KV inspection and reset through the V1 C API.

The release uses the I25 four-row AVX2 weight layout and dynamic block-32 activation quantization. The qualified target is Windows x64 Clang64, single-core decode without VNNI. The F64 reference has its own build preset. The public V1 ABI is `0x00060003`.

## Build and run

Run from the repository root in Windows Bash.

```bash
bash scripts/build-int8-release-windows.sh
./build-int8-release/bin/micius_cli.exe --version
./build-int8-release/bin/micius_cli.exe --model-benchmark \
  --model models/smollm2-360m-instruct-q8_0.gguf --token 42 --warmup 4 --steps 32
```

The build script checks the source manifest, builds the CLI, static core and stress program, and runs CTest. Place the GGUF in `models/` or pass an absolute path. Start the CLI without arguments for the interactive shell. After `load`, use `generate` with numeric token IDs. For F64, run `bash scripts/build-f64-reference-windows.sh`.

## Release evidence

On two fixed SmolLM2-360M Q8_0 texts of 2,048 decode positions, I25 improved paired throughput over I22 by **18.46%**. Outputs matched the I22 reference byte for byte. After load and after decode, private commit fell by about **207 MiB**. Peak commit over the process lifetime rose by about **389.5 MiB**. These measurements belong to the frozen research build. Release integration was checked against that program through the ordinary CLI and V1 runtime.

| Entry | Contents |
|---|---|
| [Documentation / 中文文档](docs/README.md) | Supported configurations, usage and release status |
| [C API](include/micius_runtime.h) · [runtime semantics](docs/RUNTIME_ARCHITECTURE_AND_API.md) | Binding, ownership, sequence advancement and errors |
| [Build](docs/BUILD_AND_CONFIGURATION.md) · [verification](docs/VALIDATION_AND_BENCHMARKING.md) | Reproducible Windows builds and tests |
| [INT8 research ledger](docs/INT8_EXPERIMENT_LEDGER.md) | I1–I25 results, profiling, stopped candidates and repeats |
| [License](LICENSE) | Project license |

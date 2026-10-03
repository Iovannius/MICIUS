# Reference attention query F64 cache experiment (r7)

Date: 2026-08-31

Public ABI: `0x00060003` (unchanged)

Status: retained as experimental and default-off; not promoted after target
profiling

## Targeted redundant work

The matching low-frequency Windows VTune runs attribute roughly 29–31% of CPU
time to `AIEngineFront::RunReferenceToken`. Its hottest source line is the
ordered QK dot product. For each query head, the same 32 FP32 query values are
converted to F64 again for every historical key row. At context length 512,
that is 16,384 query conversions instead of 32.

`MICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE=ON` allocates one aligned
`head_dim`-element F64 scratch buffer per reference engine. Immediately before
scanning a query head's key rows, it converts that head once and reuses the
prepared values for every ordered dot product.

Short-context crossover testing found the cache slower for 1–3 rows, only
0.93% ahead at 4 rows, and 4.96% ahead at 8 rows. The engine therefore keeps
the original inline conversion for the first seven rows and enables the F64
scratch path at sequence length 8. Selection occurs outside the key loop.

This is not a different reduction algorithm:

- binary32 to binary64 conversion is exact;
- each key value is still converted at the same point in the hot loop;
- each multiplication and addition remains scalar F64;
- dimension order and the score division are unchanged;
- the cache is private single-thread reference-path scratch, not shared KV
  state and not the proposed 90-degree AoSoA layout.

Fast-math is rejected because the parity claim assumes the ordered reference
reduction remains intact. The sandbox reports the active path as:

```text
attention_score_query_cache=per_head_exact_f64_experimental
```

The baseline reports `per_dot_f32_to_f64`.

## Local pre-screen

An isolated strict-AVX2 benchmark used the sandbox shape `head_dim=32`,
`kv_dim/stride=64`, and 512 key rows. Eleven alternating 20,000-scan runs gave:

- inline-conversion median: 90,979 scans/s;
- prepared-query median: 118,389 scans/s;
- median improvement: 30.13%;
- wins: 11/11;
- score hash and checksum: identical in every run.

Strict `-march=haswell -mno-avx512f` assembly shows two scalar `vcvtss2sd`
instructions per hot dimension in the baseline and one in the cached path.
The one-time query preparation is vectorized to `vcvtps2pd`.

This operator result only promotes the candidate to target-machine testing. It
does not predict a 30% whole-engine gain.

A manually linked strict-AVX2 full sandbox (the local image has no CMake)
provided an integration pre-screen with the r6 direct-load GEMV enabled. Five
alternating 20,000-step runs produced:

- baseline median: 7,285.023 steps/s;
- prepared-query median: 7,777.866 steps/s;
- median improvement: 6.765%;
- wins: 5/5;
- status, CPU affinity, and checksum: identical in every pair.

All six test executables passed. The query-cache full sandbox also passed a
2,000-step ASan+UBSan run with seven verification points. This is GCC/local
evidence only; CMake/Clang64 and Raptor Lake remain target-machine gates.

As a baseline-preservation audit, r6 and r7-with-cache-OFF were compiled with
the same strict AVX2/GCC flags. Both `RunReferenceToken` bodies were `0x11ed`
bytes, and their address-normalized symbolic disassemblies had the same
SHA-256 (`efc61b6f...369b1955`). The only raw-byte differences were ten
relative call displacements caused by the whole function moving 16 bytes in
the object file.

## Target-machine result and decision

Windows 11, Clang 22.1.8, and fixed Raptor Lake-P CPU 4 completed both clean
Release builds and all six CTests. The verified 2,000-step runs produced the
same checksum (`0xb1754270c8a88bcb`). Five alternating 40,000-step runs also
produced the same checksum (`0x4b60bb23cf866c7e`) in every pair:

- baseline median: 3,834.646 steps/s;
- query-cache median: 3,889.153 steps/s;
- median improvement: 1.422%;
- paired median improvement: 1.976%;
- wins: 5/5.

The naked result is positive but small. The line-level hardware-sampling data
explains why it did not scale like the local operator pre-screen. The baseline
hot multiply-add line retired about 12.081 billion instructions at CPI 0.395.
The prepared-query loop retired about 7.180 billion instructions on its
multiply-add line plus 0.924 billion on its loop line, but their CPI rose to
0.560 and 0.547. Loading an eight-byte query cache entry replaces a four-byte
query load plus conversion, increasing hot-loop data traffic and exposing a
latency/bandwidth tradeoff.

Across the full sampled run, retired instructions fell 7.43%, while CPI rose
5.07%; their product implies roughly 2.73% fewer cycles. The profiler wall
time itself moved in the opposite direction, so it is not used as the naked
performance result. The option remains available for controlled comparisons,
but stays default-off and is not the next optimization base.

## Target-machine build matrix

Use the existing r6 direct-load GEMV as the common base and switch only the new
cache option. This revision appends private scratch storage to `AIEngineFront`,
so do not reuse objects built from an older source overlay. Remove both build
directories before configuring:

```bash
cmake -E remove_directory build-r7-baseline
cmake -E remove_directory build-r7-query-cache
```

Then configure the two clean trees:

```bash
cmake -S . -B build-r7-baseline -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DMICIUS_REFERENCE_GEMV_AVX2=ON \
  -DMICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2=ON \
  -DMICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES=ON \
  -DMICIUS_REFERENCE_ATTENTION_VALUE_AVX2=ON \
  -DMICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE=OFF \
  -DMICIUS_ENABLE_FAST_MATH=OFF

cmake -S . -B build-r7-query-cache -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DMICIUS_REFERENCE_GEMV_AVX2=ON \
  -DMICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2=ON \
  -DMICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES=ON \
  -DMICIUS_REFERENCE_ATTENTION_VALUE_AVX2=ON \
  -DMICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE=ON \
  -DMICIUS_ENABLE_FAST_MATH=OFF
```

Both configurations must pass all six CTests. Then run the same alternating
40,000-step CPU-4 protocol. Promotion requires identical checksum/status and a
repeatable whole-sandbox improvement; a single short run is not sufficient.

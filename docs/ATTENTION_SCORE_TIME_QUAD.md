# Reference attention score time-quad experiment (r8)

Date: 2026-08-31

Public ABI: `0x00060003` (unchanged)

Status: experimental, default-off, and ready for target-machine A/B

## Why r8 follows r7

The r7 query-F64 cache reduced retired instructions on the target machine, but
its hot-loop CPI rose from 0.395 to 0.560. Replacing each four-byte FP32 query
load and conversion with an eight-byte F64 scratch load removed arithmetic
work while increasing hot data traffic. Its five-run naked median improved
only 1.422%, so r7 remains default-off.

r8 keeps the query in FP32. Instead of scanning one historical key row at a
time, it evaluates four chronological rows together. For each head dimension,
the query value is loaded and converted to F64 once and then feeds four
independent scalar-double accumulators. This also exposes four independent
dependency chains to the out-of-order core.

## Arithmetic and state contract

`MICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD=ON` changes only the ordered QK
score scan:

- every row has its own accumulator initialized to positive zero;
- every row visits dimensions in the original increasing order;
- each multiplication and addition retains the original scalar F64 order;
- the four completed dots are divided, checked, stored, and incorporated into
  `maximum_score` in chronological row order;
- one to three remaining rows use the original scalar loop;
- sequences shorter than four rows use only the original loop;
- no query scratch allocation and no KV layout change are introduced.

The option is mutually exclusive with
`MICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE`. Fast-math is rejected by both
CMake and direct GNU/Clang compilation because the parity claim requires
ordered accumulation.

The sandbox reports:

```text
attention_score_query_cache=none_time_quad_shared_conversion
attention_score_time_tile=four_rows_shared_query_experimental
```

The baseline reports `per_dot_f32_to_f64` and `single_row`.

## Local qualification

The dedicated differential test covers 312 misaligned/random/special-value
four-row dot cases and 119 complete scans spanning 1–17 rows. Every dot,
stored FP32 score, scalar tail, and chronological maximum matched the baseline
bit-for-bit.

Both local configurations passed all seven contract-test executables. The
candidate also passed the dedicated test and a verified 600-step full sandbox
under ASan+UBSan. LeakSanitizer was disabled because the execution container
cannot enumerate traced `/proc` tasks; this is an environment limitation, not
a leak result.

GCC assembly for the grouped hot loop contains one query `vcvtss2sd`, four key
`vcvtss2sd` instructions, and four independent scalar FMA chains per
dimension. An option-off r7/r8 audit produced byte-identical
`AICoreEngine.cpp` objects, proving the new default-off source path does not
change the baseline core binary.

At `head_dim=32`, key stride 64, and 512 rows, seven alternating strict-math
operator runs gave:

- baseline median: 35,439.955 scans/s;
- time-quad median: 71,936.942 scans/s;
- median improvement: 102.98%;
- wins: 7/7;
- checksum and score hash: identical in every run.

Five alternating 20,000-step full-sandbox pairs gave:

- baseline median: 2,263.143 steps/s;
- time-quad median: 2,629.456 steps/s;
- median improvement: 16.186%;
- paired median improvement: 15.579%;
- wins: 4/5;
- checksum: `0xf5c202a8d9fb36ac` in every run.

The local host showed large frequency/load drift, including one reversed pair.
These numbers qualify the design for Windows testing; they are not a target
performance claim.

## Target-machine build matrix

Use the r6 direct-load GEMV and attention-V backend as the common base. Remove
both build trees first:

```bash
cmake -E remove_directory build-r8-baseline
cmake -E remove_directory build-r8-time-quad
```

Configure clean baseline and candidate trees:

```bash
cmake -S . -B build-r8-baseline -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DMICIUS_REFERENCE_GEMV_AVX2=ON \
  -DMICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2=ON \
  -DMICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES=ON \
  -DMICIUS_REFERENCE_ATTENTION_VALUE_AVX2=ON \
  -DMICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE=OFF \
  -DMICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD=OFF \
  -DMICIUS_ENABLE_FAST_MATH=OFF

cmake -S . -B build-r8-time-quad -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_TESTING=ON \
  -DMICIUS_REFERENCE_GEMV_AVX2=ON \
  -DMICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2=ON \
  -DMICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES=ON \
  -DMICIUS_REFERENCE_ATTENTION_VALUE_AVX2=ON \
  -DMICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE=OFF \
  -DMICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD=ON \
  -DMICIUS_ENABLE_FAST_MATH=OFF

cmake --build build-r8-baseline --parallel &&
ctest --test-dir build-r8-baseline --output-on-failure

cmake --build build-r8-time-quad --parallel &&
ctest --test-dir build-r8-time-quad --output-on-failure
```

Both trees must pass all seven tests. A mutual-exclusion probe should fail at
configure time:

```bash
cmake -S . -B build-r8-invalid -G Ninja \
  -DMICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE=ON \
  -DMICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD=ON
```

## Target A/B protocol

First run the verified smoke pair:

```bash
for dir in build-r8-baseline build-r8-time-quad; do
  echo "build=$dir"
  ./"$dir"/bin/micius_sandbox.exe \
    --mode throughput --steps 2000 --warmup 128 \
    --verify-every 256 --threads 1 --cpu 4 |
    grep -E '^(attention_score_query_cache|attention_score_time_tile|status|wall_seconds|steps_per_second|verifications|worker_start_cpu_0|checksum)='
done
```

Then use five alternating 40,000-step pairs with verification disabled only in
the timed region:

```bash
run_one() {
  label="$1"
  dir="$2"
  echo "run=$i backend=$label"
  ./"$dir"/bin/micius_sandbox.exe \
    --mode throughput --steps 40000 --warmup 256 \
    --verify-every 0 --threads 1 --cpu 4 |
    grep -E '^(attention_score_query_cache|attention_score_time_tile|status|wall_seconds|steps_per_second|worker_start_cpu_0|checksum)='
}

for i in {1..5}; do
  if (( i % 2 == 1 )); then
    run_one baseline build-r8-baseline
    run_one time_quad build-r8-time-quad
  else
    run_one time_quad build-r8-time-quad
    run_one baseline build-r8-baseline
  fi
done
```

Promotion requires identical status/checksum, correct CPU affinity, at least
four paired wins, and both a positive median and paired-median gain above the
2% noise gate. Only after that gate should matching VTune runs be collected.

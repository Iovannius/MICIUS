# Private compact Q8_32 scale experiment (r9)

Date: 2026-09-03

Target-machine updates: 2026-09-04 and 2026-09-07

Public ABI: `0x00060003` (unchanged)

Status: the recorded correctness checks passed within the scopes below. Both
synthetic and SmolLM2 real-model target performance gates failed. R9 remains
experimental and default-off; the performance experiment is closed without
promotion and R8 remains the performance baseline.

## Contract

`MICIUS_REFERENCE_GEMV_COMPACT_SCALES=ON` requires both
`MICIUS_REFERENCE_GEMV_AVX2=ON` and
`MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES=ON`. CMake and the internal kernel
header reject unsupported combinations.

The public `MiciusQ8Block32Scale` and internal `QuantDescriptor` remain 16-byte
descriptors. The V1 binder still validates tensor type, shape, stride, byte
size, alignment and overflow, then checks every descriptor's finite F32 scale,
zero zero-point, and zero padding. Only after this validation succeeds does it
copy each scale's four bytes into an engine-owned contiguous `float` array.

In the candidate, a Q8_32 block's hot weight/scale representation is therefore
32 bytes of unchanged int8 weights plus 4 bytes of scale instead of 32 plus 16.
No private 16-byte-per-block descriptor copy is retained by a successfully
bound compact model. QKV packing likewise allocates only compact F32 scales.

The compact GEMV entries read the F32 array directly. They perform no per-block
scale checks or descriptor reconstruction, while retaining pointer/dimension,
activation-finiteness and output-range checks. Each scale is converted exactly
from binary32 to binary64 at the same point as before; weight loads, four F64
accumulation chains, reduction order, odd-row path, tail path and final F32
conversion are unchanged. Non-V1/untrusted states retain the checked descriptor
entry.

Candidate construction is transactional. Compact buffers remain local until
the complete core bind and reset succeed. A rejected replacement leaves the
previous model, its private scale storage and its session active.

The stress sandbox reports:

```text
gemv_scale_storage=private_f32_per_block_experimental
```

## Local correctness results

- Default-OFF Release clean build: 7/7 CTest passed.
- R8 plus compact-scales Release clean build: 7/7 CTest passed.
- Candidate ASan+UBSan clean build: 7/7 CTest passed.
- LeakSanitizer: NOT RUN (`detect_leaks=0` in this environment).
- Dedicated GEMV coverage includes exact F32 bit copying, `-128`, odd output
  rows, block tails/boundaries, checked/prevalidated/compact bitwise comparison,
  null inputs, non-finite activations and FP32 output overflow.
- Runtime contracts retain rejection of non-finite scales, non-zero zero-point
  or padding, wrong size/stride/alignment, and retain the previous model after a
  rejected rebind. Caller descriptor mutation after a successful bind remains
  isolated from the active model.
- SmolLM2-360M-Instruct Q8_0, existing trusted token IDs and existing harness:
  32/32 greedy tokens matched R8; all 32 by 49,152 logits matched byte-for-byte.
- Candidate logits trajectory SHA-256:
  `d3da3fad66f19ddb87ae7ae551e925992a21559358c8d1bffb66e797ecbb08b3`.

No external oracle was regenerated and no local throughput conclusion was
produced. The target-machine conclusion below was supplied after the original
R9 source archive was created.

## Windows target-machine synthetic result

The submitted target run used Windows 11, Clang 22.1.8
(`x86_64-w64-windows-gnu`), CMake 4.4.2, Ninja 1.13.2, and a Raptor Lake-P
logical CPU fixed to CPU 4. Both builds were clean and differed only in
`MICIUS_REFERENCE_GEMV_COMPACT_SCALES`.

- R8 baseline clean build: 7/7 CTest passed.
- R9 candidate clean build: 7/7 CTest passed.
- The inherited signedness and unused-field warnings appeared symmetrically in
  both builds and were not introduced by the compact-scale switch.
- The 2,000-step smoke pair reported `status=PASS`, seven verifications,
  `worker_start_cpu_0=4`, and the common checksum
  `0xb1754270c8a88bcb`.
- Runtime identity distinguished the expected storage paths:
  `private_q8_32_descriptor_per_block` for the baseline and
  `private_f32_per_block_experimental` for the candidate.

Five 40,000-step pairs were run in alternating order:

| Pair | R8 baseline (steps/s) | R9 candidate (steps/s) | Candidate change |
| ---: | ---: | ---: | ---: |
| 1 | 4350.971698 | 4319.109128 | -0.7323% |
| 2 | 4400.892043 | 4429.602961 | +0.6524% |
| 3 | 4385.762023 | 4365.428301 | -0.4636% |
| 4 | 4370.067832 | 4366.281219 | -0.0866% |
| 5 | 4387.125336 | 4397.625950 | +0.2394% |

All ten timed runs reported `status=PASS`, `worker_start_cpu_0=4`, and checksum
`0x4b60bb23cf866c7e`. The candidate won 2/5 pairs. The R8 and R9 ordinary
medians were respectively 4385.762023 and 4366.281219 steps/s, a -0.4442%
change. The paired-change median was -0.0866%; the arithmetic mean of the five
paired percentage changes was -0.0782%.

The predeclared performance gate required at least 4/5 candidate wins and both
the ordinary-median and paired-median improvements to exceed 2%. R9 met none
of these conditions. Therefore:

```text
TARGET BUILD:                    PASS
TARGET CTEST:                    PASS
TARGET CORRECTNESS SMOKE:        PASS
SYNTHETIC PERFORMANCE GATE:      FAIL
VTUNE:                           NOT RUN (gate did not pass)
DEFAULT-ON:                      REJECTED
DEFAULT-OFF EXPERIMENT:          RETAINED
```

The sandbox reports `model_bytes=1165824`, so the absence of a measurable
benefit is consistent with a cache-resident workload that does not exercise the
intended DRAM-traffic reduction. Cache residency was not measured with hardware
counters, however, and is an explanation rather than a demonstrated result.
CTest wall times are not benchmark evidence.

## Real-model timing harness follow-up

After the synthetic gate failed, the CLI gained a separate non-interactive
measurement mode:

```text
micius_cli --model-benchmark --model PATH --warmup N --steps N \
  [--cpu N] --token ID [--token ID ...]
```

This mode reuses the existing GGUF parser/codec, V1 binder and reference
forward path. Model mapping, tensor conversion, binding, engine reset, prompt
ingestion, warmup, allocation, token hashing and final-logits hashing are all
outside the timed interval. The interval contains only the requested sequence
of `Micius_DecodeOneV1` calls and one preallocated token-array store per step.
It reports the actual start/end logical CPU, all relevant compile-time backend
identities, the complete timed-token checksum, and a bit-pattern checksum of
the last logits vector.

The new mode does not modify the C ABI, parser/codec, model binding, forward
math, attention, tokenizer, scheduler or speculative runtime. Token ID 1 is
used below only as a deterministic, range-checked performance stimulus; it is
not a language prompt, an external oracle, or a replacement for the existing
trusted-token 32-step parity result.

Follow-up validation in the current Linux workspace:

- CMake/CTest: BLOCKED because this follow-up container has neither CMake nor
  Ninja installed.
- Direct GCC 13.3.0 compilation of the complete CLI succeeded for the R8 and
  R9 configurations with the same C++20/AVX2/R8 feature definitions; this was
  a compile/smoke fallback, not a substitute for the Windows CMake/CTest gate.
- The help path succeeded and a missing-model invocation returned nonzero with
  `status=FAIL`.
- Both direct builds loaded the supplied 386,404,992-byte SmolLM2 GGUF and
  completed `--warmup 4 --steps 32 --token 1` with `status=PASS`.
- R8 and R9 both produced final token 41530, token checksum
  `0x211e5b382d5e4752`, and last-logits checksum `0x3375e07bd5e96c75`.
- The local runs were unpinned and scheduled on different CPUs. Their timing is
  not performance evidence and no local speed conclusion is recorded.

## Windows target results completed on 2026-09-07

The following evidence was supplied as complete pasted terminal output by the
user. It was not rerun on the local host. The source log filenames below identify
the target-side files; no original log-file SHA-256 was supplied.

### Follow-up clean builds and second synthetic batch

The measurement-harness source built successfully with the same Windows
Clang 22.1.8 / CMake 4.4.2 / Ninja 1.13.2 configuration recorded below:

- `build-r9-ab-r8-baseline`: compact scales OFF, CLI linked, 7/7 CTest PASS
  (CTest wall time 10.43 s).
- `build-r9-ab-r9-candidate`: compact scales ON, CLI linked, 7/7 CTest PASS
  (CTest wall time 11.02 s).
- These are the builds subsequently reused for the real-model runs. Their
  CTest results close the Windows build/test gate for the measurement harness.
  The earlier Linux CMake/Ninja limitation remains a historical local BLOCKED
  result; CTest wall times carry no speedup claim.

The repeated synthetic smoke used 2,000 steps, 128 warmup, verification every
256 steps, one Context and CPU 4. R8/R9 wall times were 0.448912/0.444001 s and
rates were 4455.218152/4504.493345 steps/s. Both were PASS with seven
verifications and checksum `0xb1754270c8a88bcb`.

The second synthetic batch used 40,000 steps, 256 warmup, no periodic
verification, one Context and CPU 4. Odd pairs ran R8 first; even pairs R9 first.

| Pair | R8 wall (s) | R8 (steps/s) | R9 wall (s) | R9 (steps/s) | R9 change |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 9.111359 | 4390.124652 | 8.884368 | 4502.289713 | +2.5549% |
| 2 | 9.197233 | 4349.134042 | 9.050731 | 4419.532375 | +1.6187% |
| 3 | 9.021766 | 4433.721833 | 9.143099 | 4374.884175 | -1.3270% |
| 4 | 9.267939 | 4315.954080 | 9.110362 | 4390.604896 | +1.7296% |
| 5 | 9.164772 | 4364.538522 | 9.203909 | 4345.979565 | -0.4252% |

All ten runs were PASS, reported CPU 4, and had checksum
`0x4b60bb23cf866c7e`. R9 won 3/5. Ordinary medians:
4364.538522 -> 4390.604896 steps/s (+0.5972%). Paired-change median:
+1.6187%. This second batch also failed all three gate criteria. Both synthetic
batches are retained separately; the second does not replace the first.

### SmolLM2 32-step smoke

Model: `./models/smollm2-360m-instruct-q8_0.gguf`. Each fresh process used
`--warmup 4 --steps 32 --cpu 4 --token 1`. The console reported 290 tensors
and `general.architecture=llama`.

| Field | R8 baseline | R9 candidate |
| --- | --- | --- |
| status / completed steps | PASS / 32 | PASS / 32 |
| wall seconds | 2.424589 | 2.357808 |
| steps per second | 13.198113 | 13.571928 |
| CPU affinity / start / end | 4 / 4 / 4 | 4 / 4 / 4 |
| final token | 41530 | 41530 |
| timed-token checksum | `0x211e5b382d5e4752` | `0x211e5b382d5e4752` |
| last-logits checksum | `0xf251fe102e07e8f1` | `0xf251fe102e07e8f1` |

Target log directory: `r9-real-smoke.kYJOul`.
`real_model_smoke_comparison=PASS` was reported. These are checksum checks
for this deterministic stimulus, not a comparison of every intermediate logits
vector. The earlier trusted-token byte-for-byte parity result is a separate
test with its own trajectory SHA-256. The Windows last-logits checksum also
differs from the earlier GCC/Linux smoke checksum; cross-toolchain bitwise
identity was not established, and no cause is inferred from hashes alone.

### SmolLM2 five-pair throughput result

Each process used the same model and
`--model-benchmark --warmup 8 --steps 128 --cpu 4 --token 1`.
All ten runs used a fresh Context, the same 1-token stimulus, eight warmup
decodes and 128 timed decodes. Only compact-scale storage differed between
the two builds; the other reported GEMV/attention identities were identical.

| Pair | Order | R8 wall (s) | R8 (steps/s) | R9 wall (s) | R9 (steps/s) | R9 change |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | R8 then R9 | 14.165207 | 9.036225 | 13.979520 | 9.156251 | +1.3283% |
| 2 | R9 then R8 | 13.814462 | 9.265652 | 14.199658 | 9.014302 | -2.7127% |
| 3 | R8 then R9 | 13.625807 | 9.393939 | 13.518142 | 9.468757 | +0.7964% |
| 4 | R9 then R8 | 13.071082 | 9.792609 | 12.806515 | 9.994913 | +2.0659% |
| 5 | R8 then R9 | 13.206141 | 9.692460 | 13.031290 | 9.822512 | +1.3418% |

For every run: `status=PASS`, `completed_steps=128`, requested/start/end CPU
4, final token 30, timed-token checksum `0x4f9cb9a464361ddf`, and
last-logits checksum `0x796b3ffb22c87b08`. All five pair comparisons and
the final `real_model_pair_comparison` reported PASS. Storage identities
were `private_q8_32_descriptor_per_block` (R8) and
`private_f32_per_block_experimental` (R9).

Target aggregate log:
`r9-real-ab.JNVf2t/r9-compact-scales-smollm2-cpu4-ab.txt`.

The calculation uses the printed throughput samples:
paired change = `100 * (R9_i / R8_i - 1)`; ordinary-median change =
`100 * (median(R9) / median(R8) - 1)`.

| Gate metric | Observed | Required | Result |
| --- | ---: | ---: | --- |
| R9 pair wins | 4/5 | at least 4/5 | PASS |
| Ordinary-median change | +0.7964% | greater than +2% | FAIL |
| Paired-change median | +1.3283% | greater than +2% | FAIL |

Ordinary throughput medians were 9.393939 (R8) and 9.468757 (R9) steps/s.
The arithmetic mean of the five paired percentage changes was +0.5639%.
The real-model performance gate therefore FAILS despite 4/5 wins.

Both series ended about 7.3% above their first-pair rates, with intervening
variation. This common time drift is visible in the samples, but its cause is
not established: frequency, power, temperatures and background load were not
logged. No observations are discarded and no drift correction is applied.
The 32-step smoke and 128-step trial use different position ranges/warmup;
their rates are not an A/B comparison of the scale-storage switch.

### Final decision and evidence limits

```text
TARGET HARNESS BUILD / CTEST:     PASS (7/7 on each build)
TARGET REAL-MODEL SMOKE:         PASS (reported checksum checks)
TARGET REAL-MODEL PAIR CHECKS:   PASS (10 runs, common token/final-logits hashes)
SYNTHETIC PERFORMANCE GATE:      FAIL (both separately retained batches)
REAL-MODEL PERFORMANCE GATE:    FAIL
VTUNE:                           NOT RUN (gate did not pass)
DEFAULT-ON:                      REJECTED
DEFAULT-OFF EXPERIMENT:          RETAINED
PERFORMANCE BASELINE:            R8 (compact scales OFF)
R9 PERFORMANCE EXPERIMENT:       CLOSED WITHOUT PROMOTION
```

The 48 -> 36 B/block representation saves 25% of weight-plus-scale bytes
and 75% of scale bytes. It is not evidence of a measured 25% DRAM-traffic
reduction or a speedup. A model larger than cache alone does not establish a
DRAM-bandwidth bottleneck. No new profiler/counter or assembly result is
claimed by this results-only checkpoint; instructions, CPI and cycles are
NOT RUN for this target R9 comparison.

No additional benchmark repetitions, builds, external-oracle regeneration or
kernel changes were performed for this final record. Only this document and
`docs/CHANGELOG_CHECKPOINT.md` changed relative to the 2026-09-04 harness
source archive (SHA-256
`0100c2d120c9e3922587a4cc8e41a3e363034edb8cd615e2019df3c35a17ff3a`).
The recorded negative results close this experiment; they do not demonstrate
that compact storage can never help other models or execution designs.

## Modified source files

- `CMakeLists.txt`
- `include/AI/AICoreEngine.h`
- `src/AI/AICompactScaleInternal.h` (new)
- `src/AI/AICoreEngine.cpp`
- `src/AI/AIComputKrnl.cpp`
- `src/AI/AIComputKrnlInternal.h`
- `src/micius_runtime.cpp`
- `include/app/MiciusShell.h` (post-target measurement harness)
- `src/app/MiciusCLIMain.cpp` (post-target measurement harness)
- `src/app/MiciusShell.cpp` (post-target measurement harness)
- `tests/q8_32_gemv_tests.cpp`
- `tests/stress/AIEmuSandbox.cpp`
- `docs/CHANGELOG_CHECKPOINT.md`
- `docs/Q8_32_COMPACT_SCALES.md` (new)

No attention, tokenizer, scheduler, speculative ABI, GGUF codec/parser or
public ABI source was modified.

## Exact local build configurations

All three builds used CMake 4.4.3, Ninja 1.13.2 and GCC 13.3.0 on the local
validation host. `ARCH=native`, `ALIGN=32`, CLI, sandbox and `BUILD_TESTING`
retained their defaults unless stated below.

Default-OFF Release:

```text
CMAKE_BUILD_TYPE=Release
MICIUS_ENABLE_LEGACY_TUNING=ON
MICIUS_ENABLE_FAST_MATH=OFF
MICIUS_ENABLE_LTO=ON
MICIUS_USE_LLD=ON (requested; LLD was not found, so the default linker was used)
MICIUS_STATIC_RUNTIME=ON
MICIUS_REFERENCE_GEMV_AVX2=OFF
MICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2=OFF
MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES=OFF
MICIUS_REFERENCE_GEMV_COMPACT_SCALES=OFF
MICIUS_REFERENCE_ATTENTION_VALUE_AVX2=OFF
MICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE=OFF
MICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD=OFF
```

R8 plus compact-scales Release candidate:

```text
CMAKE_BUILD_TYPE=Release
MICIUS_ENABLE_LEGACY_TUNING=ON
MICIUS_ENABLE_FAST_MATH=OFF
MICIUS_ENABLE_LTO=ON
MICIUS_USE_LLD=ON (requested; LLD was not found, so the default linker was used)
MICIUS_STATIC_RUNTIME=ON
MICIUS_REFERENCE_GEMV_AVX2=ON
MICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2=ON
MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES=ON
MICIUS_REFERENCE_GEMV_COMPACT_SCALES=ON
MICIUS_REFERENCE_ATTENTION_VALUE_AVX2=ON
MICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE=OFF
MICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD=ON
```

Candidate ASan+UBSan:

```text
CMAKE_BUILD_TYPE=Debug
CMAKE_CXX_FLAGS=-O1 -g -fsanitize=address,undefined -fno-omit-frame-pointer -fno-sanitize-recover=all
CMAKE_EXE_LINKER_FLAGS=-fsanitize=address,undefined
MICIUS_ENABLE_LEGACY_TUNING=OFF
MICIUS_ENABLE_FAST_MATH=OFF
MICIUS_ENABLE_LTO=OFF
MICIUS_USE_LLD=OFF
MICIUS_STATIC_RUNTIME=OFF
MICIUS_REFERENCE_GEMV_AVX2=ON
MICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2=ON
MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES=ON
MICIUS_REFERENCE_GEMV_COMPACT_SCALES=ON
MICIUS_REFERENCE_ATTENTION_VALUE_AVX2=ON
MICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE=OFF
MICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD=ON
ASAN_OPTIONS=detect_leaks=0:halt_on_error=1:abort_on_error=1
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1
```

## Windows target-machine synthetic A/B commands (completed)

Historical commands, retained for provenance only; both synthetic batches are
complete and are not to be repeated for this gate. They used Git Bash at the
R9 source root, refused existing build directories, and differed only in
`MICIUS_REFERENCE_GEMV_COMPACT_SCALES`. These three blocks are a record of the
original session, not a new paste-and-run instruction.

```bash
set -euo pipefail

BASE=build-r9-ab-r8-baseline
CAND=build-r9-ab-r9-candidate

clang++ --version
cmake --version
ninja --version

for dir in "$BASE" "$CAND"; do
  if [ -e "$dir" ]; then
    echo "refusing non-clean build directory: $dir" >&2
    exit 1
  fi
done

cmake -S . -B "$BASE" -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DARCH=native -DALIGN=32 \
  -DBUILD_TESTING=ON \
  -DMICIUS_ENABLE_LEGACY_TUNING=ON \
  -DMICIUS_ENABLE_LTO=ON \
  -DMICIUS_USE_LLD=ON \
  -DMICIUS_STATIC_RUNTIME=ON \
  -DMICIUS_REFERENCE_GEMV_AVX2=ON \
  -DMICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2=ON \
  -DMICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES=ON \
  -DMICIUS_REFERENCE_GEMV_COMPACT_SCALES=OFF \
  -DMICIUS_REFERENCE_ATTENTION_VALUE_AVX2=ON \
  -DMICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE=OFF \
  -DMICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD=ON \
  -DMICIUS_ENABLE_FAST_MATH=OFF

cmake -S . -B "$CAND" -G Ninja \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_BUILD_TYPE=Release \
  -DARCH=native -DALIGN=32 \
  -DBUILD_TESTING=ON \
  -DMICIUS_ENABLE_LEGACY_TUNING=ON \
  -DMICIUS_ENABLE_LTO=ON \
  -DMICIUS_USE_LLD=ON \
  -DMICIUS_STATIC_RUNTIME=ON \
  -DMICIUS_REFERENCE_GEMV_AVX2=ON \
  -DMICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2=ON \
  -DMICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES=ON \
  -DMICIUS_REFERENCE_GEMV_COMPACT_SCALES=ON \
  -DMICIUS_REFERENCE_ATTENTION_VALUE_AVX2=ON \
  -DMICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE=OFF \
  -DMICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD=ON \
  -DMICIUS_ENABLE_FAST_MATH=OFF

cmake --build "$BASE" --parallel
ctest --test-dir "$BASE" --output-on-failure
cmake --build "$CAND" --parallel
ctest --test-dir "$CAND" --output-on-failure
```

First run the correctness smoke pair on Raptor Lake-P logical CPU 4:

```bash
FIELDS='^(gemv_backend|gemv_scale_validation|gemv_scale_storage|attention_value_backend|attention_score_query_cache|attention_score_time_tile|status|wall_seconds|steps_per_second|verifications|worker_start_cpu_0|checksum)='

for item in "baseline:$BASE" "candidate:$CAND"; do
  label=${item%%:*}
  dir=${item#*:}
  echo "phase=smoke backend=$label"
  "./$dir/bin/micius_sandbox.exe" \
    --mode throughput --steps 2000 --warmup 128 \
    --verify-every 256 --threads 1 --cpu 4 |
    grep -E "$FIELDS"
done
```

Only if both smoke runs pass with identical verification count and checksum,
run five 40,000-step pairs in alternating order:

```bash
run_one() {
  label=$1
  dir=$2
  pair=$3
  echo "phase=timed pair=$pair backend=$label"
  "./$dir/bin/micius_sandbox.exe" \
    --mode throughput --steps 40000 --warmup 256 \
    --verify-every 0 --threads 1 --cpu 4 |
    grep -E "$FIELDS"
}

for pair in 1 2 3 4 5; do
  if (( pair % 2 == 1 )); then
    run_one baseline "$BASE" "$pair"
    run_one candidate "$CAND" "$pair"
  else
    run_one candidate "$CAND" "$pair"
    run_one baseline "$BASE" "$pair"
  fi
done | tee r9-compact-scales-cpu4-ab.txt
```

Every timed run must report `status=PASS`, `worker_start_cpu_0=4`, and one
common checksum. Report all ten raw throughput samples. Promotion still
requires at least four candidate wins and both ordinary-median and paired-
median improvement above the 2% noise gate. Do not collect VTune data before
that gate passes.

## Windows SmolLM2 real-model commands (completed; reproduction reference)

The real-model tests are complete and the performance gate failed. These
commands document the procedure; they are not a request to repeat the tests.
The clean build configuration in the preceding section was used for the
2026-09-07 harness builds too. Both CLI executables were built by default and
both 7/7 CTest results were submitted before the real-model runs. The same
`build-r9-ab-r8-baseline` and `build-r9-ab-r9-candidate` binaries were reused.

The actual model location is `./models/smollm2-360m-instruct-q8_0.gguf`.
An earlier root-directory path was incorrect and caused the file guard to
exit before the smoke started. The blocks below run in a child Bash so their
failure handling does not directly terminate the interactive shell. Each
invocation creates a new log directory and preserves earlier logs.

### 32-step checksum smoke

```bash
bash <<'BASH'
set -euo pipefail
cd /c/Projects/MICIUS

BASE=build-r9-ab-r8-baseline
CAND=build-r9-ab-r9-candidate
MODEL='./models/smollm2-360m-instruct-q8_0.gguf'

if [ ! -f "$MODEL" ]; then
  echo "model file not found: $MODEL" >&2
  exit 1
fi

OUT=$(mktemp -d r9-real-smoke.XXXXXX)
echo "logs=$OUT"

for item in "baseline:$BASE" "candidate:$CAND"; do
  label=${item%%:*}
  dir=${item#*:}
  log="$OUT/$label.txt"

  echo "phase=real_model_smoke backend=$label"
  "./$dir/bin/micius_cli.exe" --model-benchmark \
    --model "$MODEL" --warmup 4 --steps 32 --cpu 4 --token 1 \
    2>&1 | tr -d '\r' | tee "$log"

  storage=private_q8_32_descriptor_per_block
  if [ "$label" = candidate ]; then
    storage=private_f32_per_block_experimental
  fi

  for expected in \
    mode=real_model_decode_throughput status=PASS completed_steps=32 \
    cpu_affinity=4 worker_start_cpu_0=4 worker_end_cpu_0=4 \
    "gemv_scale_storage=$storage"; do
    if ! grep -qx "$expected" "$log"; then
      echo "smoke failed: $label missing $expected" >&2
      exit 1
    fi
  done
done

for key in final_token token_checksum last_logits_checksum; do
  lhs=$(sed -n "s/^$key=//p" "$OUT/baseline.txt")
  rhs=$(sed -n "s/^$key=//p" "$OUT/candidate.txt")
  if [ -z "$lhs" ] || [ "$lhs" != "$rhs" ]; then
    echo "smoke mismatch: $key baseline=$lhs candidate=$rhs" >&2
    exit 1
  fi
done

echo "real_model_smoke_comparison=PASS logs=$OUT"
BASH
```

### Five 128-step interleaved pairs

```bash
bash <<'BASH'
set -euo pipefail
cd /c/Projects/MICIUS

BASE=build-r9-ab-r8-baseline
CAND=build-r9-ab-r9-candidate
MODEL='./models/smollm2-360m-instruct-q8_0.gguf'

if [ ! -f "$MODEL" ]; then
  echo "model file not found: $MODEL" >&2
  exit 1
fi

OUT=$(mktemp -d r9-real-ab.XXXXXX)
MASTER="$OUT/r9-compact-scales-smollm2-cpu4-ab.txt"
echo "aggregate_log=$MASTER"

run_one() {
  local label=$1 dir=$2 pair=$3
  local log="$OUT/${label}-pair${pair}.txt"
  local storage=private_q8_32_descriptor_per_block
  if [ "$label" = candidate ]; then
    storage=private_f32_per_block_experimental
  fi

  {
    echo "phase=real_model_timed pair=$pair backend=$label"
    "./$dir/bin/micius_cli.exe" --model-benchmark \
      --model "$MODEL" --warmup 8 --steps 128 --cpu 4 --token 1
  } 2>&1 | tr -d '\r' | tee "$log" | tee -a "$MASTER"

  for expected in \
    mode=real_model_decode_throughput status=PASS \
    prompt_tokens=1 warmup_steps=8 timed_steps=128 completed_steps=128 \
    cpu_affinity=4 worker_start_cpu_0=4 worker_end_cpu_0=4 \
    "gemv_scale_storage=$storage"; do
    if ! grep -qx "$expected" "$log"; then
      echo "run failed: pair=$pair backend=$label missing $expected" >&2
      exit 1
    fi
  done
}

field() {
  sed -n "s/^$2=//p" "$1"
}

compare_pair() {
  local pair=$1 key lhs rhs reference
  for key in final_token token_checksum last_logits_checksum; do
    lhs=$(field "$OUT/baseline-pair${pair}.txt" "$key")
    rhs=$(field "$OUT/candidate-pair${pair}.txt" "$key")
    reference=$(field "$OUT/baseline-pair1.txt" "$key")
    if [ -z "$lhs" ] || [ "$lhs" != "$rhs" ] || [ "$lhs" != "$reference" ]; then
      echo "mismatch: pair=$pair key=$key baseline=$lhs candidate=$rhs first_baseline=$reference" >&2
      exit 1
    fi
  done
  echo "pair_comparison=PASS pair=$pair" | tee -a "$MASTER"
}

for pair in 1 2 3 4 5; do
  if (( pair % 2 == 1 )); then
    run_one baseline "$BASE" "$pair"
    run_one candidate "$CAND" "$pair"
  else
    run_one candidate "$CAND" "$pair"
    run_one baseline "$BASE" "$pair"
  fi
  compare_pair "$pair"
done

echo 'real_model_pair_comparison=PASS' | tee -a "$MASTER"
echo "aggregate_log=$MASTER"
BASH
```

The predeclared criterion remains at least 4/5 wins and both ordinary-median
and paired-median improvement greater than 2%. Only the win count passed.
VTune was not collected and the experiment is closed without promotion.

## Remaining limits

- The experiment remains default-off.
- Windows Clang 22.1.8 / Raptor Lake-P CTest and the reported checksum checks
  passed; both synthetic batches and the real-model performance gate failed.
  VTune was NOT RUN by design.
- The 128-step timed harness verifies timed-token and final-logits hashes; it
  does not dump/byte-compare every intermediate logits vector. Its checks must
  not be described as 128-step full-logits parity.
- The measured 9.393939/9.468757 steps/s medians apply to this single-thread,
  short-context SmolLM2 stimulus and configuration only. Long-context and
  sustained-duration real-model qualification are NOT RUN.
- Cross-platform and additional model/shape performance qualification are NOT RUN.
- LeakSanitizer is NOT RUN.
- No tokenizer, scheduler or speculative-runtime capability is added by R9.

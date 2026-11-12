#ifndef MICIUS_SAMPLING_H
#define MICIUS_SAMPLING_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define MICIUS_SAMPLING_ABI_V1 UINT32_C(0x00010000)
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    double temperature;       /* 0 = greedy, default 0 */
    double top_p;             /* (0,1], default 1 */
    double repeat_penalty;    /* positive, default 1; applied once per ID */
    uint32_t top_k;           /* 0 = full vocabulary */
    uint32_t repeat_last_n;   /* 0 = no repetition penalty */
} MiciusSamplingOptionsV1;
int Micius_InitSamplingV1(MiciusSamplingOptionsV1* options, size_t size);
/* Does not modify logits or history. Penalty -> temperature -> top-k -> top-p.
 * allowed is null or count bytes (zero excludes a token). All logits must be finite.
 * Ties use ascending token ID. Temperature zero skips stochastic filtering.
 * rng_state is SplitMix64 state, initialized to the configured seed. A successful
 * stochastic call consumes exactly one draw; a greedy call consumes none.
 * Failure returns 0 without changing output/RNG. Callers serialize this state
 * with their session. Bit-exact stochastic replay requires the same build. */
int Micius_SampleTokenV1(const MiciusSamplingOptionsV1* options,
    const float* logits, size_t count, const uint8_t* allowed, const uint32_t* history, size_t history_count,
    uint64_t* rng_state, uint32_t* token);
#ifdef __cplusplus
}
#endif
#endif

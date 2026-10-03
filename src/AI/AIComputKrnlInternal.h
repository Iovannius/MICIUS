#pragma once

#include "AIComputKrnl.h"

// The prevalidated path depends on IEEE non-finite checks performed by the V1
// binder. Reject direct GNU/Clang builds that would invalidate that contract;
// the supported CMake path also rejects the corresponding option combination.
#if defined(MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES) && defined(__FAST_MATH__)
#error "prevalidated GEMV scales require strict floating-point validation"
#endif

#if defined(MICIUS_REFERENCE_GEMV_COMPACT_SCALES) && \
    !defined(MICIUS_REFERENCE_GEMV_AVX2)
#error "compact GEMV scales require the AVX2 reference backend"
#endif

#if defined(MICIUS_REFERENCE_GEMV_COMPACT_SCALES) && \
    !defined(MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES)
#error "compact GEMV scales require the private prevalidated V1 boundary"
#endif

#if defined(MICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE) && defined(__FAST_MATH__)
#error "attention query F64 caching requires ordered reference dot products"
#endif

#if defined(MICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD) && defined(__FAST_MATH__)
#error "attention score time quads require ordered reference dot products"
#endif

#if defined(MICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD) && \
    defined(MICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE)
#error "attention score time quads and query F64 caching are mutually exclusive"
#endif

#if defined(_MSC_VER)
#define MICIUS_INTERNAL_FORCE_INLINE __forceinline
#elif defined(__GNUC__) || defined(__clang__)
#define MICIUS_INTERNAL_FORCE_INLINE inline __attribute__((always_inline))
#else
#define MICIUS_INTERNAL_FORCE_INLINE inline
#endif

namespace micius::internal {

// Converting an IEEE binary32 value to binary64 is exact. Preparing one query
// head outside the key/time loop therefore removes repeated conversions
// without changing any multiplication, addition, or reduction ordering.
inline void PrepareAttentionQueryF64(
    const float* query,
    double* prepared_query,
    uint32_t head_dim) noexcept
{
    for (uint32_t dimension = 0; dimension < head_dim; ++dimension) {
        prepared_query[dimension] = static_cast<double>(query[dimension]);
    }
}

inline double AttentionScoreDotPreparedQueryF64(
    const double* prepared_query,
    const float* key,
    uint32_t head_dim) noexcept
{
    double dot = 0.0;
    for (uint32_t dimension = 0; dimension < head_dim; ++dimension) {
        dot += prepared_query[dimension] * static_cast<double>(key[dimension]);
    }
    return dot;
}

// Four chronological key rows keep four independent scalar-double
// accumulators. Each query element is converted exactly once per group, while
// every row retains the baseline dimension order and floating-point operation
// order. The caller consumes the four results chronologically.
MICIUS_INTERNAL_FORCE_INLINE void AttentionScoreDotTimeQuadF32(
    const float* query,
    const float* key0,
    const float* key1,
    const float* key2,
    const float* key3,
    uint32_t head_dim,
    double* dots) noexcept
{
    double dot0 = 0.0;
    double dot1 = 0.0;
    double dot2 = 0.0;
    double dot3 = 0.0;
    for (uint32_t dimension = 0; dimension < head_dim; ++dimension) {
        const double query_value = static_cast<double>(query[dimension]);
        dot0 += query_value * static_cast<double>(key0[dimension]);
        dot1 += query_value * static_cast<double>(key1[dimension]);
        dot2 += query_value * static_cast<double>(key2[dimension]);
        dot3 += query_value * static_cast<double>(key3[dimension]);
    }
    dots[0] = dot0;
    dots[1] = dot1;
    dots[2] = dot2;
    dots[3] = dot3;
}

// Private entry for immutable Q8_32 descriptors owned by an AIEngineFront
// instance after its validated V1 binding path has checked every scale.
// Pointer/dimension, activation, and output-range checks remain enabled.
Q8_32_GEMVStatus GEMV_Q8_32_F32_AVX2_PrevalidatedScales(
    const int8_t* quant_weights,
    const QuantDescriptor* scales,
    const float* input,
    float* output,
    uint32_t input_dim,
    uint32_t output_dim) noexcept;

// Experimental row-pair backend. Each pair shares activation loads and
// F32-to-F64 conversions but preserves the baseline's independent four-chain
// accumulation order for every row.
Q8_32_GEMVStatus GEMV_Q8_32_F32_AVX2_RowPair(
    const int8_t* quant_weights,
    const QuantDescriptor* scales,
    const float* input,
    float* output,
    uint32_t input_dim,
    uint32_t output_dim) noexcept;

Q8_32_GEMVStatus GEMV_Q8_32_F32_AVX2_RowPairPrevalidatedScales(
    const int8_t* quant_weights,
    const QuantDescriptor* scales,
    const float* input,
    float* output,
    uint32_t input_dim,
    uint32_t output_dim) noexcept;

// Private compact-scale entries. The V1 binder has already validated the full
// public descriptors; these kernels consume one immutable F32 scale per block
// directly and retain pointer/dimension, activation, and output-range checks.
Q8_32_GEMVStatus GEMV_Q8_32_F32_AVX2_CompactScales(
    const int8_t* quant_weights,
    const float* scales,
    const float* input,
    float* output,
    uint32_t input_dim,
    uint32_t output_dim) noexcept;

Q8_32_GEMVStatus GEMV_Q8_32_F32_AVX2_RowPairCompactScales(
    const int8_t* quant_weights,
    const float* scales,
    const float* input,
    float* output,
    uint32_t input_dim,
    uint32_t output_dim) noexcept;

} // namespace micius::internal

#undef MICIUS_INTERNAL_FORCE_INLINE

#include "BackendInfo.h"
#include "AI/I25Binding.h"

namespace micius::backend {
const char* GEMVBackend(const MiciusContext* context) noexcept
{
    if (i25::BindingStats(context).active) return "w8a8_avx2_panel4";
#if defined(MICIUS_I13_GEMV4)
    return "w8a8_avx2_row4";
#elif defined(MICIUS_I1_MODE) && MICIUS_I1_MODE == 1
    return "i1_w8a8_block32_scalar_contract_oracle";
#elif defined(MICIUS_I1_MODE) && MICIUS_I1_MODE == 2
    return "i1_w8a8_block32_avx2_int32_experimental";
#elif defined(MICIUS_REFERENCE_GEMV_PREPARED_F64)
    return "q8_32_prepared_f64_avx2_row_pair_experimental";
#elif defined(MICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2)
    return "q8_32_f32_avx2_row_pair_direct_load_experimental";
#elif defined(MICIUS_REFERENCE_GEMV_AVX2)
    return "q8_32_f32_avx2_experimental";
#else
    return "scalar_double_reference";
#endif
}

const char* ScaleValidation() noexcept
{
#if defined(MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES)
    return "private_v1_prevalidated_checked_fallback";
#else
    return "per_call_checked";
#endif
}

const char* ScaleStorage(const MiciusContext* context) noexcept
{
    if (i25::BindingStats(context).active) return "private_panel_f32_per_block";
#if defined(MICIUS_REFERENCE_GEMV_COMPACT_SCALES)
    return "private_f32_per_block_experimental";
#elif defined(MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES)
    return "private_q8_32_descriptor_per_block";
#else
    return "q8_32_descriptor_checked";
#endif
}

const char* AttentionValue() noexcept
{
#if defined(MICIUS_I15_VALUE_TILE32)
    return "ordered_value64_tile32_when_eligible";
#elif defined(MICIUS_REFERENCE_ATTENTION_VALUE_AVX2)
    return "f32_f64_avx2_order_preserving_experimental";
#else
    return "scalar_f64_reference";
#endif
}

const char* QueryCache() noexcept
{
#if defined(MICIUS_I17_SCORE_REUSE)
    return "score64_per_query_when_eligible";
#elif defined(MICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE)
    return "per_head_exact_f64_experimental";
#elif defined(MICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD)
    return "none_time_quad_shared_conversion";
#else
    return "per_dot_f32_to_f64";
#endif
}

const char* ScoreTile() noexcept
{
#if defined(MICIUS_I19_SCORE_GROUP3)
    return "ordered_score64_group3_when_eligible";
#elif defined(MICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD)
    return "four_rows_shared_query_experimental";
#else
    return "single_row";
#endif
}

const char* InputPreparation() noexcept {
#if MICIUS_I1_MODE
    return "dynamic_s8_block32_rne_f64_scale";
#elif defined(MICIUS_REFERENCE_GEMV_PREPARED_F64)
    return "f64_snapshot_per_input";
#else
    return "inline_f32_conversion";
#endif
}
}

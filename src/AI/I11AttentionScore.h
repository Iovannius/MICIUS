#pragma once
#include <cstddef>
#include <cstdint>

// Private prevalidated reference-head scan. UNSUPPORTED leaves outputs untouched
// so the caller can use its existing path. No public runtime ABI is added.
enum class I11ScoreStatus { OK, NUMERIC_ERROR, UNSUPPORTED };
I11ScoreStatus ComputeAttentionScoresI11(
    const float* query, const float* keys, size_t stride, float* scores,
    uint32_t rows, uint32_t head_dim, double denominator, double* maximum) noexcept;

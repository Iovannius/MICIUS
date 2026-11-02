#pragma once
#include "I11AttentionScore.h"

// Private I15-compatible entry. Query preparation is local to this invocation;
// no public ABI, heap allocation or persistent KV representation is added.
I11ScoreStatus ComputeAttentionScoresI17(
    const float* query, const float* keys, size_t stride, float* scores,
    uint32_t rows, uint32_t head_dim, double denominator, double* maximum) noexcept;

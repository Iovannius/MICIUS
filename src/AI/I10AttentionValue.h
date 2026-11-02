#pragma once
#include "AIComputKrnl.h"

// Private experimental entry. Public headers/ABI and the original kernel are frozen.
// Exactly the original F64 FMA + per-time F32 rounding, including error output.
AttentionValueStatus AccumulateAttentionValuesI10(
    const float* probabilities, double sum, const float* values, size_t stride,
    float* output, uint32_t length, uint32_t dimensions) noexcept;

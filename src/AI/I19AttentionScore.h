#pragma once
#include <cstddef>
#include <cstdint>

// Private speculative preparation for exactly three adjacent query heads.
// Input ranges must be readable. scratch/maxima are disposable private storage,
// never the original scores or an externally observable attention output.
// A false return leaves scratch unspecified and preserves the incoming MXCSR;
// the caller then executes I17 in the original head/Softmax/Value order.
// A true return supplies all scores and maxima WITHOUT changing MXCSR.
bool TryPrepareAttentionScoresI19(
    const float* queries,const float* keys,size_t key_stride,
    float* scratch,size_t score_stride,double* maxima,uint32_t rows,
    uint32_t head_dim,uint32_t group_size,double denominator) noexcept;

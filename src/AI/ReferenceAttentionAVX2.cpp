#include "AIComputKrnl.h"

#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

namespace {

inline bool AllFiniteAndF32Range(__m256d values, __m256d maximum) noexcept
{
    const __m256d sign_mask = _mm256_set1_pd(-0.0);
    const __m256d absolute = _mm256_andnot_pd(sign_mask, values);
    const __m256d valid = _mm256_cmp_pd(absolute, maximum, _CMP_LE_OQ);
    return _mm256_movemask_pd(valid) == 0x0F;
}

} // namespace

AttentionValueStatus AccumulateAttentionValuesF32_AVX2(
    const float* unnormalized_probabilities,
    double probability_sum,
    const float* value_rows,
    size_t value_stride,
    float* output,
    uint32_t sequence_length,
    uint32_t head_dim) noexcept
{
    if (!unnormalized_probabilities || !value_rows || !output ||
        sequence_length == 0U || head_dim == 0U ||
        value_stride < static_cast<size_t>(head_dim)) {
        return AttentionValueStatus::INVALID_ARGUMENT;
    }
    if (!std::isfinite(probability_sum) || probability_sum <= 0.0) {
        return AttentionValueStatus::NUMERIC_ERROR;
    }

    const size_t last_time = static_cast<size_t>(sequence_length - 1U);
    const size_t final_column = static_cast<size_t>(head_dim - 1U);
    if (last_time >
        (std::numeric_limits<size_t>::max() - final_column) / value_stride) {
        return AttentionValueStatus::INVALID_ARGUMENT;
    }

    std::fill_n(output, head_dim, 0.0f);
    const __m256d maximum =
        _mm256_set1_pd(static_cast<double>(std::numeric_limits<float>::max()));
    const uint32_t vector_end = head_dim & ~uint32_t{3};

    for (uint32_t time = 0; time < sequence_length; ++time) {
        const double probability =
            static_cast<double>(unnormalized_probabilities[time]) / probability_sum;
        const __m256d probability_vector = _mm256_set1_pd(probability);
        const float* values = value_rows + static_cast<size_t>(time) * value_stride;

        uint32_t dimension = 0;
        for (; dimension < vector_end; dimension += 4U) {
            const __m256d accumulated = _mm256_fmadd_pd(
                probability_vector,
                _mm256_cvtps_pd(_mm_loadu_ps(values + dimension)),
                _mm256_cvtps_pd(_mm_loadu_ps(output + dimension)));
            if (!AllFiniteAndF32Range(accumulated, maximum)) {
                return AttentionValueStatus::NUMERIC_ERROR;
            }
            _mm_storeu_ps(output + dimension, _mm256_cvtpd_ps(accumulated));
        }

        for (; dimension < head_dim; ++dimension) {
            const double accumulated = std::fma(
                probability,
                static_cast<double>(values[dimension]),
                static_cast<double>(output[dimension]));
            if (!std::isfinite(accumulated) ||
                accumulated > static_cast<double>(std::numeric_limits<float>::max()) ||
                accumulated < -static_cast<double>(std::numeric_limits<float>::max())) {
                return AttentionValueStatus::NUMERIC_ERROR;
            }
            output[dimension] = static_cast<float>(accumulated);
        }
    }

    return AttentionValueStatus::OK;
}

#include "I10AttentionValue.h"
#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

// Eight F32 accumulators hold all 64 dimensions. Each update explicitly widens
// the previous rounded F32 value, performs F64 FMA, then rounds back to F32.
// No accumulation crosses a time boundary at F64 precision.
inline __m256 Update8(__m256 a, const float* v, __m256d p,
                      __m256d maximum, __m256d& valid) noexcept
{
    const __m256d low = _mm256_fmadd_pd(p, _mm256_cvtps_pd(_mm_loadu_ps(v)),
                                      _mm256_cvtps_pd(_mm256_castps256_ps128(a)));
    const __m256d high = _mm256_fmadd_pd(p, _mm256_cvtps_pd(_mm_loadu_ps(v + 4)),
                                       _mm256_cvtps_pd(_mm256_extractf128_ps(a, 1)));
    const __m256d sign = _mm256_set1_pd(-0.0);
    valid = _mm256_and_pd(valid, _mm256_cmp_pd(_mm256_andnot_pd(sign, low), maximum, _CMP_LE_OQ));
    valid = _mm256_and_pd(valid, _mm256_cmp_pd(_mm256_andnot_pd(sign, high), maximum, _CMP_LE_OQ));
    __m256 rounded = _mm256_insertf128_ps(_mm256_castps128_ps256(_mm256_cvtpd_ps(low)),
                                         _mm256_cvtpd_ps(high), 1);
    // Compiler-only register barrier (no emitted instruction). Keep the two
    // rounded halves packed now: otherwise LLVM scalarizes them into 16 live
    // XMM accumulators and spills them in the time loop. Also bound the lifetime
    // of the checked F64 values instead of delaying all checks to loop end.
    asm volatile("" : "+x"(rounded), "+x"(valid));
    return rounded;
}

// Kept as a separate symbol so the delivered machine-code audit can identify
// the actual candidate loop. Failure leaves output untouched for original replay.
[[gnu::noinline]] bool Value64(const float* p, double sum, const float* v,
                               size_t stride, float* out, uint32_t n) noexcept
{
    __m256 a0 = _mm256_setzero_ps(), a1 = a0, a2 = a0, a3 = a0;
    __m256 a4 = a0, a5 = a0, a6 = a0, a7 = a0;
    const __m256d maximum = _mm256_set1_pd(std::numeric_limits<float>::max());
    for (uint32_t t = 0; t < n; ++t) {
        const __m256d probability = _mm256_set1_pd(static_cast<double>(p[t]) / sum);
        __m256d valid = _mm256_castsi256_pd(_mm256_set1_epi64x(-1));
        a0 = Update8(a0, v,      probability, maximum, valid);
        a1 = Update8(a1, v + 8,  probability, maximum, valid);
        a2 = Update8(a2, v + 16, probability, maximum, valid);
        a3 = Update8(a3, v + 24, probability, maximum, valid);
        a4 = Update8(a4, v + 32, probability, maximum, valid);
        a5 = Update8(a5, v + 40, probability, maximum, valid);
        a6 = Update8(a6, v + 48, probability, maximum, valid);
        a7 = Update8(a7, v + 56, probability, maximum, valid);
        if (_mm256_movemask_pd(valid) != 15) return false;
        // Do not form a pointer past the last row when the final stride is large.
        if (t + 1U < n) v += stride;
    }
    _mm256_storeu_ps(out, a0);      _mm256_storeu_ps(out + 8, a1);
    _mm256_storeu_ps(out + 16, a2); _mm256_storeu_ps(out + 24, a3);
    _mm256_storeu_ps(out + 32, a4); _mm256_storeu_ps(out + 40, a5);
    _mm256_storeu_ps(out + 48, a6); _mm256_storeu_ps(out + 56, a7);
    return true;
}

bool Disjoint(const void* a, size_t a_bytes, const void* b, size_t b_bytes) noexcept
{
    const uintptr_t x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    if (a_bytes > UINTPTR_MAX - x || b_bytes > UINTPTR_MAX - y) return false;
    return x + a_bytes <= y || y + b_bytes <= x;
}
} // namespace

AttentionValueStatus AccumulateAttentionValuesI10(
    const float* p, double sum, const float* v, size_t stride,
    float* out, uint32_t n, uint32_t dim) noexcept
{
    // Unsupported/aliased inputs keep the original observable ordering. Never
    // add restrict assumptions to the existing public contract.
    const auto original = [&]() noexcept {
        return AccumulateAttentionValuesF32_AVX2(p, sum, v, stride, out, n, dim);
    };
    if (!p || !v || !out || !n || dim != 64 || stride < dim ||
        !std::isfinite(sum) || sum <= 0.0) return original();
    const size_t last = static_cast<size_t>(n) - 1U;
    if (last > (SIZE_MAX - dim) / stride) return original();
    const size_t count = last * stride + dim;
    if (count > SIZE_MAX / sizeof(float) || static_cast<size_t>(n) > SIZE_MAX / sizeof(float) ||
        !Disjoint(out, dim * sizeof(float), v, count * sizeof(float)) ||
        !Disjoint(out, dim * sizeof(float), p, static_cast<size_t>(n) * sizeof(float))) return original();
    const unsigned csr = _mm_getcsr();
    // With unmasked exceptions, retain the original trap order as well.
    if ((csr & 0x1f80U) != 0x1f80U) return original();
    if (Value64(p, sum, v, stride, out, n)) return AttentionValueStatus::OK;
    // Replay a rare numerical error to reproduce the exact partial output and
    // exception flags of the original kernel, including errors in an early lane.
    _mm_setcsr(csr);
    return original();
}

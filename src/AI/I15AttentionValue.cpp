#include "I15AttentionValue.h"
#include "I10AttentionValue.h"
#include <algorithm>
#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <limits>

namespace {

// Work on 32 dimensions using eight four-lane F32 registers. This removes
// the packed-eight-lane extract/insert operations without storing accumulators
// in the time loop. Widen only on the NEXT update: widening the final F32
// subnormal early would incorrectly flush it under DAZ. No reassociation.
// For n <= 8192, |p/sum| <= 1 and |v| <= 2^100, all partial sums
// including directed F32 rounding stay below 2^114 (well below FLT_MAX).
// Check eight input values once instead of checking two F64 FMA results.
// Any failed precondition replays I10; this is not a relaxed numeric contract.
inline void Update8(__m128& a, __m128& b, const float* v, __m256d p,
                    __m256 bound, __m256& valid) noexcept
{
    const __m256 values=_mm256_loadu_ps(v);
    valid=_mm256_and_ps(valid,_mm256_cmp_ps(
        _mm256_andnot_ps(_mm256_set1_ps(-0.0f),values),bound,_CMP_LE_OQ));
    a=_mm256_cvtpd_ps(_mm256_fmadd_pd(p,
        _mm256_cvtps_pd(_mm256_castps256_ps128(values)),_mm256_cvtps_pd(a)));
    b=_mm256_cvtpd_ps(_mm256_fmadd_pd(p,
        _mm256_cvtps_pd(_mm256_extractf128_ps(values,1)),_mm256_cvtps_pd(b)));
    asm volatile("" : "+x"(a), "+x"(b), "+x"(valid));
}

// Separate linked symbol for final machine-code inspection.
[[gnu::noinline]] bool Value32Tile(const double* probabilities, const float* v,
    size_t stride, float* accum, uint32_t count) noexcept
{
    __m128 a0=_mm_loadu_ps(accum);
    __m128 a1=_mm_loadu_ps(accum+4);
    __m128 a2=_mm_loadu_ps(accum+8);
    __m128 a3=_mm_loadu_ps(accum+12);
    __m128 a4=_mm_loadu_ps(accum+16);
    __m128 a5=_mm_loadu_ps(accum+20);
    __m128 a6=_mm_loadu_ps(accum+24);
    __m128 a7=_mm_loadu_ps(accum+28);
    const __m256 bound=_mm256_set1_ps(0x1p100f);
    __m256 valid=_mm256_castsi256_ps(_mm256_set1_epi32(-1));
    for(uint32_t t=0;t<count;++t) {
        const __m256d probability=_mm256_set1_pd(probabilities[t]);
        Update8(a0,a1,v,probability,bound,valid);
        Update8(a2,a3,v+8,probability,bound,valid);
        Update8(a4,a5,v+16,probability,bound,valid);
        Update8(a6,a7,v+24,probability,bound,valid);
        if(t+1U<count)v+=stride;
    }
    if(_mm256_movemask_ps(valid)!=255)return false;
    _mm_storeu_ps(accum,a0);
    _mm_storeu_ps(accum+4,a1);
    _mm_storeu_ps(accum+8,a2);
    _mm_storeu_ps(accum+12,a3);
    _mm_storeu_ps(accum+16,a4);
    _mm_storeu_ps(accum+20,a5);
    _mm_storeu_ps(accum+24,a6);
    _mm_storeu_ps(accum+28,a7);
    return true;
}

bool Value64(const float* p,double sum,const float* v,size_t stride,
             float* out,uint32_t n) noexcept
{
    alignas(32) float accum[64]={};
    alignas(32) double probabilities[32];
    // Bound the revisited value window and normalize each probability only once.
    // Output is committed only on success. On an error the caller restores MXCSR
    // and replays the old entry, including the original partial-output ordering.
    uint32_t begin=0;
    while(begin<n) {
        const uint32_t count=std::min(uint32_t{32},n-begin);
        for(uint32_t j=0;j<count;++j) {
            const double probability=static_cast<double>(p[begin+j])/sum;
            if(!(std::abs(probability)<=1.0))return false;
            probabilities[j]=probability;
        }
        const float* rows=v+static_cast<size_t>(begin)*stride;
        if(!Value32Tile(probabilities,rows,stride,accum,count) ||
           !Value32Tile(probabilities,rows+32,stride,accum+32,count))return false;
        begin+=count;
    }
    std::copy_n(accum,64,out);
    return true;
}

bool Disjoint(const void* a, size_t a_bytes, const void* b, size_t b_bytes) noexcept
{
    const uintptr_t x = reinterpret_cast<uintptr_t>(a), y = reinterpret_cast<uintptr_t>(b);
    if (a_bytes > UINTPTR_MAX - x || b_bytes > UINTPTR_MAX - y) return false;
    return x + a_bytes <= y || y + b_bytes <= x;
}
} // namespace

AttentionValueStatus AccumulateAttentionValuesI15(
    const float* p, double sum, const float* v, size_t stride,
    float* out, uint32_t n, uint32_t dim) noexcept
{
    // Unsupported/aliased inputs keep the original observable ordering. Never
    // add restrict assumptions to the existing public contract.
    const auto original = [&]() noexcept {
        return AccumulateAttentionValuesI10(p, sum, v, stride, out, n, dim);
    };
    if (!p || !v || !out || n < 128U || n > 8192U || dim != 64 || stride < dim ||
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

#include "I13W8A8GEMV.h"
#include <cstddef>
#include <limits>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error "I13 requires strict floating-point arithmetic"
#endif

namespace micius::internal::i13 {
// Separate from the dispatcher so the final linked code can be inspected.
[[gnu::noinline]] Q8_32_GEMVStatus Consume4Rows(const int8_t* weights,
    const QuantDescriptor* scales,const PreparedW8A8Input& x,float* out,
    uint32_t begin,uint32_t end) noexcept {
    const size_t n=x.size(), blocks=n/32U;
    for (uint32_t row=begin;row<end;row+=4U) {
        const auto* w0=weights+static_cast<size_t>(row)*n;
        const auto* w1=w0+n; const auto* w2=w1+n; const auto* w3=w2+n;
        const auto* s0=scales+static_cast<size_t>(row)*blocks;
        const auto* s1=s0+blocks; const auto* s2=s1+blocks; const auto* s3=s2+blocks;
        auto sum=_mm256_setzero_pd();
        for (size_t b=0;b<blocks;++b) {
            const size_t k=b*32U;
            const auto a=_mm256_loadu_si256(reinterpret_cast<const __m256i*>(x.values()+k));
            const auto dot=_mm256_cvtepi32_pd(Dot4(a,w0+k,w1+k,w2+k,w3+k));
            const auto scale=_mm256_cvtps_pd(_mm_set_ps(s3[b].scale,s2[b].scale,s1[b].scale,s0[b].scale));
            // Exactly I1's dot*weight_scale, then *activation_scale, then +sum.
            // Each lane is one row, and blocks remain chronological. No FMA.
            const auto weighted=_mm256_mul_pd(dot,scale);
            sum=_mm256_add_pd(sum,_mm256_mul_pd(weighted,_mm256_set1_pd(x.scales()[b])));
        }
        const auto magnitude=_mm256_andnot_pd(_mm256_set1_pd(-0.0),sum);
        const auto valid=_mm256_cmp_pd(magnitude,_mm256_set1_pd(std::numeric_limits<float>::max()),_CMP_LE_OQ);
        if (_mm256_movemask_pd(valid)!=15) return Q8_32_GEMVStatus::NUMERIC_ERROR;
        _mm_storeu_ps(out+row,_mm256_cvtpd_ps(sum));
    }
    return Q8_32_GEMVStatus::OK;
}
}

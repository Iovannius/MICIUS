#include "PreparedW8A8Input.h"
#if defined(MICIUS_I13_GEMV4)
#include "I13W8A8GEMV.h"
#endif
#if defined(MICIUS_I2_DIAGNOSTICS)
#include "I2Diagnostic.h"
#endif
#include <algorithm>
#include <cmath>
#include <limits>
#include <immintrin.h>

namespace micius::internal {
namespace {
bool SupportedFP() noexcept { return (_mm_getcsr() & 0xffc0U) == 0x1f80U; }
int32_t Reduce(__m256i x) noexcept {
    __m128i s = _mm_add_epi32(_mm256_castsi256_si128(x), _mm256_extracti128_si256(x,1));
    s = _mm_hadd_epi32(s,s); s = _mm_hadd_epi32(s,s);
    return _mm_cvtsi128_si32(s);
}
template<bool Scalar, bool Checked>
Q8_32_GEMVStatus Consume(const int8_t* weights, const QuantDescriptor* scales,
    const PreparedW8A8Input& x, float* out, uint32_t begin, uint32_t end) noexcept {
    const uint32_t n=x.size(), blocks=n/32U+(n%32U!=0);
    for (uint32_t row=begin;row<end;) {
        const bool pair=end-row>1U;
        double sums[2]={0.0,0.0};
        for (uint32_t b=0;b<blocks;++b) {
            const uint32_t start=b*32U, count=std::min(32U,n-start);
            int32_t dots[2]={0,0};
            if constexpr (!Scalar) {
                if (count==32U) {
                    const auto a0=_mm256_cvtepi8_epi16(_mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(x.values()+start)));
                    const auto a1=_mm256_cvtepi8_epi16(_mm_loadu_si128(
                        reinterpret_cast<const __m128i*>(x.values()+start+16U)));
                    for (uint32_t r=0;r<(pair?2U:1U);++r) {
                        const auto* w=weights+static_cast<size_t>(row+r)*n+start;
                        const auto w0=_mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w)));
                        const auto w1=_mm256_cvtepi8_epi16(_mm_loadu_si128(reinterpret_cast<const __m128i*>(w+16)));
                        dots[r]=Reduce(_mm256_add_epi32(_mm256_madd_epi16(a0,w0),_mm256_madd_epi16(a1,w1)));
                    }
                } else {
                    for (uint32_t r=0;r<(pair?2U:1U);++r)
                        for (uint32_t k=0;k<count;++k)
                            dots[r]+=static_cast<int32_t>(weights[static_cast<size_t>(row+r)*n+start+k])*x.values()[start+k];
                }
            } else {
                // Independent integer dot, accumulated in int64 before range proof.
                for (uint32_t r=0;r<(pair?2U:1U);++r) {
                    int64_t dot=0;
                    for (uint32_t k=0;k<count;++k)
                        dot+=static_cast<int64_t>(weights[static_cast<size_t>(row+r)*n+start+k])*x.values()[start+k];
                    dots[r]=static_cast<int32_t>(dot); // |dot| <= 32*128*127.
                }
            }
            for (uint32_t r=0;r<(pair?2U:1U);++r) {
#if defined(MICIUS_I2_DIAGNOSTICS)
                if (micius::i2::dot_sink)
                    micius::i2::dot_sink[static_cast<size_t>(row+r)*micius::i2::dot_stride+b]=dots[r];
#endif
                const auto& s=scales[static_cast<size_t>(row+r)*blocks+b];
                if constexpr (Checked) {
                    if (!std::isfinite(s.scale)||s.zero_point||s.padding[0]||s.padding[1])
                        return Q8_32_GEMVStatus::INVALID_MODEL;
                }
                const double weighted=static_cast<double>(dots[r])*static_cast<double>(s.scale);
                sums[r]+=weighted*x.scales()[b];
            }
        }
        for (uint32_t r=0;r<(pair?2U:1U);++r) {
            if (!std::isfinite(sums[r]) || std::abs(sums[r])>std::numeric_limits<float>::max())
                return Q8_32_GEMVStatus::NUMERIC_ERROR;
            out[row+r]=static_cast<float>(sums[r]);
        }
        row+=pair?2U:1U;
    }
    return Q8_32_GEMVStatus::OK;
}
} // namespace

Q8_32_GEMVStatus PrepareW8A8Input(std::span<const float> input,
    std::span<int8_t> values, std::span<double> scales, PreparedW8A8Input& out) noexcept {
    out.values_=nullptr;out.scales_=nullptr;out.size_=0;
    if (input.empty()||input.size()>std::numeric_limits<uint32_t>::max()||
        values.size()<input.size()||scales.size()<(input.size()+31U)/32U||!SupportedFP())
        return Q8_32_GEMVStatus::INVALID_ARGUMENT;
    for (size_t begin=0;begin<input.size();begin+=32U) {
        const size_t end=std::min(begin+32U,input.size());
        double maximum=0;
        for (size_t k=begin;k<end;++k) {
            if (!std::isfinite(input[k])) return Q8_32_GEMVStatus::NUMERIC_ERROR;
            maximum=std::max(maximum,std::abs(static_cast<double>(input[k])));
        }
        const double scale=maximum/127.0;
        scales[begin/32U]=scale;
        for (size_t k=begin;k<end;++k) {
            const double q=scale==0 ? 0 : std::nearbyint(static_cast<double>(input[k])/scale);
            values[k]=static_cast<int8_t>(std::clamp(q,-127.0,127.0));
        }
    }
    out.values_=values.data();out.scales_=scales.data();out.size_=static_cast<uint32_t>(input.size());
    return Q8_32_GEMVStatus::OK;
}

Q8_32_GEMVStatus GEMV_W8A8Rows(const int8_t* w,const QuantDescriptor* s,
    const PreparedW8A8Input& x,float* y,uint32_t rows,uint32_t begin,uint32_t end,
    bool validated,bool scalar) noexcept {
    if (!w||!s||!y||!x.values()||!x.scales()||!x.size()||!rows||begin>end||end>rows||!SupportedFP())
        return Q8_32_GEMVStatus::INVALID_ARGUMENT;
#if defined(MICIUS_I13_GEMV4)
    if (validated && !scalar && x.size()%32U==0U && end-begin>=4U) {
        const uint32_t bulk_end=begin+(end-begin)/4U*4U;
        const auto status=i13::Consume4Rows(w,s,x,y,begin,bulk_end);
        if (status!=Q8_32_GEMVStatus::OK || bulk_end==end) return status;
        begin=bulk_end; // Original implementation handles the final one to three rows.
    }
#endif
    if (scalar) return validated ? Consume<true,false>(w,s,x,y,begin,end) : Consume<true,true>(w,s,x,y,begin,end);
    return validated ? Consume<false,false>(w,s,x,y,begin,end) : Consume<false,true>(w,s,x,y,begin,end);
}
} // namespace micius::internal

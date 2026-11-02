#pragma once
#include "PreparedW8A8Input.h"
#include <immintrin.h>

namespace micius::internal::i13 {
// Prepared activation codes are in [-127,127]. Weight bytes cover [-128,127].
// abs(w) is an UNSIGNED byte (so abs(-128) is exactly 128); sign(a,w) stays
// representable, including negative weights. Each maddubs pair has magnitude
// at most 2*128*127=32512, strictly below the signed-16 saturation boundary.
// Do not swap the operands/sign transformation: sign(w,a) can overflow at -128.
inline __m256i DotPartials(__m256i a, __m256i w) noexcept {
    const auto pair = _mm256_maddubs_epi16(_mm256_abs_epi8(w), _mm256_sign_epi8(a,w));
    return _mm256_madd_epi16(pair,_mm256_set1_epi16(1));
}
inline __m128i Fold(__m256i x) noexcept {
    return _mm_add_epi32(_mm256_castsi256_si128(x),_mm256_extracti128_si256(x,1));
}
inline __m128i Dot4(__m256i a, const int8_t* w0, const int8_t* w1,
                    const int8_t* w2, const int8_t* w3) noexcept {
    const auto s0=Fold(DotPartials(a,_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w0))));
    const auto s1=Fold(DotPartials(a,_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w1))));
    const auto s2=Fold(DotPartials(a,_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w2))));
    const auto s3=Fold(DotPartials(a,_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w3))));
    return _mm_hadd_epi32(_mm_hadd_epi32(s0,s1),_mm_hadd_epi32(s2,s3));
}
// Private checked-dispatch boundary only: validated immutable descriptors,
// supported MXCSR, nonempty columns divisible by 32, row count divisible by 4.
// No allocation, repacking, new numerical approximation or persistent scratch.
Q8_32_GEMVStatus Consume4Rows(const int8_t*, const QuantDescriptor*,
    const PreparedW8A8Input&,float*,uint32_t begin,uint32_t end) noexcept;
}

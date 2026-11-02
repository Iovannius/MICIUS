#pragma once
#include "AI/I13W8A8GEMV.h"
#include <span>
#include <vector>
#include <limits>

namespace micius::i23 {
using Status=Q8_32_GEMVStatus;
using Input=micius::internal::PreparedW8A8Input;
enum class PackStatus { OK, UNSUPPORTED, INVALID, RESOURCE };

// Owned, four-row packed matrix. Source buffers remain the responsibility of the binder.
// Successful packing owns independent bytes; failed packing preserves *this.
// Public runtime binding and loader ownership are not changed by this library.
class Matrix {
    uint32_t rows_=0,cols_=0;
    std::vector<int8_t> weights_;
    std::vector<float> scales_;
public:
    Matrix()=default;
    Matrix(const Matrix&)=delete;
    Matrix& operator=(const Matrix&)=delete;
    Matrix(Matrix&& other) noexcept {Swap(other);}
    Matrix& operator=(Matrix&& other) noexcept {Matrix next(std::move(other));Swap(next);return *this;}
    uint32_t rows() const noexcept{return rows_;}
    uint32_t cols() const noexcept{return cols_;}
    size_t blocks() const noexcept{return cols_/32U;}
    const int8_t* weights() const noexcept{return weights_.data();}
    const float* scales() const noexcept{return scales_.data();}
    size_t payload() const noexcept{return weights_.size()+4*scales_.size();}
    size_t capacity_bytes() const noexcept{return weights_.capacity()+4*scales_.capacity();}
    PackStatus Pack(std::span<const int8_t>,std::span<const QuantDescriptor>,uint32_t,uint32_t,
                    size_t budget=std::numeric_limits<size_t>::max()) noexcept;
    void Swap(Matrix&) noexcept;
};

// Binds a borrowed expanded-code scratch and original double scales only after
// successful expansion. Same synchronous lifetime rules as PreparedW8A8Input.
class Expanded {
    const int8_t* values_=nullptr;const double* scales_=nullptr;uint32_t cols_=0;
    friend Status Expand(const Input&,std::span<int8_t>,Expanded&) noexcept;
public:
    uint32_t cols() const noexcept{return cols_;}
    const int8_t* values() const noexcept{return values_;}
    const double* scales() const noexcept{return scales_;}
};
Status Expand(const Input&,std::span<int8_t>,Expanded&) noexcept;
// Private full-matrix operation; only nonempty rows%4==0 and columns%32==0.
// Output/scratch/model storage must be non-overlapping, as in I13's private API.
// Other shapes/ranges must remain on the original implementation at dispatch.
Status Consume(const Matrix&,const Expanded&,std::span<float>) noexcept;

inline __m128i Dot(const int8_t* a,const int8_t* w) noexcept {
    using micius::internal::i13::DotPartials;
    const auto d0=DotPartials(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a)),_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w)));
    const auto d1=DotPartials(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a+32)),_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w+32)));
    const auto d2=DotPartials(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a+64)),_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w+64)));
    const auto d3=DotPartials(_mm256_loadu_si256(reinterpret_cast<const __m256i*>(a+96)),_mm256_loadu_si256(reinterpret_cast<const __m256i*>(w+96)));
    const auto x=_mm256_add_epi32(_mm256_add_epi32(d0,d1),_mm256_add_epi32(d2,d3));
    return _mm_add_epi32(_mm256_castsi256_si128(x),_mm256_extracti128_si256(x,1));
}
}

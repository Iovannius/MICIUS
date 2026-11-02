#pragma once
#include "AIComputKrnl.h"
#include <span>

// I1 is an approximate activation-quantized backend, not an exact F64 rewrite.
#if defined(MICIUS_I1_MODE) && MICIUS_I1_MODE && (defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__))
#error "I1 requires strict floating-point validation"
#endif
namespace micius::internal {
class PreparedW8A8Input final {
public:
    PreparedW8A8Input() noexcept = default;
    PreparedW8A8Input(const PreparedW8A8Input&) = delete;
    PreparedW8A8Input& operator=(const PreparedW8A8Input&) = delete;
    const int8_t* values() const noexcept { return values_; }
    const double* scales() const noexcept { return scales_; }
    uint32_t size() const noexcept { return size_; }
private:
    const int8_t* values_ = nullptr;
    const double* scales_ = nullptr;
    uint32_t size_ = 0;
    friend Q8_32_GEMVStatus PrepareW8A8Input(std::span<const float>,
        std::span<int8_t>, std::span<double>, PreparedW8A8Input&) noexcept;
};
// Non-overlapping caller-owned scratch; invalidated on every preparation.
// Lifetime: until the synchronous consumers finish, never across scratch reuse.
// RN-even, masked exceptions, FTZ/DAZ off. Reject other control environments.
Q8_32_GEMVStatus PrepareW8A8Input(std::span<const float>, std::span<int8_t>,
    std::span<double>, PreparedW8A8Input&) noexcept;
// Full-matrix bases; arbitrary in-bounds [begin,end), including odd rows/tails.
// Buffers must not alias. On failure suppress all partial output.
// validated_scales=true is reserved for the private V1 model-owned boundary.
Q8_32_GEMVStatus GEMV_W8A8Rows(const int8_t*, const QuantDescriptor*,
    const PreparedW8A8Input&, float*, uint32_t rows, uint32_t begin, uint32_t end,
    bool validated_scales, bool scalar_oracle) noexcept;
} // namespace micius::internal

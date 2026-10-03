#pragma once

#include "AIComputKrnl.h"
#include <cstddef>
#include <span>

// The prepared-input route still relies on finite checks. This also catches
// unsupported direct compiler invocations outside the CMake configuration.
#if defined(MICIUS_REFERENCE_GEMV_PREPARED_F64) && \
    (defined(__FAST_MATH__) || \
     (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__))
#error "prepared GEMV input requires strict floating-point validation"
#endif
#if defined(MICIUS_REFERENCE_GEMV_PREPARED_F64) && \
    (!defined(MICIUS_REFERENCE_GEMV_AVX2) || \
     !defined(MICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2))
#error "prepared GEMV input requires the AVX2 row-pair backend"
#endif

namespace micius::internal {

// Borrowed, invocation-scoped snapshot. The owner supplies non-overlapping
// scratch, keeps it alive/read-only until ALL row consumers finish, and never
// retains this view across scratch reuse, reset, or model/context destruction.
// Source F32 input may change after preparation: the snapshot is independent.
// No global cache, pointer-address cache, allocator, or worker scheduler here.
class PreparedF64GEMVInput final {
public:
    PreparedF64GEMVInput() noexcept = default;
    PreparedF64GEMVInput(const PreparedF64GEMVInput&) = delete;
    PreparedF64GEMVInput& operator=(const PreparedF64GEMVInput&) = delete;
    const double* data() const noexcept { return values_; }
    uint32_t size() const noexcept { return size_; }
    uint32_t fp_control() const noexcept { return fp_control_; }

private:
    const double* values_ = nullptr;
    uint32_t size_ = 0;
    uint32_t fp_control_ = 0;
    friend Q8_32_GEMVStatus PrepareF64GEMVInput(
        std::span<const float>, std::span<double>, PreparedF64GEMVInput&) noexcept;
};

// Validates finite inputs and converts F32 to F64 under the caller's MXCSR.
// On failure the view is invalidated; scratch may have partial preparation.
// Consumers must retain the same FP control bits (sticky flags may differ).
Q8_32_GEMVStatus PrepareF64GEMVInput(
    std::span<const float> input, std::span<double> scratch,
    PreparedF64GEMVInput& prepared) noexcept;

// All matrix/output pointers are full-matrix bases, never pre-offset by callers.
// Row range is global [row_begin, row_end). Nonempty ranges start on an even
// row and end on an even row or the matrix end; only the final task owns an odd
// last row. Empty in-bounds ranges are no-ops. Outputs must not alias weights,
// scales or prepared scratch. Each row has one owner. Failure may leave earlier
// rows written; the executor must drain ALL tasks and suppress partial output.
Q8_32_GEMVStatus GEMV_Q8_32_PreparedF64Rows(
    const int8_t* weights, const QuantDescriptor* scales,
    const PreparedF64GEMVInput& input, float* output, uint32_t total_rows,
    uint32_t row_begin, uint32_t row_end) noexcept;

// Same private V1 model-owned scale trust boundary as the existing prevalidated
// entries. Call only for descriptors belonging to that validated model instance.
Q8_32_GEMVStatus GEMV_Q8_32_PreparedF64RowsPrevalidatedScales(
    const int8_t* weights, const QuantDescriptor* scales,
    const PreparedF64GEMVInput& input, float* output, uint32_t total_rows,
    uint32_t row_begin, uint32_t row_end) noexcept;

Q8_32_GEMVStatus GEMV_Q8_32_PreparedF64RowsCompactScales(
    const int8_t* weights, const float* scales,
    const PreparedF64GEMVInput& input, float* output, uint32_t total_rows,
    uint32_t row_begin, uint32_t row_end) noexcept;

} // namespace micius::internal

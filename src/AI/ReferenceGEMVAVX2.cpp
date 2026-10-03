// Q8 reference arithmetic: separate F64 multiplies/adds, no contraction.
// This translation unit is compiled with strict FP flags independently of
// legacy draft kernels. Keep the four-chain and final reduction order fixed.
#include "AIComputKrnlInternal.h"
#include "PreparedGEMVInput.h"
#include <immintrin.h>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error "reference GEMV requires finite checks and strict floating-point compilation"
#endif
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif

namespace {

inline float Q8ScaleValue(const QuantDescriptor* scales, size_t index) noexcept
{
    return scales[index].scale;
}

inline float Q8ScaleValue(const float* scales, size_t index) noexcept
{
    return scales[index];
}

inline __m256d LoadInputF64(const float* input) noexcept
{
    return _mm256_cvtps_pd(_mm_loadu_ps(input));
}

inline __m256d LoadInputF64(const double* input) noexcept
{
    return _mm256_loadu_pd(input);
}

template <typename Input>
inline __m256d AccumulateQ8x4(
    __m256d accumulator,
    const Input* input,
    __m128i packed_weights,
    __m256d scale) noexcept
{
    const __m256d input_pd = LoadInputF64(input);
    const __m128i weights_i32 = _mm_cvtepi8_epi32(packed_weights);
    const __m256d weights_pd = _mm256_cvtepi32_pd(weights_i32);
    const __m256d products = _mm256_mul_pd(input_pd, weights_pd);
    return _mm256_add_pd(accumulator, _mm256_mul_pd(products, scale));
}

inline __m256d AccumulatePreparedQ8x4(
    __m256d accumulator,
    __m256d input_pd,
    __m128i packed_weights,
    __m256d scale) noexcept
{
    const __m128i weights_i32 = _mm_cvtepi8_epi32(packed_weights);
    const __m256d weights_pd = _mm256_cvtepi32_pd(weights_i32);
    const __m256d products = _mm256_mul_pd(input_pd, weights_pd);
    return _mm256_add_pd(accumulator, _mm256_mul_pd(products, scale));
}

inline __m128i LoadPackedQ8x4(const int8_t* weights) noexcept
{
    int32_t packed = 0;
    std::memcpy(&packed, weights, sizeof(packed));
    return _mm_cvtsi32_si128(packed);
}

template <typename Input>
inline void AccumulateRowPairQ8x4(
    __m256d& accumulator0,
    __m256d& accumulator1,
    const Input* input,
    const int8_t* weights0,
    const int8_t* weights1,
    __m256d scale0,
    __m256d scale1) noexcept
{
    const __m256d input_pd = LoadInputF64(input);
    accumulator0 = AccumulatePreparedQ8x4(
        accumulator0, input_pd, LoadPackedQ8x4(weights0), scale0);
    accumulator1 = AccumulatePreparedQ8x4(
        accumulator1, input_pd, LoadPackedQ8x4(weights1), scale1);
}

inline double ReduceAddPD(__m256d values) noexcept
{
    const __m128d low = _mm256_castpd256_pd128(values);
    const __m128d high = _mm256_extractf128_pd(values, 1);
    const __m128d pair_sum = _mm_add_pd(low, high);
    const __m128d upper = _mm_unpackhi_pd(pair_sum, pair_sum);
    return _mm_cvtsd_f64(_mm_add_sd(pair_sum, upper));
}

} // namespace

namespace {

template <bool ValidateScales, bool ValidateInput = true,
          typename ScaleStorage, typename Input>
Q8_32_GEMVStatus GEMV_Q8_32_F32_AVX2_Impl(
    const int8_t* quant_weights,
    const ScaleStorage* scales,
    const Input* input,
    float* output,
    uint32_t input_dim,
    uint32_t output_dim) noexcept
{
    if (!quant_weights || !scales || !input || !output ||
        input_dim == 0 || output_dim == 0) {
        return Q8_32_GEMVStatus::INVALID_ARGUMENT;
    }

    const size_t input_size = static_cast<size_t>(input_dim);
    const size_t output_size = static_cast<size_t>(output_dim);
    const uint32_t blocks_per_row =
        input_dim / 32U + (input_dim % 32U != 0U ? 1U : 0U);
    const size_t block_count = static_cast<size_t>(blocks_per_row);
    if (output_size > std::numeric_limits<size_t>::max() / input_size ||
        output_size > std::numeric_limits<size_t>::max() / block_count) {
        return Q8_32_GEMVStatus::INVALID_ARGUMENT;
    }

    // The scalar reference checks the same activation once for every output
    // row. The optimized path validates it once before entering the row loop.
    if constexpr (ValidateInput) {
        for (uint32_t column = 0; column < input_dim; ++column) {
            if (!std::isfinite(input[column])) {
                return Q8_32_GEMVStatus::NUMERIC_ERROR;
            }
        }
    }

    const uint32_t full_blocks = input_dim / 32U;
    const uint32_t tail_columns = input_dim % 32U;

    for (uint32_t row = 0; row < output_dim; ++row) {
        const size_t weight_row = static_cast<size_t>(row) * input_size;
        const size_t scale_row = static_cast<size_t>(row) * block_count;
        __m256d accumulator0 = _mm256_setzero_pd();
        __m256d accumulator1 = _mm256_setzero_pd();
        __m256d accumulator2 = _mm256_setzero_pd();
        __m256d accumulator3 = _mm256_setzero_pd();

        for (uint32_t block = 0; block < full_blocks; ++block) {
            if constexpr (ValidateScales) {
                const QuantDescriptor& descriptor = scales[scale_row + block];
                if (descriptor.zero_point != 0 || !std::isfinite(descriptor.scale)) {
                    return Q8_32_GEMVStatus::INVALID_MODEL;
                }
            }

            const size_t column = static_cast<size_t>(block) * 32U;
            const int8_t* weight_block = quant_weights + weight_row + column;
            const Input* input_block = input + column;
            const __m256d scale = _mm256_set1_pd(static_cast<double>(
                Q8ScaleValue(scales, scale_row + block)));
            const __m128i weights_low = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(weight_block));
            const __m128i weights_high = _mm_loadu_si128(
                reinterpret_cast<const __m128i*>(weight_block + 16));

            accumulator0 = AccumulateQ8x4(
                accumulator0, input_block, weights_low, scale);
            accumulator1 = AccumulateQ8x4(
                accumulator1, input_block + 4, _mm_srli_si128(weights_low, 4), scale);
            accumulator2 = AccumulateQ8x4(
                accumulator2, input_block + 8, _mm_srli_si128(weights_low, 8), scale);
            accumulator3 = AccumulateQ8x4(
                accumulator3, input_block + 12, _mm_srli_si128(weights_low, 12), scale);
            accumulator0 = AccumulateQ8x4(
                accumulator0, input_block + 16, weights_high, scale);
            accumulator1 = AccumulateQ8x4(
                accumulator1, input_block + 20, _mm_srli_si128(weights_high, 4), scale);
            accumulator2 = AccumulateQ8x4(
                accumulator2, input_block + 24, _mm_srli_si128(weights_high, 8), scale);
            accumulator3 = AccumulateQ8x4(
                accumulator3, input_block + 28, _mm_srli_si128(weights_high, 12), scale);
        }

        const __m256d vector_sum = _mm256_add_pd(
            _mm256_add_pd(accumulator0, accumulator1),
            _mm256_add_pd(accumulator2, accumulator3));
        double sum = ReduceAddPD(vector_sum);

        if (tail_columns != 0U) {
            const uint32_t block = full_blocks;
            if constexpr (ValidateScales) {
                const QuantDescriptor& descriptor = scales[scale_row + block];
                if (descriptor.zero_point != 0 || !std::isfinite(descriptor.scale)) {
                    return Q8_32_GEMVStatus::INVALID_MODEL;
                }
            }
            const uint32_t begin = block * 32U;
            const double scale = static_cast<double>(
                Q8ScaleValue(scales, scale_row + block));
            for (uint32_t column = begin; column < input_dim; ++column) {
                sum += static_cast<double>(input[column]) *
                       static_cast<double>(quant_weights[weight_row + column]) *
                       scale;
            }
        }

        if (!std::isfinite(sum) ||
            sum > static_cast<double>(std::numeric_limits<float>::max()) ||
            sum < -static_cast<double>(std::numeric_limits<float>::max())) {
            return Q8_32_GEMVStatus::NUMERIC_ERROR;
        }
        output[row] = static_cast<float>(sum);
    }

    return Q8_32_GEMVStatus::OK;
}

// Two rows share each F32-to-F64 activation conversion while retaining four
// independent F64 accumulation chains per row. A four-row tile would require
// all 16 architectural YMM registers for accumulators alone and would force
// scale/weight/input temporaries to spill on AVX2 targets.
template <bool ValidateScales, bool ValidateInput = true,
          typename ScaleStorage, typename Input>
Q8_32_GEMVStatus GEMV_Q8_32_F32_AVX2_RowPairImpl(
    const int8_t* quant_weights,
    const ScaleStorage* scales,
    const Input* input,
    float* output,
    uint32_t input_dim,
    uint32_t output_dim) noexcept
{
    if (!quant_weights || !scales || !input || !output ||
        input_dim == 0 || output_dim == 0) {
        return Q8_32_GEMVStatus::INVALID_ARGUMENT;
    }

    const size_t input_size = static_cast<size_t>(input_dim);
    const size_t output_size = static_cast<size_t>(output_dim);
    const uint32_t blocks_per_row =
        input_dim / 32U + (input_dim % 32U != 0U ? 1U : 0U);
    const size_t block_count = static_cast<size_t>(blocks_per_row);
    if (output_size > std::numeric_limits<size_t>::max() / input_size ||
        output_size > std::numeric_limits<size_t>::max() / block_count) {
        return Q8_32_GEMVStatus::INVALID_ARGUMENT;
    }

    if constexpr (ValidateInput) {
        for (uint32_t column = 0; column < input_dim; ++column) {
            if (!std::isfinite(input[column])) {
                return Q8_32_GEMVStatus::NUMERIC_ERROR;
            }
        }
    }

    const uint32_t full_blocks = input_dim / 32U;
    const uint32_t tail_columns = input_dim % 32U;
    uint32_t row = 0;

    for (; output_dim - row >= 2U; row += 2U) {
        const size_t weight_row0 = static_cast<size_t>(row) * input_size;
        const size_t weight_row1 = weight_row0 + input_size;
        const size_t scale_row0 = static_cast<size_t>(row) * block_count;
        const size_t scale_row1 = scale_row0 + block_count;

        __m256d accumulator00 = _mm256_setzero_pd();
        __m256d accumulator01 = _mm256_setzero_pd();
        __m256d accumulator02 = _mm256_setzero_pd();
        __m256d accumulator03 = _mm256_setzero_pd();
        __m256d accumulator10 = _mm256_setzero_pd();
        __m256d accumulator11 = _mm256_setzero_pd();
        __m256d accumulator12 = _mm256_setzero_pd();
        __m256d accumulator13 = _mm256_setzero_pd();

        for (uint32_t block = 0; block < full_blocks; ++block) {
            if constexpr (ValidateScales) {
                const QuantDescriptor& descriptor0 = scales[scale_row0 + block];
                const QuantDescriptor& descriptor1 = scales[scale_row1 + block];
                if (descriptor0.zero_point != 0 ||
                    !std::isfinite(descriptor0.scale) ||
                    descriptor1.zero_point != 0 ||
                    !std::isfinite(descriptor1.scale)) {
                    return Q8_32_GEMVStatus::INVALID_MODEL;
                }
            }

            const size_t column = static_cast<size_t>(block) * 32U;
            const Input* input_block = input + column;
            const int8_t* weight_block0 = quant_weights + weight_row0 + column;
            const int8_t* weight_block1 = quant_weights + weight_row1 + column;
            const __m256d scale0 =
                _mm256_set1_pd(static_cast<double>(
                    Q8ScaleValue(scales, scale_row0 + block)));
            const __m256d scale1 =
                _mm256_set1_pd(static_cast<double>(
                    Q8ScaleValue(scales, scale_row1 + block)));
            // Four-byte weight loads keep no shifted 16-byte block live across
            // calls. They preserve total bytes read, replace shuffle uops with
            // directly addressable loads, and leave all eight accumulators in
            // registers on a 16-register AVX2 target.
            AccumulateRowPairQ8x4(
                accumulator00, accumulator10, input_block,
                weight_block0, weight_block1, scale0, scale1);
            AccumulateRowPairQ8x4(
                accumulator01, accumulator11, input_block + 4,
                weight_block0 + 4, weight_block1 + 4, scale0, scale1);
            AccumulateRowPairQ8x4(
                accumulator02, accumulator12, input_block + 8,
                weight_block0 + 8, weight_block1 + 8, scale0, scale1);
            AccumulateRowPairQ8x4(
                accumulator03, accumulator13, input_block + 12,
                weight_block0 + 12, weight_block1 + 12, scale0, scale1);
            AccumulateRowPairQ8x4(
                accumulator00, accumulator10, input_block + 16,
                weight_block0 + 16, weight_block1 + 16, scale0, scale1);
            AccumulateRowPairQ8x4(
                accumulator01, accumulator11, input_block + 20,
                weight_block0 + 20, weight_block1 + 20, scale0, scale1);
            AccumulateRowPairQ8x4(
                accumulator02, accumulator12, input_block + 24,
                weight_block0 + 24, weight_block1 + 24, scale0, scale1);
            AccumulateRowPairQ8x4(
                accumulator03, accumulator13, input_block + 28,
                weight_block0 + 28, weight_block1 + 28, scale0, scale1);
        }

        const __m256d vector_sum0 = _mm256_add_pd(
            _mm256_add_pd(accumulator00, accumulator01),
            _mm256_add_pd(accumulator02, accumulator03));
        const __m256d vector_sum1 = _mm256_add_pd(
            _mm256_add_pd(accumulator10, accumulator11),
            _mm256_add_pd(accumulator12, accumulator13));
        double sum0 = ReduceAddPD(vector_sum0);
        double sum1 = ReduceAddPD(vector_sum1);

        if (tail_columns != 0U) {
            const uint32_t block = full_blocks;
            if constexpr (ValidateScales) {
                const QuantDescriptor& descriptor0 = scales[scale_row0 + block];
                const QuantDescriptor& descriptor1 = scales[scale_row1 + block];
                if (descriptor0.zero_point != 0 ||
                    !std::isfinite(descriptor0.scale) ||
                    descriptor1.zero_point != 0 ||
                    !std::isfinite(descriptor1.scale)) {
                    return Q8_32_GEMVStatus::INVALID_MODEL;
                }
            }

            const uint32_t begin = block * 32U;
            const double scale0 = static_cast<double>(
                Q8ScaleValue(scales, scale_row0 + block));
            const double scale1 = static_cast<double>(
                Q8ScaleValue(scales, scale_row1 + block));
            for (uint32_t column = begin; column < input_dim; ++column) {
                sum0 += static_cast<double>(input[column]) *
                        static_cast<double>(quant_weights[weight_row0 + column]) *
                        scale0;
            }
            for (uint32_t column = begin; column < input_dim; ++column) {
                sum1 += static_cast<double>(input[column]) *
                        static_cast<double>(quant_weights[weight_row1 + column]) *
                        scale1;
            }
        }

        if (!std::isfinite(sum0) ||
            sum0 > static_cast<double>(std::numeric_limits<float>::max()) ||
            sum0 < -static_cast<double>(std::numeric_limits<float>::max())) {
            return Q8_32_GEMVStatus::NUMERIC_ERROR;
        }
        output[row] = static_cast<float>(sum0);

        if (!std::isfinite(sum1) ||
            sum1 > static_cast<double>(std::numeric_limits<float>::max()) ||
            sum1 < -static_cast<double>(std::numeric_limits<float>::max())) {
            return Q8_32_GEMVStatus::NUMERIC_ERROR;
        }
        output[row + 1U] = static_cast<float>(sum1);
    }

    if (row < output_dim) {
        return GEMV_Q8_32_F32_AVX2_Impl<ValidateScales, ValidateInput>(
            quant_weights + static_cast<size_t>(row) * input_size,
            scales + static_cast<size_t>(row) * block_count,
            input,
            output + row,
            input_dim,
            1U);
    }

    return Q8_32_GEMVStatus::OK;
}

} // namespace

Q8_32_GEMVStatus GEMV_Q8_32_F32_AVX2(
    const int8_t* quant_weights,
    const QuantDescriptor* scales,
    const float* input,
    float* output,
    uint32_t input_dim,
    uint32_t output_dim) noexcept
{
    return GEMV_Q8_32_F32_AVX2_Impl<true>(
        quant_weights, scales, input, output, input_dim, output_dim);
}

Q8_32_GEMVStatus micius::internal::GEMV_Q8_32_F32_AVX2_PrevalidatedScales(
    const int8_t* quant_weights,
    const QuantDescriptor* scales,
    const float* input,
    float* output,
    uint32_t input_dim,
    uint32_t output_dim) noexcept
{
    return GEMV_Q8_32_F32_AVX2_Impl<false>(
        quant_weights, scales, input, output, input_dim, output_dim);
}

Q8_32_GEMVStatus micius::internal::GEMV_Q8_32_F32_AVX2_RowPair(
    const int8_t* quant_weights,
    const QuantDescriptor* scales,
    const float* input,
    float* output,
    uint32_t input_dim,
    uint32_t output_dim) noexcept
{
    return GEMV_Q8_32_F32_AVX2_RowPairImpl<true>(
        quant_weights, scales, input, output, input_dim, output_dim);
}

Q8_32_GEMVStatus
micius::internal::GEMV_Q8_32_F32_AVX2_RowPairPrevalidatedScales(
    const int8_t* quant_weights,
    const QuantDescriptor* scales,
    const float* input,
    float* output,
    uint32_t input_dim,
    uint32_t output_dim) noexcept
{
    return GEMV_Q8_32_F32_AVX2_RowPairImpl<false>(
        quant_weights, scales, input, output, input_dim, output_dim);
}

Q8_32_GEMVStatus micius::internal::GEMV_Q8_32_F32_AVX2_CompactScales(
    const int8_t* quant_weights,
    const float* scales,
    const float* input,
    float* output,
    uint32_t input_dim,
    uint32_t output_dim) noexcept
{
    return GEMV_Q8_32_F32_AVX2_Impl<false>(
        quant_weights, scales, input, output, input_dim, output_dim);
}

Q8_32_GEMVStatus
micius::internal::GEMV_Q8_32_F32_AVX2_RowPairCompactScales(
    const int8_t* quant_weights,
    const float* scales,
    const float* input,
    float* output,
    uint32_t input_dim,
    uint32_t output_dim) noexcept
{
    return GEMV_Q8_32_F32_AVX2_RowPairImpl<false>(
        quant_weights, scales, input, output, input_dim, output_dim);
}

namespace {
constexpr uint32_t kFPControlMask = 0xffc0U; // excludes sticky exception flags

struct AddressRange { uintptr_t begin; uintptr_t end; };

bool GetAddressRange(const void* pointer, size_t count, size_t element_size,
                     AddressRange& range) noexcept
{
    if (!pointer || count > std::numeric_limits<size_t>::max() / element_size)
        return false;
    const size_t bytes = count * element_size;
    const uintptr_t begin = reinterpret_cast<uintptr_t>(pointer);
    if (bytes > std::numeric_limits<uintptr_t>::max() - begin) return false;
    range = {begin, begin + bytes};
    return true;
}

bool Overlaps(AddressRange a, AddressRange b) noexcept
{
    return a.begin < b.end && b.begin < a.end;
}

template <bool ValidateScales, typename ScaleStorage>
Q8_32_GEMVStatus PreparedRows(
    const int8_t* weights, const ScaleStorage* scales,
    const micius::internal::PreparedF64GEMVInput& input, float* output,
    uint32_t total_rows, uint32_t row_begin, uint32_t row_end) noexcept
{
    if (!input.data() || input.size() == 0U || row_begin > row_end ||
        row_end > total_rows ||
        input.fp_control() != (_mm_getcsr() & kFPControlMask))
        return Q8_32_GEMVStatus::INVALID_ARGUMENT;
    if (row_begin == row_end) return Q8_32_GEMVStatus::OK;
    if ((row_begin % 2U) != 0U ||
        ((row_end % 2U) != 0U && row_end != total_rows))
        return Q8_32_GEMVStatus::INVALID_ARGUMENT;

    const size_t cols = input.size();
    const size_t blocks = cols / 32U + (cols % 32U != 0U ? 1U : 0U);
    if (total_rows > std::numeric_limits<size_t>::max() / cols ||
        total_rows > std::numeric_limits<size_t>::max() / blocks)
        return Q8_32_GEMVStatus::INVALID_ARGUMENT;
    AddressRange w{}, s{}, x{}, y{};
    if (!GetAddressRange(weights, total_rows * cols, sizeof(int8_t), w) ||
        !GetAddressRange(scales, total_rows * blocks, sizeof(ScaleStorage), s) ||
        !GetAddressRange(input.data(), cols, sizeof(double), x) ||
        !GetAddressRange(output, total_rows, sizeof(float), y))
        return Q8_32_GEMVStatus::INVALID_ARGUMENT;
    // Check the actual written range against all read-only operands. Adjacent
    // tasks are allowed to own disjoint slices of the same output allocation.
    y.begin += static_cast<size_t>(row_begin) * sizeof(float);
    y.end -= static_cast<size_t>(total_rows - row_end) * sizeof(float);
    if (Overlaps(y, w) || Overlaps(y, s) || Overlaps(y, x))
        return Q8_32_GEMVStatus::INVALID_ARGUMENT;
    return GEMV_Q8_32_F32_AVX2_RowPairImpl<ValidateScales, false>(
        weights + static_cast<size_t>(row_begin) * cols,
        scales + static_cast<size_t>(row_begin) * blocks,
        input.data(), output + row_begin, input.size(), row_end - row_begin);
}
} // namespace

Q8_32_GEMVStatus micius::internal::PrepareF64GEMVInput(
    std::span<const float> input, std::span<double> scratch,
    PreparedF64GEMVInput& prepared) noexcept
{
    prepared.values_ = nullptr;
    prepared.size_ = 0;
    prepared.fp_control_ = 0;
    AddressRange x{}, temporary{};
    if (input.empty() || input.size() > std::numeric_limits<uint32_t>::max() ||
        scratch.size() < input.size() ||
        !GetAddressRange(input.data(), input.size(), sizeof(float), x) ||
        !GetAddressRange(scratch.data(), input.size(), sizeof(double), temporary) ||
        Overlaps(x, temporary))
        return Q8_32_GEMVStatus::INVALID_ARGUMENT;
    for (size_t i = 0; i < input.size(); ++i) {
        if (!std::isfinite(input[i])) return Q8_32_GEMVStatus::NUMERIC_ERROR;
        scratch[i] = static_cast<double>(input[i]);
    }
    prepared.values_ = scratch.data();
    prepared.size_ = static_cast<uint32_t>(input.size());
    prepared.fp_control_ = _mm_getcsr() & kFPControlMask;
    return Q8_32_GEMVStatus::OK;
}

Q8_32_GEMVStatus micius::internal::GEMV_Q8_32_PreparedF64Rows(
    const int8_t* weights, const QuantDescriptor* scales,
    const PreparedF64GEMVInput& input, float* output, uint32_t total_rows,
    uint32_t row_begin, uint32_t row_end) noexcept
{
    return PreparedRows<true>(weights, scales, input, output,
                              total_rows, row_begin, row_end);
}

Q8_32_GEMVStatus micius::internal::GEMV_Q8_32_PreparedF64RowsPrevalidatedScales(
    const int8_t* weights, const QuantDescriptor* scales,
    const PreparedF64GEMVInput& input, float* output, uint32_t total_rows,
    uint32_t row_begin, uint32_t row_end) noexcept
{
    return PreparedRows<false>(weights, scales, input, output,
                               total_rows, row_begin, row_end);
}

Q8_32_GEMVStatus micius::internal::GEMV_Q8_32_PreparedF64RowsCompactScales(
    const int8_t* weights, const float* scales,
    const PreparedF64GEMVInput& input, float* output, uint32_t total_rows,
    uint32_t row_begin, uint32_t row_end) noexcept
{
    return PreparedRows<false>(weights, scales, input, output,
                               total_rows, row_begin, row_end);
}

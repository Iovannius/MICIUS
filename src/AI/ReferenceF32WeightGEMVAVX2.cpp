// PW2: derived arithmetic from r1, with raw integer-valued float weights, converted exactly to double on load.
// Q8 runtime and frozen oracle remain in their original, unmodified files.
#include "PreparedGEMVWeights.h"
#include "PreparedGEMVInput.h"
#include <immintrin.h>
#include <cmath>
#include <cstdint>
#include <limits>
#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error "PW2 requires strict floating point"
#endif
#if defined(__clang__)
#pragma clang fp contract(off)
#pragma clang fp reassociate(off)
#elif defined(__GNUC__)
#pragma GCC optimize ("fp-contract=off")
#endif
namespace {
inline float Q8ScaleValue(const QuantDescriptor* s, size_t i) noexcept { return s[i].scale; }
inline __m256d AccumulateF32Weightx4(__m256d acc, const double* x, const float* w, __m256d s) noexcept {
    const __m256d product = _mm256_mul_pd(_mm256_loadu_pd(x), _mm256_cvtps_pd(_mm_loadu_ps(w)));
    return _mm256_add_pd(acc, _mm256_mul_pd(product, s));
}
inline void AccumulateRowPairF32Weightx4(__m256d& a, __m256d& b, const double* x,
    const float* w0, const float* w1, __m256d s0, __m256d s1) noexcept {
    const __m256d input = _mm256_loadu_pd(x);
    const __m256d p0 = _mm256_mul_pd(input, _mm256_cvtps_pd(_mm_loadu_ps(w0)));
    a = _mm256_add_pd(a, _mm256_mul_pd(p0, s0));
    const __m256d p1 = _mm256_mul_pd(input, _mm256_cvtps_pd(_mm_loadu_ps(w1)));
    b = _mm256_add_pd(b, _mm256_mul_pd(p1, s1));
}
inline double ReduceAddPD(__m256d values) noexcept {
    const __m128d low = _mm256_castpd256_pd128(values);
    const __m128d high = _mm256_extractf128_pd(values, 1);
    const __m128d pair_sum = _mm_add_pd(low, high);
    const __m128d upper = _mm_unpackhi_pd(pair_sum, pair_sum);
    return _mm_cvtsd_f64(_mm_add_sd(pair_sum, upper));
}
template <bool ValidateScales, bool ValidateInput = true,
          typename ScaleStorage, typename Input>
Q8_32_GEMVStatus F32WeightSingle(
    const float* quant_weights,
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
            const float* weight_block = quant_weights + weight_row + column;
            const Input* input_block = input + column;
            const __m256d scale = _mm256_set1_pd(static_cast<double>(
                Q8ScaleValue(scales, scale_row + block)));
            accumulator0 = AccumulateF32Weightx4(
                accumulator0, input_block + 0, weight_block + 0, scale);
            accumulator1 = AccumulateF32Weightx4(
                accumulator1, input_block + 4, weight_block + 4, scale);
            accumulator2 = AccumulateF32Weightx4(
                accumulator2, input_block + 8, weight_block + 8, scale);
            accumulator3 = AccumulateF32Weightx4(
                accumulator3, input_block + 12, weight_block + 12, scale);
            accumulator0 = AccumulateF32Weightx4(
                accumulator0, input_block + 16, weight_block + 16, scale);
            accumulator1 = AccumulateF32Weightx4(
                accumulator1, input_block + 20, weight_block + 20, scale);
            accumulator2 = AccumulateF32Weightx4(
                accumulator2, input_block + 24, weight_block + 24, scale);
            accumulator3 = AccumulateF32Weightx4(
                accumulator3, input_block + 28, weight_block + 28, scale);
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

template <bool ValidateScales, bool ValidateInput = true,
          typename ScaleStorage, typename Input>
Q8_32_GEMVStatus F32WeightPair(
    const float* quant_weights,
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
            const float* weight_block0 = quant_weights + weight_row0 + column;
            const float* weight_block1 = quant_weights + weight_row1 + column;
            const __m256d scale0 =
                _mm256_set1_pd(static_cast<double>(
                    Q8ScaleValue(scales, scale_row0 + block)));
            const __m256d scale1 =
                _mm256_set1_pd(static_cast<double>(
                    Q8ScaleValue(scales, scale_row1 + block)));
            AccumulateRowPairF32Weightx4(
                accumulator00, accumulator10, input_block,
                weight_block0, weight_block1, scale0, scale1);
            AccumulateRowPairF32Weightx4(
                accumulator01, accumulator11, input_block + 4,
                weight_block0 + 4, weight_block1 + 4, scale0, scale1);
            AccumulateRowPairF32Weightx4(
                accumulator02, accumulator12, input_block + 8,
                weight_block0 + 8, weight_block1 + 8, scale0, scale1);
            AccumulateRowPairF32Weightx4(
                accumulator03, accumulator13, input_block + 12,
                weight_block0 + 12, weight_block1 + 12, scale0, scale1);
            AccumulateRowPairF32Weightx4(
                accumulator00, accumulator10, input_block + 16,
                weight_block0 + 16, weight_block1 + 16, scale0, scale1);
            AccumulateRowPairF32Weightx4(
                accumulator01, accumulator11, input_block + 20,
                weight_block0 + 20, weight_block1 + 20, scale0, scale1);
            AccumulateRowPairF32Weightx4(
                accumulator02, accumulator12, input_block + 24,
                weight_block0 + 24, weight_block1 + 24, scale0, scale1);
            AccumulateRowPairF32Weightx4(
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
        return F32WeightSingle<ValidateScales, ValidateInput>(
            quant_weights + static_cast<size_t>(row) * input_size,
            scales + static_cast<size_t>(row) * block_count,
            input,
            output + row,
            input_dim,
            1U);
    }

    return Q8_32_GEMVStatus::OK;
}
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
    const float* weights, const ScaleStorage* scales,
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
    if (!GetAddressRange(weights, total_rows * cols, sizeof(float), w) ||
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
    return F32WeightPair<ValidateScales, false>(
        weights + static_cast<size_t>(row_begin) * cols,
        scales + static_cast<size_t>(row_begin) * blocks,
        input.data(), output + row_begin, input.size(), row_end - row_begin);
}
} // namespace
namespace micius::internal {
Q8_32_GEMVStatus GEMV_F32WeightsPreparedRows(
    const PreparedF32GEMVWeights& weights, const QuantDescriptor* scales,
    const PreparedF64GEMVInput& input, float* output, uint32_t begin, uint32_t end) noexcept
{
    if (!weights.data() || weights.cols() != input.size())
        return Q8_32_GEMVStatus::INVALID_ARGUMENT;
    return PreparedRows<true>(weights.data(), scales, input, output, weights.rows(), begin, end);
}
Q8_32_GEMVStatus GEMV_F32WeightsPreparedRowsPrevalidatedScales(
    const PreparedF32GEMVWeights& weights, const QuantDescriptor* scales,
    const PreparedF64GEMVInput& input, float* output, uint32_t begin, uint32_t end) noexcept
{
    if (!weights.data() || weights.cols() != input.size())
        return Q8_32_GEMVStatus::INVALID_ARGUMENT;
    return PreparedRows<false>(weights.data(), scales, input, output, weights.rows(), begin, end);
}
} // namespace micius::internal

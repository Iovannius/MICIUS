#include "AI/AIComputKrnlInternal.h"
#include "AI/AICompactScaleInternal.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <limits>
#include <vector>

namespace {

[[noreturn]] void Fail(const char* message)
{
    std::cerr << "q8_32_gemv_tests: " << message << '\n';
    std::abort();
}

void Require(bool condition, const char* message)
{
    if (!condition) Fail(message);
}

uint32_t NextRandom(uint32_t* state)
{
    *state = *state * 1664525U + 1013904223U;
    return *state;
}

void ScalarQ8_32GEMV(
    const int8_t* weights,
    const QuantDescriptor* scales,
    const float* input,
    float* output,
    uint32_t input_dim,
    uint32_t output_dim)
{
    const uint32_t blocks_per_row =
        input_dim / 32U + (input_dim % 32U != 0U ? 1U : 0U);
    for (uint32_t row = 0; row < output_dim; ++row) {
        double sum = 0.0;
        const size_t weight_row = static_cast<size_t>(row) * input_dim;
        const size_t scale_row = static_cast<size_t>(row) * blocks_per_row;
        for (uint32_t block = 0; block < blocks_per_row; ++block) {
            const uint32_t begin = block * 32U;
            const uint32_t end = std::min(begin + 32U, input_dim);
            const double scale = static_cast<double>(scales[scale_row + block].scale);
            for (uint32_t column = begin; column < end; ++column) {
                sum += static_cast<double>(input[column]) *
                       static_cast<double>(weights[weight_row + column]) *
                       scale;
            }
        }
        output[row] = static_cast<float>(sum);
    }
}

void TestDifferentialShapesAndValues()
{
    constexpr uint32_t kDimensions[] = {
        1U, 7U, 31U, 32U, 33U, 63U, 64U, 65U, 95U, 96U, 127U, 128U, 129U,
        256U, 768U, 4096U
    };
    constexpr uint32_t kOutputDimensions[] = {1U, 2U, 3U, 4U, 5U, 7U, 16U};
    uint32_t random_state = 0x9E3779B9U;
    double maximum_absolute_error = 0.0;
    double maximum_relative_error = 0.0;

    for (uint32_t input_dim : kDimensions) {
        const uint32_t blocks_per_row =
            input_dim / 32U + (input_dim % 32U != 0U ? 1U : 0U);
        for (uint32_t output_dim : kOutputDimensions) {
            const size_t weight_count = static_cast<size_t>(input_dim) * output_dim;
            const size_t scale_count = static_cast<size_t>(blocks_per_row) * output_dim;

            // Offset the base pointers deliberately. Rows with non-multiple-of-32
            // dimensions then exercise both unaligned vector loads and tails.
            std::vector<int8_t> weight_storage(weight_count + 1U);
            std::vector<float> input_storage(static_cast<size_t>(input_dim) + 1U);
            std::vector<float> actual_storage(static_cast<size_t>(output_dim) + 1U, -99.0f);
            std::vector<float> prevalidated_storage(
                static_cast<size_t>(output_dim) + 1U,
                -101.0f);
            std::vector<float> row_pair_storage(
                static_cast<size_t>(output_dim) + 1U,
                -103.0f);
            std::vector<float> row_pair_prevalidated_storage(
                static_cast<size_t>(output_dim) + 1U,
                -105.0f);
            std::vector<float> compact_storage(
                static_cast<size_t>(output_dim) + 1U,
                -107.0f);
            std::vector<float> row_pair_compact_storage(
                static_cast<size_t>(output_dim) + 1U,
                -109.0f);
            int8_t* weights = weight_storage.data() + 1;
            float* input = input_storage.data() + 1;
            float* actual = actual_storage.data() + 1;
            float* prevalidated = prevalidated_storage.data() + 1;
            float* row_pair = row_pair_storage.data() + 1;
            float* row_pair_prevalidated = row_pair_prevalidated_storage.data() + 1;
            float* compact = compact_storage.data() + 1;
            float* row_pair_compact = row_pair_compact_storage.data() + 1;
            std::vector<QuantDescriptor> scales(scale_count);
            std::vector<MiciusQ8Block32Scale> public_scales(scale_count);
            std::vector<float> compact_scales(scale_count);
            std::vector<float> expected(output_dim, 0.0f);

            for (size_t i = 0; i < weight_count; ++i) {
                weights[i] = static_cast<int8_t>(NextRandom(&random_state) >> 24U);
            }
            if (weight_count != 0U) weights[0] = std::numeric_limits<int8_t>::min();

            for (uint32_t i = 0; i < input_dim; ++i) {
                const int32_t centered =
                    static_cast<int32_t>(NextRandom(&random_state) % 2001U) - 1000;
                input[i] = static_cast<float>(centered) / 257.0f;
            }

            for (size_t i = 0; i < scale_count; ++i) {
                float scale = 0.0f;
                switch (i % 7U) {
                    case 0U: scale = 0.0f; break;
                    case 1U: scale = -0.03125f; break;
                    case 2U: scale = 0.0009765625f; break;
                    case 3U: scale = 0.125f; break;
                    case 4U: scale = -0.0078125f; break;
                    case 5U: scale = 0.013f; break;
                    default: scale = -0.021f; break;
                }
                scales[i] = QuantDescriptor(scale, 0);
                public_scales[i] = MiciusQ8Block32Scale{scale, 0, {0, 0}};
            }
            micius::internal::CopyCompactScaleBits(
                public_scales.data(), compact_scales.data(), scale_count);

            ScalarQ8_32GEMV(
                weights, scales.data(), input, expected.data(), input_dim, output_dim);
            Require(
                GEMV_Q8_32_F32_AVX2(
                    weights, scales.data(), input, actual, input_dim, output_dim) ==
                    Q8_32_GEMVStatus::OK,
                "optimized kernel rejected valid data");
            Require(
                micius::internal::GEMV_Q8_32_F32_AVX2_PrevalidatedScales(
                    weights,
                    scales.data(),
                    input,
                    prevalidated,
                    input_dim,
                    output_dim) == Q8_32_GEMVStatus::OK,
                "prevalidated kernel rejected valid data");
            Require(
                micius::internal::GEMV_Q8_32_F32_AVX2_RowPair(
                    weights,
                    scales.data(),
                    input,
                    row_pair,
                    input_dim,
                    output_dim) == Q8_32_GEMVStatus::OK,
                "row-pair checked kernel rejected valid data");
            Require(
                micius::internal::GEMV_Q8_32_F32_AVX2_RowPairPrevalidatedScales(
                    weights,
                    scales.data(),
                    input,
                    row_pair_prevalidated,
                    input_dim,
                    output_dim) == Q8_32_GEMVStatus::OK,
                "row-pair prevalidated kernel rejected valid data");
            Require(
                micius::internal::GEMV_Q8_32_F32_AVX2_CompactScales(
                    weights,
                    compact_scales.data(),
                    input,
                    compact,
                    input_dim,
                    output_dim) == Q8_32_GEMVStatus::OK,
                "compact-scale kernel rejected valid data");
            Require(
                micius::internal::GEMV_Q8_32_F32_AVX2_RowPairCompactScales(
                    weights,
                    compact_scales.data(),
                    input,
                    row_pair_compact,
                    input_dim,
                    output_dim) == Q8_32_GEMVStatus::OK,
                "row-pair compact-scale kernel rejected valid data");

            for (uint32_t row = 0; row < output_dim; ++row) {
                Require(
                    std::bit_cast<uint32_t>(actual[row]) ==
                        std::bit_cast<uint32_t>(prevalidated[row]),
                    "checked and prevalidated kernels differ bitwise");
                Require(
                    std::bit_cast<uint32_t>(actual[row]) ==
                        std::bit_cast<uint32_t>(row_pair[row]),
                    "baseline and row-pair checked kernels differ bitwise");
                Require(
                    std::bit_cast<uint32_t>(actual[row]) ==
                        std::bit_cast<uint32_t>(row_pair_prevalidated[row]),
                    "baseline and row-pair prevalidated kernels differ bitwise");
                Require(
                    std::bit_cast<uint32_t>(actual[row]) ==
                        std::bit_cast<uint32_t>(compact[row]),
                    "baseline and compact-scale kernels differ bitwise");
                Require(
                    std::bit_cast<uint32_t>(actual[row]) ==
                        std::bit_cast<uint32_t>(row_pair_compact[row]),
                    "baseline and row-pair compact-scale kernels differ bitwise");
                const double error = std::abs(
                    static_cast<double>(actual[row]) - static_cast<double>(expected[row]));
                const double relative_error = error /
                    std::max(1.0, std::abs(static_cast<double>(expected[row])));
                maximum_absolute_error = std::max(maximum_absolute_error, error);
                maximum_relative_error = std::max(maximum_relative_error, relative_error);
                if (error > 2.0e-5 *
                    std::max(1.0, std::abs(static_cast<double>(expected[row])))) {
                    Fail("optimized result exceeded the scalar-double tolerance");
                }
            }
            Require(actual_storage[0] == -99.0f, "checked kernel wrote before output");
            Require(
                prevalidated_storage[0] == -101.0f,
                "prevalidated kernel wrote before output");
            Require(
                row_pair_storage[0] == -103.0f,
                "row-pair checked kernel wrote before output");
            Require(
                row_pair_prevalidated_storage[0] == -105.0f,
                "row-pair prevalidated kernel wrote before output");
            Require(compact_storage[0] == -107.0f,
                "compact-scale kernel wrote before output");
            Require(row_pair_compact_storage[0] == -109.0f,
                "row-pair compact-scale kernel wrote before output");
        }
    }

    std::cout << "differential_cases="
              << (sizeof(kDimensions) / sizeof(kDimensions[0])) *
                 (sizeof(kOutputDimensions) / sizeof(kOutputDimensions[0]))
              << " max_abs_error=" << maximum_absolute_error
              << " max_rel_error=" << maximum_relative_error << '\n';
}

void TestCompactScaleBitCopy()
{
    constexpr uint32_t kFinitePatterns[] = {
        0x00000000U,
        0x80000000U,
        0x00000001U,
        0x807FFFFFU,
        0x3EAAAAABU,
        0xBE4CCCCDU,
        0x7F7FFFFFU
    };
    MiciusQ8Block32Scale source[std::size(kFinitePatterns)]{};
    float compact[std::size(kFinitePatterns)]{};
    for (size_t i = 0; i < std::size(kFinitePatterns); ++i) {
        source[i] = MiciusQ8Block32Scale{
            std::bit_cast<float>(kFinitePatterns[i]), 0, {0, 0}};
    }
    micius::internal::CopyCompactScaleBits(
        source, compact, std::size(kFinitePatterns));
    for (size_t i = 0; i < std::size(kFinitePatterns); ++i) {
        Require(
            std::bit_cast<uint32_t>(compact[i]) == kFinitePatterns[i],
            "compact scale copy changed the F32 bit pattern");
    }
}

void TestRowPairSignedExtremesAndOddTail()
{
    constexpr uint32_t kInputDim = 65U;
    constexpr uint32_t kOutputDim = 5U;
    constexpr uint32_t kBlocksPerRow = 3U;
    std::vector<int8_t> weights(
        static_cast<size_t>(kInputDim) * kOutputDim);
    std::vector<float> input(kInputDim);
    std::vector<QuantDescriptor> scales(
        static_cast<size_t>(kBlocksPerRow) * kOutputDim);
    std::vector<float> compact_scales(
        static_cast<size_t>(kBlocksPerRow) * kOutputDim);
    std::vector<float> baseline(kOutputDim, 0.0f);
    std::vector<float> row_pair(kOutputDim, 0.0f);
    std::vector<float> row_pair_compact(kOutputDim, 0.0f);

    constexpr int8_t kWeightPattern[] = {
        std::numeric_limits<int8_t>::min(),
        std::numeric_limits<int8_t>::max(),
        int8_t{-1},
        int8_t{1}
    };
    for (uint32_t row = 0; row < kOutputDim; ++row) {
        for (uint32_t column = 0; column < kInputDim; ++column) {
            weights[static_cast<size_t>(row) * kInputDim + column] =
                kWeightPattern[(row + column) % 4U];
        }
    }
    for (uint32_t column = 0; column < kInputDim; ++column) {
        const int32_t centered = static_cast<int32_t>(column % 17U) - 8;
        input[column] = static_cast<float>(centered) * -0.375f;
    }
    for (size_t index = 0; index < scales.size(); ++index) {
        const float scale = (index % 2U == 0U) ? -0.03125f : 0.015625f;
        scales[index] = QuantDescriptor(scale, 0);
        compact_scales[index] = scale;
    }

    Require(
        GEMV_Q8_32_F32_AVX2(
            weights.data(),
            scales.data(),
            input.data(),
            baseline.data(),
            kInputDim,
            kOutputDim) == Q8_32_GEMVStatus::OK,
        "baseline rejected signed-extreme row-pair case");
    Require(
        micius::internal::GEMV_Q8_32_F32_AVX2_RowPair(
            weights.data(),
            scales.data(),
            input.data(),
            row_pair.data(),
            kInputDim,
            kOutputDim) == Q8_32_GEMVStatus::OK,
        "row-pair kernel rejected signed-extreme odd-tail case");
    Require(
        micius::internal::GEMV_Q8_32_F32_AVX2_RowPairCompactScales(
            weights.data(),
            compact_scales.data(),
            input.data(),
            row_pair_compact.data(),
            kInputDim,
            kOutputDim) == Q8_32_GEMVStatus::OK,
        "compact row-pair kernel rejected signed-extreme odd-tail case");

    for (uint32_t row = 0; row < kOutputDim; ++row) {
        Require(
            std::bit_cast<uint32_t>(baseline[row]) ==
                std::bit_cast<uint32_t>(row_pair[row]),
            "row-pair signed-extreme odd-tail result differs bitwise");
        Require(
            std::bit_cast<uint32_t>(baseline[row]) ==
                std::bit_cast<uint32_t>(row_pair_compact[row]),
            "compact row-pair signed-extreme odd-tail result differs bitwise");
    }
}

void TestRowPairFailureModes()
{
    constexpr uint32_t kInputDim = 32U;
    constexpr uint32_t kOutputDim = 2U;
    std::vector<int8_t> weights(
        static_cast<size_t>(kInputDim) * kOutputDim,
        int8_t{1});
    std::vector<float> input(kInputDim, 1.0f);
    std::vector<float> output(kOutputDim, 123.0f);
    QuantDescriptor scales[kOutputDim] = {
        QuantDescriptor(0.25f, 0),
        QuantDescriptor(0.5f, 0)
    };

    Require(
        micius::internal::GEMV_Q8_32_F32_AVX2_RowPair(
            nullptr,
            scales,
            input.data(),
            output.data(),
            kInputDim,
            kOutputDim) == Q8_32_GEMVStatus::INVALID_ARGUMENT,
        "row-pair kernel accepted null weights");

    input[7] = -std::numeric_limits<float>::infinity();
    Require(
        micius::internal::GEMV_Q8_32_F32_AVX2_RowPair(
            weights.data(),
            scales,
            input.data(),
            output.data(),
            kInputDim,
            kOutputDim) == Q8_32_GEMVStatus::NUMERIC_ERROR,
        "row-pair kernel accepted a non-finite activation");
    input[7] = 1.0f;

    scales[1].zero_point = 1;
    Require(
        micius::internal::GEMV_Q8_32_F32_AVX2_RowPair(
            weights.data(),
            scales,
            input.data(),
            output.data(),
            kInputDim,
            kOutputDim) == Q8_32_GEMVStatus::INVALID_MODEL,
        "row-pair kernel accepted an invalid second-row descriptor");
    scales[1].zero_point = 0;

    scales[0].scale = std::numeric_limits<float>::max();
    scales[1].scale = std::numeric_limits<float>::max();
    std::fill(input.begin(), input.end(), std::numeric_limits<float>::max());
    std::fill(weights.begin(), weights.end(), std::numeric_limits<int8_t>::max());
    Require(
        micius::internal::GEMV_Q8_32_F32_AVX2_RowPairPrevalidatedScales(
            weights.data(),
            scales,
            input.data(),
            output.data(),
            kInputDim,
            kOutputDim) == Q8_32_GEMVStatus::NUMERIC_ERROR,
        "row-pair prevalidated kernel accepted FP32 output overflow");
}

void TestPrevalidatedEntryRetainsRuntimeChecks()
{
    std::vector<int8_t> weights(32U, int8_t{1});
    std::vector<float> input(32U, 1.0f);
    std::vector<float> output(1U, 123.0f);
    QuantDescriptor scale(0.25f, 0);

    Require(
        micius::internal::GEMV_Q8_32_F32_AVX2_PrevalidatedScales(
            nullptr, &scale, input.data(), output.data(), 32U, 1U) ==
            Q8_32_GEMVStatus::INVALID_ARGUMENT,
        "prevalidated entry accepted null weights");

    input[5] = std::numeric_limits<float>::quiet_NaN();
    Require(
        micius::internal::GEMV_Q8_32_F32_AVX2_PrevalidatedScales(
            weights.data(), &scale, input.data(), output.data(), 32U, 1U) ==
            Q8_32_GEMVStatus::NUMERIC_ERROR,
        "prevalidated entry accepted a non-finite activation");
    input[5] = 1.0f;

    scale.scale = std::numeric_limits<float>::max();
    std::fill(input.begin(), input.end(), std::numeric_limits<float>::max());
    std::fill(weights.begin(), weights.end(), std::numeric_limits<int8_t>::max());
    Require(
        micius::internal::GEMV_Q8_32_F32_AVX2_PrevalidatedScales(
            weights.data(), &scale, input.data(), output.data(), 32U, 1U) ==
            Q8_32_GEMVStatus::NUMERIC_ERROR,
        "prevalidated entry accepted FP32 output overflow");
}

void TestCompactEntryRetainsRuntimeChecks()
{
    std::vector<int8_t> weights(32U, int8_t{1});
    std::vector<float> input(32U, 1.0f);
    std::vector<float> output(1U, 123.0f);
    float scale = 0.25f;

    Require(
        micius::internal::GEMV_Q8_32_F32_AVX2_CompactScales(
            nullptr, &scale, input.data(), output.data(), 32U, 1U) ==
            Q8_32_GEMVStatus::INVALID_ARGUMENT,
        "compact-scale entry accepted null weights");

    input[5] = std::numeric_limits<float>::quiet_NaN();
    Require(
        micius::internal::GEMV_Q8_32_F32_AVX2_CompactScales(
            weights.data(), &scale, input.data(), output.data(), 32U, 1U) ==
            Q8_32_GEMVStatus::NUMERIC_ERROR,
        "compact-scale entry accepted a non-finite activation");
    input[5] = 1.0f;

    scale = std::numeric_limits<float>::max();
    std::fill(input.begin(), input.end(), std::numeric_limits<float>::max());
    std::fill(weights.begin(), weights.end(), std::numeric_limits<int8_t>::max());
    Require(
        micius::internal::GEMV_Q8_32_F32_AVX2_CompactScales(
            weights.data(), &scale, input.data(), output.data(), 32U, 1U) ==
            Q8_32_GEMVStatus::NUMERIC_ERROR,
        "compact-scale entry accepted FP32 output overflow");
}

void TestValidationAndFailureModes()
{
    std::vector<int8_t> weights(32U, int8_t{1});
    std::vector<float> input(32U, 1.0f);
    std::vector<float> output(1U, 123.0f);
    QuantDescriptor scale(0.25f, 0);

    Require(
        GEMV_Q8_32_F32_AVX2(
            nullptr, &scale, input.data(), output.data(), 32U, 1U) ==
            Q8_32_GEMVStatus::INVALID_ARGUMENT,
        "null weights were accepted");
    Require(
        GEMV_Q8_32_F32_AVX2(
            weights.data(), &scale, input.data(), output.data(), 0U, 1U) ==
            Q8_32_GEMVStatus::INVALID_ARGUMENT,
        "zero input dimension was accepted");

    input[17] = std::numeric_limits<float>::infinity();
    Require(
        GEMV_Q8_32_F32_AVX2(
            weights.data(), &scale, input.data(), output.data(), 32U, 1U) ==
            Q8_32_GEMVStatus::NUMERIC_ERROR,
        "non-finite activation was accepted");
    input[17] = 1.0f;

    scale.zero_point = 1;
    Require(
        GEMV_Q8_32_F32_AVX2(
            weights.data(), &scale, input.data(), output.data(), 32U, 1U) ==
            Q8_32_GEMVStatus::INVALID_MODEL,
        "non-zero zero point was accepted");
    scale.zero_point = 0;
    scale.scale = std::numeric_limits<float>::quiet_NaN();
    Require(
        GEMV_Q8_32_F32_AVX2(
            weights.data(), &scale, input.data(), output.data(), 32U, 1U) ==
            Q8_32_GEMVStatus::INVALID_MODEL,
        "non-finite scale was accepted");

    scale.scale = std::numeric_limits<float>::max();
    std::fill(input.begin(), input.end(), std::numeric_limits<float>::max());
    std::fill(weights.begin(), weights.end(), std::numeric_limits<int8_t>::max());
    Require(
        GEMV_Q8_32_F32_AVX2(
            weights.data(), &scale, input.data(), output.data(), 32U, 1U) ==
            Q8_32_GEMVStatus::NUMERIC_ERROR,
        "FP32 output overflow was accepted");
}

} // namespace

int main()
{
    TestCompactScaleBitCopy();
    TestDifferentialShapesAndValues();
    TestRowPairSignedExtremesAndOddTail();
    TestRowPairFailureModes();
    TestValidationAndFailureModes();
    TestPrevalidatedEntryRetainsRuntimeChecks();
    TestCompactEntryRetainsRuntimeChecks();
    std::cout << "q8_32 GEMV tests passed\n";
    return 0;
}

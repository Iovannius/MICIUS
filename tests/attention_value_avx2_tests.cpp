#include "AIComputKrnl.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <vector>

namespace {

[[noreturn]] void Fail(const char* message)
{
    std::cerr << "attention_value_avx2_tests: " << message << '\n';
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

void ScalarAttentionValues(
    const float* unnormalized_probabilities,
    double probability_sum,
    const float* value_rows,
    size_t value_stride,
    float* output,
    uint32_t sequence_length,
    uint32_t head_dim)
{
    std::fill_n(output, head_dim, 0.0f);
    for (uint32_t time = 0; time < sequence_length; ++time) {
        const double probability =
            static_cast<double>(unnormalized_probabilities[time]) / probability_sum;
        const float* values = value_rows + static_cast<size_t>(time) * value_stride;
        for (uint32_t dimension = 0; dimension < head_dim; ++dimension) {
            const double accumulated = std::fma(
                probability,
                static_cast<double>(values[dimension]),
                static_cast<double>(output[dimension]));
            output[dimension] = static_cast<float>(accumulated);
        }
    }
}

void TestBitwiseDifferentialShapes()
{
    constexpr uint32_t kSequenceLengths[] = {
        1U, 2U, 3U, 7U, 31U, 64U, 255U, 512U
    };
    constexpr uint32_t kHeadDimensions[] = {
        1U, 3U, 4U, 5U, 7U, 8U, 15U, 31U, 32U, 33U, 64U, 128U
    };
    uint32_t random_state = 0x4154544EU;
    size_t case_count = 0;

    for (uint32_t sequence_length : kSequenceLengths) {
        for (uint32_t head_dim : kHeadDimensions) {
            const size_t value_stride =
                static_cast<size_t>(head_dim) + (case_count % 5U);
            const size_t value_count =
                static_cast<size_t>(sequence_length - 1U) * value_stride + head_dim;

            std::vector<float> probability_storage(
                static_cast<size_t>(sequence_length) + 2U, -7.0f);
            std::vector<float> value_storage(value_count + 2U, -9.0f);
            std::vector<float> expected_storage(static_cast<size_t>(head_dim) + 2U, -11.0f);
            std::vector<float> actual_storage(static_cast<size_t>(head_dim) + 2U, -13.0f);
            float* probabilities = probability_storage.data() + 1;
            float* values = value_storage.data() + 1;
            float* expected = expected_storage.data() + 1;
            float* actual = actual_storage.data() + 1;

            double probability_sum = 0.0;
            for (uint32_t time = 0; time < sequence_length; ++time) {
                probabilities[time] = static_cast<float>(
                    1U + (NextRandom(&random_state) % 4096U)) / 4096.0f;
                probability_sum += static_cast<double>(probabilities[time]);
                float* row = values + static_cast<size_t>(time) * value_stride;
                for (uint32_t dimension = 0; dimension < head_dim; ++dimension) {
                    const int32_t centered =
                        static_cast<int32_t>(NextRandom(&random_state) % 32769U) - 16384;
                    row[dimension] = static_cast<float>(centered) / 2048.0f;
                }
            }

            ScalarAttentionValues(
                probabilities,
                probability_sum,
                values,
                value_stride,
                expected,
                sequence_length,
                head_dim);
            Require(
                AccumulateAttentionValuesF32_AVX2(
                    probabilities,
                    probability_sum,
                    values,
                    value_stride,
                    actual,
                    sequence_length,
                    head_dim) == AttentionValueStatus::OK,
                "optimized accumulator rejected valid data");

            for (uint32_t dimension = 0; dimension < head_dim; ++dimension) {
                if (std::bit_cast<uint32_t>(actual[dimension]) !=
                    std::bit_cast<uint32_t>(expected[dimension])) {
                    Fail("optimized accumulator changed the per-time FP32 result");
                }
            }
            Require(actual_storage.front() == -13.0f, "output prefix sentinel was overwritten");
            Require(actual_storage.back() == -13.0f, "output suffix sentinel was overwritten");
            ++case_count;
        }
    }

    std::cout << "bitwise_differential_cases=" << case_count << '\n';
}

void TestValidationAndNumericFailures()
{
    float probabilities[2] = {1.0f, 1.0f};
    float values[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    float output[4] = {};

    Require(
        AccumulateAttentionValuesF32_AVX2(
            nullptr, 2.0, values, 4U, output, 2U, 4U) ==
            AttentionValueStatus::INVALID_ARGUMENT,
        "null probability array was accepted");
    Require(
        AccumulateAttentionValuesF32_AVX2(
            probabilities, 2.0, values, 3U, output, 2U, 4U) ==
            AttentionValueStatus::INVALID_ARGUMENT,
        "value stride smaller than the head was accepted");
    Require(
        AccumulateAttentionValuesF32_AVX2(
            probabilities, 2.0, values, 4U, output, 0U, 4U) ==
            AttentionValueStatus::INVALID_ARGUMENT,
        "zero sequence length was accepted");
    Require(
        AccumulateAttentionValuesF32_AVX2(
            probabilities, 0.0, values, 4U, output, 2U, 4U) ==
            AttentionValueStatus::NUMERIC_ERROR,
        "zero probability sum was accepted");
    Require(
        AccumulateAttentionValuesF32_AVX2(
            probabilities,
            std::numeric_limits<double>::infinity(),
            values,
            4U,
            output,
            2U,
            4U) == AttentionValueStatus::NUMERIC_ERROR,
        "non-finite probability sum was accepted");

    probabilities[1] = std::numeric_limits<float>::quiet_NaN();
    Require(
        AccumulateAttentionValuesF32_AVX2(
            probabilities, 2.0, values, 4U, output, 2U, 4U) ==
            AttentionValueStatus::NUMERIC_ERROR,
        "non-finite probability was accepted");
    probabilities[1] = 1.0f;

    values[6] = std::numeric_limits<float>::infinity();
    Require(
        AccumulateAttentionValuesF32_AVX2(
            probabilities, 2.0, values, 4U, output, 2U, 4U) ==
            AttentionValueStatus::NUMERIC_ERROR,
        "non-finite value was accepted");
    values[6] = 7.0f;

    std::fill_n(values, 8U, std::numeric_limits<float>::max());
    Require(
        AccumulateAttentionValuesF32_AVX2(
            probabilities, 1.0, values, 4U, output, 2U, 4U) ==
            AttentionValueStatus::NUMERIC_ERROR,
        "intermediate FP32 overflow was accepted");
}

void TestOutOfRangeThatRoundsBackToFinite()
{
    const float maximum = std::numeric_limits<float>::max();
    const float tiny_probability = std::ldexp(1.0f, -52);
    const double just_out_of_range = std::fma(
        static_cast<double>(tiny_probability),
        static_cast<double>(maximum),
        static_cast<double>(maximum));
    Require(
        just_out_of_range > static_cast<double>(maximum),
        "test setup did not exceed the FP32 range");
    Require(
        static_cast<float>(just_out_of_range) == maximum,
        "test setup did not round the out-of-range value back to finite FP32");

    float probabilities[2] = {1.0f, tiny_probability};
    float vector_values[8];
    std::fill_n(vector_values, 8U, maximum);
    float vector_output[4] = {};
    Require(
        AccumulateAttentionValuesF32_AVX2(
            probabilities,
            1.0,
            vector_values,
            4U,
            vector_output,
            2U,
            4U) == AttentionValueStatus::NUMERIC_ERROR,
        "vector path lost an out-of-range value that rounded back to finite FP32");

    float scalar_values[2] = {maximum, maximum};
    float scalar_output[1] = {};
    Require(
        AccumulateAttentionValuesF32_AVX2(
            probabilities,
            1.0,
            scalar_values,
            1U,
            scalar_output,
            2U,
            1U) == AttentionValueStatus::NUMERIC_ERROR,
        "scalar tail lost an out-of-range value that rounded back to finite FP32");
}

} // namespace

int main()
{
    TestBitwiseDifferentialShapes();
    TestValidationAndNumericFailures();
    TestOutOfRangeThatRoundsBackToFinite();
    std::cout << "attention value AVX2 tests passed\n";
    return 0;
}

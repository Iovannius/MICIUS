#include "AI/AIComputKrnlInternal.h"

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
    std::cerr << "attention_score_time_quad_tests: " << message << '\n';
    std::abort();
}

void Require(bool condition, const char* message)
{
    if (!condition) Fail(message);
}

uint32_t NextRandom(uint32_t* state) noexcept
{
    *state = *state * 1664525U + 1013904223U;
    return *state;
}

double OrderedDot(
    const float* query,
    const float* key,
    uint32_t head_dim) noexcept
{
    double dot = 0.0;
    for (uint32_t dimension = 0; dimension < head_dim; ++dimension) {
        dot += static_cast<double>(query[dimension]) * key[dimension];
    }
    return dot;
}

struct ScoreScanResult {
    std::vector<float> scores;
    double maximum = -std::numeric_limits<double>::infinity();
};

ScoreScanResult ScalarScoreScan(
    const float* query,
    const float* keys,
    uint32_t row_count,
    uint32_t head_dim)
{
    ScoreScanResult result;
    result.scores.resize(row_count);
    const double denominator = std::sqrt(static_cast<double>(head_dim));
    for (uint32_t row = 0U; row < row_count; ++row) {
        const double score =
            OrderedDot(
                query,
                keys + static_cast<size_t>(row) * head_dim,
                head_dim) /
            denominator;
        result.scores[row] = static_cast<float>(score);
        result.maximum = std::max(result.maximum, score);
    }
    return result;
}

ScoreScanResult TimeQuadScoreScan(
    const float* query,
    const float* keys,
    uint32_t row_count,
    uint32_t head_dim)
{
    ScoreScanResult result;
    result.scores.resize(row_count);
    const double denominator = std::sqrt(static_cast<double>(head_dim));
    uint32_t row = 0U;
    for (; row + 3U < row_count; row += 4U) {
        const float* key0 = keys + static_cast<size_t>(row) * head_dim;
        const float* key1 = key0 + head_dim;
        const float* key2 = key1 + head_dim;
        const float* key3 = key2 + head_dim;
        double dots[4];
        micius::internal::AttentionScoreDotTimeQuadF32(
            query, key0, key1, key2, key3, head_dim, dots);
        for (uint32_t lane = 0U; lane < 4U; ++lane) {
            const double score = dots[lane] / denominator;
            result.scores[row + lane] = static_cast<float>(score);
            result.maximum = std::max(result.maximum, score);
        }
    }
    for (; row < row_count; ++row) {
        const double score =
            OrderedDot(
                query,
                keys + static_cast<size_t>(row) * head_dim,
                head_dim) /
            denominator;
        result.scores[row] = static_cast<float>(score);
        result.maximum = std::max(result.maximum, score);
    }
    return result;
}

void TestOrderedDotParity()
{
    constexpr uint32_t kHeadDimensions[] = {
        1U, 2U, 3U, 4U, 7U, 8U, 15U, 31U, 32U, 33U, 64U, 127U, 128U
    };
    uint32_t random_state = 0x54494D45U;
    size_t case_count = 0U;

    for (uint32_t head_dim : kHeadDimensions) {
        for (uint32_t sample = 0U; sample < 24U; ++sample) {
            std::vector<float> query_storage(static_cast<size_t>(head_dim) + 2U);
            std::vector<float> key_storage(static_cast<size_t>(head_dim) * 4U + 2U);
            float* query = query_storage.data() + 1;
            float* keys = key_storage.data() + 1;

            for (uint32_t dimension = 0U; dimension < head_dim; ++dimension) {
                const int32_t bits =
                    static_cast<int32_t>(NextRandom(&random_state) % 131073U) -
                    65536;
                query[dimension] = static_cast<float>(bits) / 8192.0f;
            }
            for (uint32_t row = 0U; row < 4U; ++row) {
                for (uint32_t dimension = 0U; dimension < head_dim; ++dimension) {
                    const int32_t bits =
                        static_cast<int32_t>(NextRandom(&random_state) % 131073U) -
                        65536;
                    keys[static_cast<size_t>(row) * head_dim + dimension] =
                        static_cast<float>(bits) / 4096.0f;
                }
            }

            if (head_dim >= 4U && sample == 0U) {
                query[0] = 0.0f;
                query[1] = -0.0f;
                query[2] = std::numeric_limits<float>::denorm_min();
                query[3] = -std::numeric_limits<float>::denorm_min();
            }

            double actual[4];
            micius::internal::AttentionScoreDotTimeQuadF32(
                query,
                keys,
                keys + head_dim,
                keys + 2U * head_dim,
                keys + 3U * head_dim,
                head_dim,
                actual);
            for (uint32_t row = 0U; row < 4U; ++row) {
                const double expected = OrderedDot(
                    query,
                    keys + static_cast<size_t>(row) * head_dim,
                    head_dim);
                Require(
                    std::bit_cast<uint64_t>(actual[row]) ==
                        std::bit_cast<uint64_t>(expected),
                    "time-quad dot changed ordered accumulation bits");
            }
            ++case_count;
        }
    }

    std::cout << "attention_score_time_quad_dot_cases=" << case_count << '\n';
}

void TestChronologicalScanAndTails()
{
    constexpr uint32_t kHeadDimensions[] = {1U, 3U, 8U, 31U, 32U, 64U, 128U};
    uint32_t random_state = 0x51554144U;
    size_t case_count = 0U;

    for (uint32_t head_dim : kHeadDimensions) {
        for (uint32_t row_count = 1U; row_count <= 17U; ++row_count) {
            std::vector<float> query(head_dim);
            std::vector<float> keys(static_cast<size_t>(row_count) * head_dim);
            for (float& value : query) {
                const int32_t bits =
                    static_cast<int32_t>(NextRandom(&random_state) % 65537U) -
                    32768;
                value = static_cast<float>(bits) / 4096.0f;
            }
            for (float& value : keys) {
                const int32_t bits =
                    static_cast<int32_t>(NextRandom(&random_state) % 65537U) -
                    32768;
                value = static_cast<float>(bits) / 2048.0f;
            }

            const ScoreScanResult expected =
                ScalarScoreScan(query.data(), keys.data(), row_count, head_dim);
            const ScoreScanResult actual =
                TimeQuadScoreScan(query.data(), keys.data(), row_count, head_dim);
            Require(
                std::bit_cast<uint64_t>(actual.maximum) ==
                    std::bit_cast<uint64_t>(expected.maximum),
                "time-quad scan changed chronological maximum bits");
            for (uint32_t row = 0U; row < row_count; ++row) {
                Require(
                    std::bit_cast<uint32_t>(actual.scores[row]) ==
                        std::bit_cast<uint32_t>(expected.scores[row]),
                    "time-quad scan or scalar tail changed score bits");
            }
            ++case_count;
        }
    }

    std::cout << "attention_score_time_quad_scan_cases=" << case_count << '\n';
}

} // namespace

int main()
{
    TestOrderedDotParity();
    TestChronologicalScanAndTails();
    std::cout << "attention score time-quad tests passed\n";
    return 0;
}

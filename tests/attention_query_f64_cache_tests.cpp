#include "AI/AIComputKrnlInternal.h"

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
    std::cerr << "attention_query_f64_cache_tests: " << message << '\n';
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

double InlineConversionDot(
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

void TestExactPreparationAndOrderedDot()
{
    constexpr uint32_t kHeadDimensions[] = {
        1U, 2U, 3U, 4U, 7U, 8U, 15U, 31U, 32U, 33U, 64U, 127U, 128U
    };
    uint32_t random_state = 0x514B4637U;
    size_t case_count = 0U;

    for (uint32_t head_dim : kHeadDimensions) {
        for (uint32_t sample = 0; sample < 16U; ++sample) {
            std::vector<float> query_storage(static_cast<size_t>(head_dim) + 2U);
            std::vector<float> key_storage(static_cast<size_t>(head_dim) + 2U);
            std::vector<double> prepared_storage(static_cast<size_t>(head_dim) + 2U);
            float* query = query_storage.data() + 1;
            float* key = key_storage.data() + 1;
            double* prepared = prepared_storage.data() + 1;

            for (uint32_t dimension = 0; dimension < head_dim; ++dimension) {
                const int32_t query_bits =
                    static_cast<int32_t>(NextRandom(&random_state) % 131073U) - 65536;
                const int32_t key_bits =
                    static_cast<int32_t>(NextRandom(&random_state) % 131073U) - 65536;
                query[dimension] = static_cast<float>(query_bits) / 8192.0f;
                key[dimension] = static_cast<float>(key_bits) / 4096.0f;
            }

            if (head_dim >= 4U && sample == 0U) {
                query[0] = 0.0f;
                query[1] = -0.0f;
                query[2] = std::numeric_limits<float>::denorm_min();
                query[3] = -std::numeric_limits<float>::denorm_min();
            }

            micius::internal::PrepareAttentionQueryF64(
                query, prepared, head_dim);
            for (uint32_t dimension = 0; dimension < head_dim; ++dimension) {
                Require(
                    std::bit_cast<uint64_t>(prepared[dimension]) ==
                        std::bit_cast<uint64_t>(static_cast<double>(query[dimension])),
                    "binary32-to-binary64 preparation was not exact");
            }

            const double expected = InlineConversionDot(query, key, head_dim);
            const double actual =
                micius::internal::AttentionScoreDotPreparedQueryF64(
                    prepared, key, head_dim);
            Require(
                std::bit_cast<uint64_t>(actual) == std::bit_cast<uint64_t>(expected),
                "prepared-query dot changed ordered accumulation bits");
            ++case_count;
        }
    }

    std::cout << "attention_query_f64_cache_cases=" << case_count << '\n';
}

} // namespace

int main()
{
    TestExactPreparationAndOrderedDot();
    std::cout << "attention query F64 cache tests passed\n";
    return 0;
}

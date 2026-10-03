#include "AI/AIComputKrnlInternal.h"
#include "AI/PreparedGEMVInput.h"
#include <immintrin.h>
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <span>
#include <thread>
#include <vector>

namespace r8_frozen {
Q8_32_GEMVStatus RowPair(const int8_t*, const QuantDescriptor*, const float*,
                       float*, uint32_t, uint32_t) noexcept;
}

namespace {
using namespace micius::internal;
using Status = Q8_32_GEMVStatus;
[[noreturn]] void Fail(const char* why) {
    std::cerr << "prepared_gemv_tests: " << why << '\n'; std::abort();
}
void Check(bool good, const char* why) { if (!good) Fail(why); }
void OK(Status status) { Check(status == Status::OK, "unexpected error status"); }
uint32_t Bits(float f) { return std::bit_cast<uint32_t>(f); }
uint32_t Next(uint32_t& s) { s = s * 1664525U + 1013904223U; return s; }
void Equal(std::span<const float> a, std::span<const float> b) {
    Check(a.size() == b.size(), "size mismatch");
    for (size_t i = 0; i < a.size(); ++i)
        Check(Bits(a[i]) == Bits(b[i]), "bit-pattern mismatch");
}

void Shapes() {
    uint32_t random = 0xb1790043U;
    size_t cases = 0;
    for (uint32_t cols : {1U, 7U, 31U, 32U, 33U, 63U, 64U, 65U, 129U, 256U, 768U, 4096U}) {
        for (uint32_t rows : {1U, 2U, 3U, 4U, 5U, 7U, 16U}) {
            for (unsigned trial = 0; trial < 3; ++trial) {
                const size_t n = static_cast<size_t>(rows) * cols;
                const size_t blocks = cols / 32U + (cols % 32U != 0U ? 1U : 0U);
                std::vector<int8_t> w(n + 2, -11);
                std::vector<float> x(cols + 2U, -13.0f);
                std::vector<double> scratch(cols + 2U, -17.0);
                std::vector<QuantDescriptor> scales(rows * blocks);
                std::vector<float> compact(rows * blocks);
                std::vector<float> expected(rows), baseline(rows), whole(rows);
                std::vector<float> partition(rows + 2U, -19.0f), pre(rows), small(rows);
                for (size_t i = 0; i < n; ++i)
                    w[i + 1] = static_cast<int8_t>(static_cast<int>(Next(random) & 255U) - 128);
                for (size_t i = 0; i < cols; ++i) {
                    const int32_t value = static_cast<int32_t>(Next(random) % 2001U) - 1000;
                    x[i + 1] = static_cast<float>(value) / 257.0f;
                    if (trial == 1 && i % 7 == 0) x[i + 1] = -0.0f;
                    if (trial == 2) x[i + 1] = std::ldexp(x[i + 1], static_cast<int>(i % 41) - 20);
                }
                for (size_t i = 0; i < scales.size(); ++i) {
                    const float scale = static_cast<float>(static_cast<int>(i % 17) - 8) / 521.0f;
                    scales[i] = QuantDescriptor(scale, 0); compact[i] = scale;
                }
                const auto original_w = w; const auto original_x = x;
                PreparedF64GEMVInput prepared;
                OK(PrepareF64GEMVInput({x.data() + 1, cols}, {scratch.data() + 1, cols}, prepared));
                OK(r8_frozen::RowPair(w.data() + 1, scales.data(), x.data() + 1,
                                     expected.data(), cols, rows));
                OK(GEMV_Q8_32_F32_AVX2_RowPair(w.data() + 1, scales.data(), x.data() + 1,
                                             baseline.data(), cols, rows));
                OK(GEMV_Q8_32_PreparedF64Rows(w.data() + 1, scales.data(), prepared,
                                            whole.data(), rows, 0, rows));
                // Reverse task order, with only the final task owning an odd row.
                uint32_t end = rows;
                while (end != 0) {
                    const uint32_t begin = (end - 1U) & ~1U;
                    OK(GEMV_Q8_32_PreparedF64Rows(w.data() + 1, scales.data(), prepared,
                                                partition.data() + 1, rows, begin, end));
                    end = begin;
                }
                OK(GEMV_Q8_32_PreparedF64RowsPrevalidatedScales(w.data() + 1, scales.data(),
                    prepared, pre.data(), rows, 0, rows));
                OK(GEMV_Q8_32_PreparedF64RowsCompactScales(w.data() + 1, compact.data(),
                    prepared, small.data(), rows, 0, rows));
                Equal(expected, baseline); Equal(expected, whole); Equal(expected, pre);
                Equal(expected, small); Equal(expected, {partition.data() + 1, rows});
                Check(w == original_w && x == original_x, "source modified");
                Check(partition.front() == -19.0f && partition.back() == -19.0f,
                      "output canary overwritten");
                Check(scratch.front() == -17.0 && scratch.back() == -17.0,
                      "scratch canary overwritten");
                for (size_t i = 0; i < cols; ++i)
                    Check(std::bit_cast<uint64_t>(scratch[i + 1]) ==
                          std::bit_cast<uint64_t>(static_cast<double>(x[i + 1])),
                          "prepared snapshot changed during compute");
                ++cases;
            }
        }
    }
    std::cout << "shape_bit_parity_cases=" << cases << '\n';
}

void TailContraction() {
    // Exact cancellation under separate multiply/add. A contracted final tail
    // multiply-add leaves a representable nonzero F32 residual instead.
    constexpr uint32_t cols = 33, rows = 3;
    std::vector<int8_t> w(cols * rows, 0);
    std::vector<float> x(cols, 0.0f);
    const float value = std::bit_cast<float>(0x3fc00001U);
    const float scale = std::bit_cast<float>(0x3fc00003U);
    x[0] = value; x[32] = value;
    for (size_t r = 0; r < rows; ++r) { w[r * cols] = 127; w[r * cols + 32] = -127; }
    std::vector<QuantDescriptor> s(rows * 2U, QuantDescriptor(scale, 0));
    std::vector<double> scratch(cols);
    std::array<float, rows> a{}, b{}, frozen{};
    PreparedF64GEMVInput prepared;
    OK(PrepareF64GEMVInput(x, scratch, prepared));
    OK(GEMV_Q8_32_F32_AVX2_RowPair(w.data(), s.data(), x.data(), a.data(), cols, rows));
    OK(GEMV_Q8_32_PreparedF64Rows(w.data(), s.data(), prepared, b.data(), rows, 0, rows));
    OK(r8_frozen::RowPair(w.data(), s.data(), x.data(), frozen.data(), cols, rows));
    for (size_t r = 0; r < rows; ++r) {
        Check(Bits(a[r]) == 0U && Bits(b[r]) == 0U && Bits(frozen[r]) == 0U,
              "tail contraction violated separate-rounding fixture");
    }
    volatile double rounded = (static_cast<double>(value) * 127.0) * scale;
    const double fused = std::fma(-static_cast<double>(value) * 127.0,
                                  static_cast<double>(scale), rounded);
    Check(Bits(static_cast<float>(fused)) == 0xa8800000U, "fixture no longer distinguishes FMA");
    std::cout << "tail_contraction_bit_fixture=PASS\n";
}

void FailuresAndReuse() {
    std::array<float, 33> x{}; x.fill(1.0f);
    std::array<double, 33> scratch{};
    std::array<int8_t, 99> w{}; w.fill(1);
    std::array<QuantDescriptor, 6> s{};
    for (auto& v : s) v = QuantDescriptor(0.25f, 0);
    std::array<float, 3> y{}; y.fill(-19.0f);
    PreparedF64GEMVInput input;
    auto rows = [&](uint32_t begin, uint32_t end) {
        return GEMV_Q8_32_PreparedF64Rows(w.data(), s.data(), input, y.data(), 3, begin, end);
    };
    Check(rows(0, 3) == Status::INVALID_ARGUMENT, "unprepared view accepted");
    Check(PrepareF64GEMVInput({}, scratch, input) == Status::INVALID_ARGUMENT, "empty input accepted");
    Check(PrepareF64GEMVInput(x, {scratch.data(), 32}, input) == Status::INVALID_ARGUMENT,
          "short scratch accepted");
    OK(PrepareF64GEMVInput(x, scratch, input));
    for (auto range : {std::pair{1U, 3U}, {0U, 1U}, {2U, 1U}, {0U, 4U}})
        Check(rows(range.first, range.second) == Status::INVALID_ARGUMENT, "bad row range accepted");
    OK(rows(3, 3)); Check(y[0] == -19.0f && y[2] == -19.0f, "empty range wrote output");
    Check(GEMV_Q8_32_PreparedF64Rows(nullptr, s.data(), input, y.data(), 3, 0, 3) ==
          Status::INVALID_ARGUMENT, "null weights accepted");
    Check(GEMV_Q8_32_PreparedF64Rows(w.data(), s.data(), input,
          reinterpret_cast<float*>(scratch.data()), 3, 0, 3) == Status::INVALID_ARGUMENT,
          "output/snapshot alias accepted");
    const auto old_control = _mm_getcsr();
    _mm_setcsr(old_control ^ 0x8000U);
    const auto environment_status = rows(0, 3);
    _mm_setcsr(old_control);
    Check(environment_status == Status::INVALID_ARGUMENT, "changed FP control accepted");
    for (float invalid : {std::numeric_limits<float>::infinity(),
                          -std::numeric_limits<float>::infinity(),
                          std::numeric_limits<float>::quiet_NaN()}) {
        x[32] = invalid;
        Check(PrepareF64GEMVInput(x, scratch, input) == Status::NUMERIC_ERROR,
              "non-finite activation accepted");
        Check(input.data() == nullptr && input.size() == 0, "stale view survived failed prepare");
        Check(rows(0, 3) == Status::INVALID_ARGUMENT, "failed preparation consumed");
    }
    x.fill(2.0f); OK(PrepareF64GEMVInput(x, scratch, input));
    // Mutating the source must not change its already prepared snapshot.
    x.fill(9.0f); OK(rows(0, 3)); Check(y[0] == 16.5f, "snapshot tied to mutable source");
    OK(PrepareF64GEMVInput(x, scratch, input)); OK(rows(0, 3));
    Check(y[0] == 74.25f, "same-address new input reused stale values");
    s[0].zero_point = 1;
    Check(rows(0, 3) == Status::INVALID_MODEL, "invalid descriptor accepted");
    s[0].zero_point = 0; s[5].scale = std::numeric_limits<float>::infinity();
    Check(rows(0, 3) == Status::INVALID_MODEL, "invalid final tail scale accepted");
    s[5].scale = 0.25f;
    x.fill(std::numeric_limits<float>::max()); w.fill(127);
    OK(PrepareF64GEMVInput(x, scratch, input));
    Check(rows(0, 3) == Status::NUMERIC_ERROR, "F32 output overflow accepted");
    x.fill(1.0f); w.fill(1); OK(PrepareF64GEMVInput(x, scratch, input));
    OK(rows(0, 3)); Check(y[0] == 8.25f && y[2] == 8.25f, "recovery after failure failed");
}

void ParallelRowsAndSubnormals() {
    constexpr uint32_t cols = 65, nrows = 7;
    std::vector<int8_t> w(cols * nrows, -127);
    std::vector<QuantDescriptor> s(nrows * 3U, QuantDescriptor(0.013f, 0));
    std::vector<float> x(cols, std::numeric_limits<float>::denorm_min());
    std::vector<double> scratch(cols);
    std::vector<float> serial(nrows), split(nrows), reference(nrows);
    const auto saved = _mm_getcsr();
    for (uint32_t control : {0x1f80U, 0x9fc0U}) {
        _mm_setcsr(control);
        PreparedF64GEMVInput input;
        OK(PrepareF64GEMVInput(x, scratch, input));
        OK(r8_frozen::RowPair(w.data(), s.data(), x.data(), reference.data(), cols, nrows));
        OK(GEMV_Q8_32_PreparedF64Rows(w.data(), s.data(), input, serial.data(), nrows, 0, nrows));
        std::array<Status, 2> status{};
        std::thread a([&] { _mm_setcsr(control); status[0] = GEMV_Q8_32_PreparedF64Rows(
            w.data(), s.data(), input, split.data(), nrows, 0, 4); });
        std::thread b([&] { _mm_setcsr(control); status[1] = GEMV_Q8_32_PreparedF64Rows(
            w.data(), s.data(), input, split.data(), nrows, 4, nrows); });
        a.join(); b.join(); OK(status[0]); OK(status[1]);
        Equal(reference, serial); Equal(serial, split);
    }
    _mm_setcsr(saved);
}
} // namespace

int main() {
    const auto saved = _mm_getcsr(); _mm_setcsr(0x1f80U);
    Shapes(); TailContraction(); FailuresAndReuse(); ParallelRowsAndSubnormals();
    _mm_setcsr(saved);
    std::cout << "prepared_gemv_tests=PASS\n";
}

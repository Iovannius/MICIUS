#pragma once

#include "AIComputKrnl.h"
#include <cstdint>
#include <fstream>
#include <vector>

// Private, default-OFF, single-position numerical experiment. No public ABI.
namespace micius::i3 {
enum class Policy { None, L3GU, L3Down, L3FFN, L31GU, L31Down, L31FFN, BothFFN, AllF64 };
enum class Op { QKV, O, Gate, Up, Down, LM };
bool ParsePolicy(const char*, Policy&) noexcept;
const char* PolicyName(Policy) noexcept;
const char* OpName(Op) noexcept;
bool Select(Policy, uint32_t layer, Op) noexcept;

// Exact F32 -> F64 preparation, followed by the unchanged v0.6 row-pair
// consumer. Weight data/scales remain Q8_32. Scratch is private to this call.
Q8_32_GEMVStatus F64(const int8_t*, const QuantDescriptor*, const float*, float*,
                    uint32_t cols, uint32_t rows, bool validated) noexcept;

class Session final {
public:
    Session(uint32_t token, uint32_t position, uint32_t layers,
            const char* policy = nullptr, const char* directory = nullptr) noexcept;
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    bool good() const noexcept { return good_; }
    bool selected() const noexcept { return active_ && Select(policy_, layer_, op_); }
    void set(uint32_t layer, Op op) noexcept { layer_ = layer; op_ = op; }
    bool begin(const float*, uint32_t cols, uint32_t rows) noexcept;
    bool end(const float* input, const float* output) noexcept;
    bool boundary(int layer, const char* name, const float*, uint32_t count) noexcept;
    bool start_ffn(uint32_t layer, const float* input, uint32_t count) noexcept;
    bool end_ffn(const float* input, uint32_t count) noexcept;
    bool complete() noexcept;

private:
    bool fail() noexcept { good_ = false; return false; }
    bool write_tensor(const float*, uint32_t count);
    bool good_ = true, active_ = false, pending_ = false, ffn_pending_ = false, done_ = false;
    Policy policy_ = Policy::None;
    Op op_ = Op::QKV;
    uint32_t token_ = 0, layers_ = 0, layer_ = 0, cols_ = 0, rows_ = 0;
    uint32_t count_ = 0, selected_count_ = 0, ffn_count_ = 0, ffn_layer_ = 0, mxcsr_ = 0;
    uint64_t offset_ = 0;
    std::vector<float> snapshot_, ffn_snapshot_;
    std::ofstream records_, tensors_;
};
} // namespace micius::i3

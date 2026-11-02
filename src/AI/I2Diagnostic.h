#pragma once
#include "PreparedW8A8Input.h"
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

// Private, compile-time opt-in instrumentation. Never part of the public ABI.
namespace micius::i2 {
extern thread_local int32_t* dot_sink;
extern thread_local uint32_t dot_stride;

struct Capture {
    std::vector<float> input;
    std::vector<int32_t> dots;
    uint32_t mxcsr = 0;
};

struct Evaluation {
    std::vector<float> f64, oracle, scalar, avx, reconstructed;
    std::vector<int8_t> codes;
    std::vector<double> scales;
    std::string json;
    uint32_t worst_row = 0;
    bool ok = false;
};

// Throws on invalid data/status. Independent quantization, dot and postprocess.
Evaluation Evaluate(const float* x, uint32_t cols, uint32_t rows,
    const int8_t* weights, const QuantDescriptor* descriptors,
    const float* actual = nullptr, int mode = 2,
    const internal::PreparedW8A8Input* prepared = nullptr,
    const int32_t* actual_dots = nullptr);

class Session {
public:
    Session(uint32_t token, uint32_t position, int mode,
            const char* directory = nullptr) noexcept;
    ~Session();
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    bool good() const noexcept { return good_; }
    bool active() const noexcept { return active_; }
    void set(int layer, const char* operation) noexcept;
    Capture capture(const float* x, uint32_t cols, uint32_t rows) noexcept;
    bool finish(const Capture&, const int8_t*, const QuantDescriptor*,
        const float* live_input, const float* actual, uint32_t rows,
        const internal::PreparedW8A8Input* prepared = nullptr) noexcept;
    bool boundary(int layer, const char* name, const float* values, uint32_t count) noexcept;
    bool complete() noexcept;
private:
    void fail(const char*) noexcept;
    std::filesystem::path directory_;
    std::ofstream records_, tensors_;
    uint32_t token_, position_, sequence_ = 0;
    int mode_, layer_ = -1;
    std::string operation_;
    bool active_ = false, good_ = true, complete_ = false;
};

// Replays at most two selected weight rows from each saved full-input snapshot.
// No GGUF, forward pass, or Context is needed. Throws on malformed files.
void Replay(const std::filesystem::path& input, const std::filesystem::path& output);
} // namespace micius::i2

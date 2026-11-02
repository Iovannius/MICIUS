#include "I3Intervention.h"
#include "PreparedGEMVInput.h"
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <immintrin.h>
#include <string>

namespace micius::i3 {
namespace {
constexpr const char* names[] = {"NONE", "L3_GU", "L3_DOWN", "L3_FFN", "L31_GU",
                                "L31_DOWN", "L31_FFN", "L3_L31_FFN", "F64_ALL"};
struct FPGuard {
    unsigned value = _mm_getcsr();
    ~FPGuard() { _mm_setcsr(value); }
};
bool supported_fp() { return (_mm_getcsr() & 0xffc0U) == 0x1f80U; }
}
bool ParsePolicy(const char* text, Policy& result) noexcept {
    if (!text) return false;
    for (unsigned i = 0; i < 9; ++i) if (std::strcmp(text, names[i]) == 0) {
        result = static_cast<Policy>(i); return true;
    }
    return false;
}
const char* PolicyName(Policy policy) noexcept { return names[static_cast<unsigned>(policy)]; }
const char* OpName(Op op) noexcept {
    constexpr const char* ops[] = {"qkv", "o", "gate", "up", "down", "lm"};
    return ops[static_cast<unsigned>(op)];
}
bool Select(Policy policy, uint32_t layer, Op op) noexcept {
    if (policy == Policy::AllF64) return true;
    const bool gu = op == Op::Gate || op == Op::Up;
    const bool down = op == Op::Down;
    if (layer == 3) {
        if (policy == Policy::L3GU) return gu;
        if (policy == Policy::L3Down) return down;
        if (policy == Policy::L3FFN || policy == Policy::BothFFN) return gu || down;
    }
    if (layer == 31) {
        if (policy == Policy::L31GU) return gu;
        if (policy == Policy::L31Down) return down;
        if (policy == Policy::L31FFN || policy == Policy::BothFFN) return gu || down;
    }
    return false;
}
Q8_32_GEMVStatus F64(const int8_t* w, const QuantDescriptor* s, const float* x, float* y,
                    uint32_t cols, uint32_t rows, bool validated) noexcept {
    if (!w || !s || !x || !y || !cols || !rows || !supported_fp()) return Q8_32_GEMVStatus::INVALID_ARGUMENT;
    try {
        std::vector<double> scratch(cols);
        internal::PreparedF64GEMVInput prepared;
        auto status = internal::PrepareF64GEMVInput({x, cols}, scratch, prepared);
        if (status != Q8_32_GEMVStatus::OK) return status;
        return validated ? internal::GEMV_Q8_32_PreparedF64RowsPrevalidatedScales(w,s,prepared,y,rows,0,rows)
                         : internal::GEMV_Q8_32_PreparedF64Rows(w,s,prepared,y,rows,0,rows);
    } catch (...) { return Q8_32_GEMVStatus::INVALID_MODEL; }
}

Session::Session(uint32_t token, uint32_t position, uint32_t layers,
                 const char* policy, const char* directory) noexcept
    : token_(token), layers_(layers) {
    FPGuard guard;
    if (!policy && !directory) {
        policy = std::getenv("MICIUS_I3_POLICY");
        directory = std::getenv("MICIUS_I3_OUTPUT");
    }
    if ((!policy || !*policy) && (!directory || !*directory)) return;
    try {
        if (!policy || !directory || !*directory || !ParsePolicy(policy, policy_) ||
            position != 0 || layers == 0 || layers > 4096 || !supported_fp()) { fail(); return; }
        const uint32_t required = (policy_ == Policy::L3GU || policy_ == Policy::L3Down || policy_ == Policy::L3FFN) ? 4U :
            (policy_ == Policy::None || policy_ == Policy::AllF64) ? 1U : 32U;
        if (layers < required) { fail(); return; }
        // Require a new directory: no appending to an old run or overwriting evidence.
        if (!std::filesystem::create_directory(directory)) { fail(); return; }
        const std::filesystem::path root(directory);
        records_.open(root / "records.jsonl", std::ios::out | std::ios::binary);
        tensors_.open(root / "tensors.f32", std::ios::out | std::ios::binary);
        if (!records_ || !tensors_) { fail(); return; }
        active_ = true;
        records_ << "{\"type\":\"begin\",\"schema\":1,\"token\":" << token_
                 << ",\"position\":0,\"layers\":" << layers_ << ",\"policy\":\"" << policy
                 << "\",\"mxcsr\":" << _mm_getcsr() << "}\n";
        if (!records_) fail();
    } catch (...) { fail(); }
}
bool Session::write_tensor(const float* values, uint32_t count) {
    if (!values || !count) return fail();
    for (uint32_t i = 0; i < count; ++i) if (!std::isfinite(values[i])) return fail();
    tensors_.write(reinterpret_cast<const char*>(values), static_cast<std::streamsize>(count) * 4);
    offset_ += static_cast<uint64_t>(count) * 4;
    return tensors_ ? true : fail();
}
bool Session::begin(const float* input, uint32_t cols, uint32_t rows) noexcept {
    if (!good_) return false;
    if (!active_) return true;
    FPGuard guard;
    try {
        const auto expected_op = count_ == layers_*5U ? Op::LM : static_cast<Op>(count_%5U);
        const auto expected_layer = count_ == layers_*5U ? layers_ : count_/5U;
        if (done_ || pending_ || !input || !cols || !rows || !supported_fp() ||
            count_ > layers_*5U || layer_ != expected_layer || op_ != expected_op) return fail();
        snapshot_.assign(input, input+cols);
        for (float v : snapshot_) if (!std::isfinite(v)) return fail();
        cols_ = cols; rows_ = rows; mxcsr_ = _mm_getcsr(); pending_ = true;
        return true;
    } catch (...) { return fail(); }
}
bool Session::end(const float* input, const float* output) noexcept {
    if (!good_) return false;
    if (!active_) return true;
    FPGuard guard;
    try {
        if (!pending_ || !input || !output || !supported_fp() ||
            std::memcmp(input,snapshot_.data(),static_cast<size_t>(cols_)*4) != 0) return fail();
        const uint64_t input_offset = offset_;
        if (!write_tensor(snapshot_.data(),cols_)) return false;
        const uint64_t output_offset = offset_;
        if (!write_tensor(output,rows_)) return false;
        records_ << "{\"type\":\"gemv\",\"layer\":" << layer_ << ",\"op\":\"" << OpName(op_)
                 << "\",\"backend\":\"" << (selected() ? "F64" : "I1_AVX2")
                 << "\",\"cols\":" << cols_ << ",\"rows\":" << rows_
                 << ",\"input_offset\":" << input_offset << ",\"output_offset\":" << output_offset
                 << ",\"input_unchanged\":true,\"mxcsr_before\":" << mxcsr_
                 << ",\"mxcsr_after\":" << _mm_getcsr() << "}\n";
        ++count_; if (selected()) ++selected_count_;
        pending_ = false;
        return records_ ? true : fail();
    } catch (...) { return fail(); }
}
bool Session::boundary(int layer, const char* name, const float* values, uint32_t count) noexcept {
    if (!good_) return false;
    if (!active_) return true;
    FPGuard guard;
    try {
        if (done_ || pending_ || !supported_fp()) return fail();
        const auto offset = offset_;
        if (!write_tensor(values,count)) return false;
        records_ << "{\"type\":\"boundary\",\"layer\":" << layer << ",\"name\":\"" << name
                 << "\",\"count\":" << count << ",\"offset\":" << offset
                 << ",\"mxcsr\":" << _mm_getcsr() << "}\n";
        return records_ ? true : fail();
    } catch (...) { return fail(); }
}
bool Session::start_ffn(uint32_t layer, const float* input, uint32_t count) noexcept {
    if (!good_) return false;
    if (!active_) return true;
    FPGuard guard;
    try {
        if (ffn_pending_ || layer != ffn_count_ || count_ != layer*5U+2U || !input || !count) return fail();
        ffn_snapshot_.assign(input,input+count); ffn_layer_ = layer; ffn_pending_ = true;
        return boundary(static_cast<int>(layer),"ffn_input_before_prepare",input,count);
    } catch (...) { return fail(); }
}
bool Session::end_ffn(const float* input, uint32_t count) noexcept {
    if (!good_) return false;
    if (!active_) return true;
    FPGuard guard;
    try {
        if (!ffn_pending_ || !input || count != ffn_snapshot_.size() || count_ != ffn_layer_*5U+4U ||
            !supported_fp() || std::memcmp(input,ffn_snapshot_.data(),static_cast<size_t>(count)*4) != 0) return fail();
        records_ << "{\"type\":\"ffn_input_check\",\"layer\":" << ffn_layer_
                 << ",\"scope\":\"before_shared_prepare_through_gate_up\",\"unchanged\":true}\n";
        ffn_pending_ = false; ++ffn_count_;
        return records_ ? true : fail();
    } catch (...) { return fail(); }
}
bool Session::complete() noexcept {
    if (!good_) return false;
    if (!active_) return true;
    FPGuard guard;
    try {
        if (done_ || pending_ || ffn_pending_ || count_ != layers_*5U+1U || ffn_count_ != layers_ || !supported_fp()) return fail();
        records_ << "{\"type\":\"complete\",\"gemvs\":" << count_ << ",\"f64_gemvs\":"
                 << selected_count_ << ",\"ffn_input_checks\":" << ffn_count_ << ",\"tensor_bytes\":" << offset_ << "}\n";
        records_.flush(); tensors_.flush();
        const bool ok = records_ && tensors_;
        records_.close(); tensors_.close(); done_ = true;
        return ok && records_ && tensors_ ? true : fail();
    } catch (...) { return fail(); }
}
} // namespace micius::i3

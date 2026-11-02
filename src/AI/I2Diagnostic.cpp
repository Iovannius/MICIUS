#include "I2Diagnostic.h"
#include "PreparedGEMVInput.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <immintrin.h>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace micius::i2 {
thread_local int32_t* dot_sink = nullptr;
thread_local uint32_t dot_stride = 0;
namespace {
using internal::PreparedW8A8Input;
using Status = Q8_32_GEMVStatus;
struct FPGuard {
    unsigned saved = _mm_getcsr();
    ~FPGuard() { _mm_setcsr(saved); }
};
struct DotGuard {
    ~DotGuard() { dot_sink = nullptr; dot_stride = 0; }
};
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class T> bool equal_bits(T a, T b) { return std::memcmp(&a, &b, sizeof(T)) == 0; }
template<class T> size_t mismatches(const T* a, const T* b, size_t n) {
    size_t count = 0;
    for (size_t i = 0; i < n; ++i) count += !equal_bits(a[i], b[i]);
    return count;
}
template<class T> int64_t first_mismatch(const T* a, const T* b, size_t n) {
    for (size_t i = 0; i < n; ++i) if (!equal_bits(a[i], b[i])) return static_cast<int64_t>(i);
    return -1;
}
template<class A, class B> std::string metrics(const A* a, const B* b, size_t n) {
    double aa = 0, bb = 0, se = 0, maximum = 0;
    size_t worst = 0;
    for (size_t i = 0; i < n; ++i) {
        const double x = static_cast<double>(a[i]), y = static_cast<double>(b[i]);
        require(std::isfinite(x) && std::isfinite(y), "Non-finite diagnostic value");
        const double d = y - x;
        aa += x*x; bb += y*y; se += d*d;
        if (std::abs(d) > maximum) { maximum = std::abs(d); worst = i; }
    }
    std::ostringstream s;
    s << std::setprecision(17) << "{\"max_abs\":" << maximum
      << ",\"rmse\":" << std::sqrt(se/static_cast<double>(n))
      << ",\"reference_l2\":" << std::sqrt(aa) << ",\"candidate_l2\":" << std::sqrt(bb)
      << ",\"relative_l2\":";
    if (aa > 0) s << std::sqrt(se/aa); else s << "null";
    s << ",\"worst_index\":" << worst << '}';
    return s.str();
}
int8_t quantize(double x, double scale) {
    if (scale == 0.0) return 0;
    const double value = std::abs(x / scale);
    const double base = std::floor(value), fraction = value - base;
    int q = static_cast<int>(base);
    if (fraction > 0.5 || (fraction == 0.5 && (q % 2) != 0)) ++q;
    q = std::min(127, q);
    return static_cast<int8_t>(std::signbit(x) ? -q : q);
}
float checked_float(double value) {
    require(std::isfinite(value) && std::abs(value) <= std::numeric_limits<float>::max(),
            "Diagnostic output overflow");
    return static_cast<float>(value);
}
// Diagnostic-only arithmetic model of the frozen four chains and lane tree.
// xhat stays binary64; it is NEVER rounded to F32 before the products.
double elementwise_tree(const double* x, const int8_t* w, const QuantDescriptor* s, uint32_t n) {
    double a[4][4] = {};
    const uint32_t full = n / 32U;
    for (uint32_t b = 0; b < full; ++b)
        for (uint32_t group = 0; group < 8; ++group)
            for (uint32_t lane = 0; lane < 4; ++lane) {
                const uint32_t k = b*32U + group*4U + lane;
                const double product = x[k]*static_cast<double>(w[k]);
                a[group%4U][lane] += product*static_cast<double>(s[b].scale);
            }
    double v[4];
    for (uint32_t lane = 0; lane < 4; ++lane)
        v[lane] = (a[0][lane]+a[1][lane])+(a[2][lane]+a[3][lane]);
    double sum = (v[0]+v[2])+(v[1]+v[3]);
    for (uint32_t k = full*32U; k < n; ++k) {
        const double product = x[k]*static_cast<double>(w[k]);
        sum += product*static_cast<double>(s[full].scale);
    }
    return sum;
}
template<class T> void write_values(std::ostream& stream, const T* data, size_t count) {
    stream.write(reinterpret_cast<const char*>(data), static_cast<std::streamsize>(count*sizeof(T)));
}
template<class T> void read_values(std::istream& stream, T* data, size_t count) {
    stream.read(reinterpret_cast<char*>(data), static_cast<std::streamsize>(count*sizeof(T)));
    require(static_cast<bool>(stream), "Truncated replay file");
}
std::string key(int layer, const std::string& op) {
    return "layer" + std::to_string(layer) + "." + op;
}
} // namespace

Evaluation Evaluate(const float* x, uint32_t cols, uint32_t rows,
    const int8_t* weights, const QuantDescriptor* descriptors,
    const float* actual, int mode, const PreparedW8A8Input* prepared,
    const int32_t* actual_dots) {
    FPGuard fp;
    DotGuard dots_guard;
    dot_sink = nullptr;
    require((_mm_getcsr() & 0xffc0U) == 0x1f80U, "Unsupported MXCSR");
    require(x && weights && descriptors && cols && rows, "Invalid diagnostic matrix");
    require(mode >= 0 && mode <= 2, "Invalid diagnostic backend");
    const uint32_t blocks = cols/32U + (cols%32U != 0U);
    const size_t ndots = static_cast<size_t>(rows)*blocks;
    require(ndots <= 100000000U, "Diagnostic matrix too large");
    Evaluation result;
    result.codes.resize(cols); result.scales.resize(blocks);
    result.f64.resize(rows); result.oracle.resize(rows); result.scalar.resize(rows);
    result.avx.resize(rows); result.reconstructed.resize(rows);
    std::vector<double> exact_x(cols), reconstructed(cols), scratch(cols);
    std::vector<int8_t> production_codes(cols);
    std::vector<double> production_scales(blocks);
    std::vector<int32_t> expected_dots(ndots), avx_dots(ndots, INT32_MIN), scalar_dots(ndots, INT32_MIN);
    std::vector<float> tree_original(rows), contract_prepared(rows);
    for (uint32_t b = 0; b < blocks; ++b) {
        double maximum = 0;
        const uint32_t end = std::min(cols, b*32U+32U);
        for (uint32_t k = b*32U; k < end; ++k) {
            require(std::isfinite(x[k]), "Non-finite activation");
            exact_x[k] = static_cast<double>(x[k]);
            maximum = std::max(maximum, std::abs(exact_x[k]));
        }
        result.scales[b] = maximum/127.0;
        for (uint32_t k = b*32U; k < end; ++k) {
            result.codes[k] = quantize(exact_x[k], result.scales[b]);
            reconstructed[k] = static_cast<double>(result.codes[k])*result.scales[b];
        }
    }
    for (size_t i = 0; i < ndots; ++i)
        require(std::isfinite(descriptors[i].scale) && descriptors[i].zero_point == 0 &&
                descriptors[i].padding[0] == 0 && descriptors[i].padding[1] == 0, "Invalid descriptor");
    PreparedW8A8Input fresh;
    require(internal::PrepareW8A8Input({x, cols}, production_codes, production_scales, fresh) == Status::OK,
            "Fresh I1 preparation failed");
    const PreparedW8A8Input& view = prepared ? *prepared : fresh;
    require(view.values() && view.scales() && view.size() == cols, "Invalid live prepared view");
    dot_stride = blocks; dot_sink = avx_dots.data();
    require(internal::GEMV_W8A8Rows(weights, descriptors, view, result.avx.data(), rows, 0, rows, false, false) == Status::OK,
            "AVX2 replay failed");
    dot_sink = scalar_dots.data();
    require(internal::GEMV_W8A8Rows(weights, descriptors, view, result.scalar.data(), rows, 0, rows, false, true) == Status::OK,
            "Scalar replay failed");
    dot_sink = nullptr;
    internal::PreparedF64GEMVInput f64;
    require(internal::PrepareF64GEMVInput({x, cols}, scratch, f64) == Status::OK, "F64 prepare failed");
    require(internal::GEMV_Q8_32_PreparedF64Rows(weights, descriptors, f64, result.f64.data(), rows, 0, rows) == Status::OK,
            "Frozen F64 replay failed");
    double worst = -1.0, grouping_double_max = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        const auto* w = weights + static_cast<size_t>(r)*cols;
        const auto* d = descriptors + static_cast<size_t>(r)*blocks;
        double sum = 0, supplied_sum = 0;
        for (uint32_t b = 0; b < blocks; ++b) {
            int64_t dot = 0, supplied_dot = 0;
            require(std::isfinite(view.scales()[b]), "Non-finite prepared scale");
            for (uint32_t k = b*32U; k < std::min(cols, b*32U+32U); ++k) {
                dot += static_cast<int64_t>(w[k])*result.codes[k];
                supplied_dot += static_cast<int64_t>(w[k])*view.values()[k];
            }
            require(std::abs(dot) <= 520192, "INT32 dot range violation");
            expected_dots[static_cast<size_t>(r)*blocks+b] = static_cast<int32_t>(supplied_dot);
            const double weighted = static_cast<double>(dot)*static_cast<double>(d[b].scale);
            sum = sum + weighted*result.scales[b];
            const double supplied_weighted = static_cast<double>(supplied_dot)*static_cast<double>(d[b].scale);
            supplied_sum = supplied_sum + supplied_weighted*view.scales()[b];
        }
        result.oracle[r] = checked_float(sum);
        contract_prepared[r] = checked_float(supplied_sum);
        const double reconstructed_sum = elementwise_tree(reconstructed.data(), w, d, cols);
        grouping_double_max = std::max(grouping_double_max, std::abs(sum-reconstructed_sum));
        result.reconstructed[r] = checked_float(reconstructed_sum);
        tree_original[r] = checked_float(elementwise_tree(exact_x.data(), w, d, cols));
        const double error = std::abs(static_cast<double>(result.oracle[r])-result.f64[r]);
        if (error > worst) { worst = error; result.worst_row = r; }
    }
    const size_t code_bad = mismatches(view.values(), result.codes.data(), cols);
    const size_t scale_bad = mismatches(view.scales(), result.scales.data(), blocks);
    const size_t fresh_code_bad = mismatches(production_codes.data(), result.codes.data(), cols);
    const size_t fresh_scale_bad = mismatches(production_scales.data(), result.scales.data(), blocks);
    const size_t avx_dot_bad = mismatches(avx_dots.data(), expected_dots.data(), ndots);
    const size_t scalar_dot_bad = mismatches(scalar_dots.data(), expected_dots.data(), ndots);
    const size_t actual_dot_bad = actual_dots ? mismatches(actual_dots, expected_dots.data(), ndots) : 0;
    const size_t avx_bad = mismatches(result.avx.data(), contract_prepared.data(), rows);
    const size_t scalar_bad = mismatches(result.scalar.data(), contract_prepared.data(), rows);
    const size_t f64_tree_bad = mismatches(result.f64.data(), tree_original.data(), rows);
    const auto* wanted = mode == 0 ? result.f64.data() : contract_prepared.data();
    const size_t actual_bad = actual ? mismatches(actual, wanted, rows) : 0;
    const int64_t first_code = first_mismatch(view.values(), result.codes.data(), cols);
    const int64_t first_scale = first_mismatch(view.scales(), result.scales.data(), blocks);
    const int64_t first_actual_dot = actual_dots ? first_mismatch(actual_dots, expected_dots.data(), ndots) : -1;
    const int64_t first_output = actual ? first_mismatch(actual, wanted, rows) : -1;
    const int64_t first_dot = first_mismatch(avx_dots.data(), expected_dots.data(), ndots);
    // On failure, retain the first failing row instead of an unrelated large
    // approximation error. Every replay still carries the full activation.
    if (first_output >= 0) result.worst_row = static_cast<uint32_t>(first_output);
    else if (first_actual_dot >= 0) result.worst_row = static_cast<uint32_t>(first_actual_dot)/blocks;
    else if (first_dot >= 0) result.worst_row = static_cast<uint32_t>(first_dot)/blocks;
    result.ok = !(code_bad || scale_bad || fresh_code_bad || fresh_scale_bad || avx_dot_bad ||
        scalar_dot_bad || actual_dot_bad || avx_bad || scalar_bad || f64_tree_bad || actual_bad);
    std::ostringstream out;
    out << std::setprecision(17) << "{\"ok\":" << (result.ok ? "true" : "false")
        << ",\"checks\":{\"prepared_codes\":" << code_bad << ",\"prepared_scales\":" << scale_bad
        << ",\"fresh_codes\":" << fresh_code_bad << ",\"fresh_scales\":" << fresh_scale_bad
        << ",\"avx_dots\":" << avx_dot_bad << ",\"scalar_dots\":" << scalar_dot_bad
        << ",\"actual_dots\":";
    if (actual_dots) out << actual_dot_bad; else out << "null";
    out << ",\"avx_output\":" << avx_bad << ",\"scalar_output\":" << scalar_bad
        << ",\"f64_tree_original\":" << f64_tree_bad << ",\"actual_output\":";
    if (actual) out << actual_bad; else out << "null";
    out << "},\"first_mismatch\":{\"code\":" << first_code
        << ",\"scale\":" << first_scale
        << ",\"avx_dot\":" << first_dot << ",\"actual_dot\":" << first_actual_dot
        << ",\"scalar_dot\":" << first_mismatch(scalar_dots.data(), expected_dots.data(), ndots)
        << ",\"actual_output\":" << first_output << "},\"first_values\":{\"code\":";
    if (first_code >= 0) out << '[' << static_cast<int>(view.values()[first_code]) << ',' << static_cast<int>(result.codes[static_cast<size_t>(first_code)]) << ']';
    else out << "null";
    out << ",\"scale\":";
    if (first_scale >= 0) out << '[' << view.scales()[first_scale] << ',' << result.scales[static_cast<size_t>(first_scale)] << ']';
    else out << "null";
    out << ",\"actual_dot\":";
    if (first_actual_dot >= 0) out << '[' << actual_dots[first_actual_dot] << ',' << expected_dots[static_cast<size_t>(first_actual_dot)] << ']';
    else out << "null";
    out << ",\"actual_output\":";
    if (first_output >= 0) out << '[' << actual[first_output] << ',' << wanted[first_output] << ']';
    else out << "null";
    out << "},\"activation_reconstruction\":" << metrics(x, reconstructed.data(), cols)
        << ",\"same_input_f64_to_i1\":" << metrics(result.f64.data(), result.oracle.data(), rows)
        << ",\"input_change_fixed_f64_tree\":" << metrics(result.f64.data(), result.reconstructed.data(), rows)
        << ",\"grouping_at_reconstructed_input\":" << metrics(result.reconstructed.data(), result.oracle.data(), rows)
        << ",\"grouping_double_max_abs\":" << grouping_double_max;
    const size_t sample = first_dot >= 0 ? static_cast<size_t>(first_dot) : first_actual_dot >= 0 ?
        static_cast<size_t>(first_actual_dot) : static_cast<size_t>(result.worst_row)*blocks;
    const size_t sample_block = sample%blocks;
    out << ",\"sample\":{\"row\":" << sample/blocks << ",\"block\":" << sample_block
        << ",\"activation_scale\":" << view.scales()[sample_block]
        << ",\"expected_activation_scale\":" << result.scales[sample_block]
        << ",\"weight_scale\":" << descriptors[sample].scale
        << ",\"dot_int64\":" << expected_dots[sample] << ",\"dot_avx\":" << avx_dots[sample]
        << ",\"dot_scalar\":" << scalar_dots[sample] << ",\"codes\":[";
    for (uint32_t k = 0; k < std::min(8U, cols-static_cast<uint32_t>(sample_block)*32U); ++k) {
        if (k) out << ',';
        out << static_cast<int>(view.values()[sample_block*32U+k]);
    }
    out << "],\"expected_codes\":[";
    for (uint32_t k = 0; k < std::min(8U, cols-static_cast<uint32_t>(sample_block)*32U); ++k) {
        if (k) out << ',';
        out << static_cast<int>(result.codes[sample_block*32U+k]);
    }
    out << "]}}";
    result.json = out.str();
    return result;
}

Session::Session(uint32_t token, uint32_t position, int mode, const char* directory) noexcept
    : token_(token), position_(position), mode_(mode) {
    FPGuard fp;
    try {
        const char* path = directory ? directory : std::getenv("MICIUS_I2_OUTPUT");
        if (!path || !*path) return;
        active_ = true;
        require(position == 0, "I2 supports a fresh Context, one step, warmup zero only");
        directory_ = std::filesystem::path(path);
        std::filesystem::create_directories(directory_);
        require(!std::filesystem::exists(directory_/"records.jsonl"), "I2 refuses to overwrite existing records");
        records_.open(directory_/"records.jsonl", std::ios::out);
        tensors_.open(directory_/"tensors.f32", std::ios::out|std::ios::binary);
        require(static_cast<bool>(records_) && static_cast<bool>(tensors_), "Cannot open I2 output");
        records_ << "{\"type\":\"begin\",\"schema\":1,\"token\":" << token_
            << ",\"position\":" << position_ << ",\"mode\":" << mode_ << ",\"mxcsr\":" << _mm_getcsr() << "}\n";
    } catch (const std::filesystem::filesystem_error&) { fail("Cannot initialize I2 output directory"); }
    catch (const std::exception& e) { fail(e.what()); }
}
Session::~Session() { dot_sink = nullptr; }
void Session::fail(const char* message) noexcept {
    good_ = false;
    // Error strings are fixed internal messages; no host paths in records.
    if (records_) records_ << "{\"type\":\"error\",\"message\":\"" << message << "\"}\n";
}
void Session::set(int layer, const char* operation) noexcept {
    layer_ = layer;
    try { if (active_) operation_ = operation; }
    catch (...) { fail("Cannot allocate operation label"); }
}
Capture Session::capture(const float* x, uint32_t cols, uint32_t rows) noexcept {
    Capture result;
    if (!active_ || !good_) return result;
    try {
        result.mxcsr = _mm_getcsr();
        result.input.assign(x, x+cols);
        if (mode_) {
            dot_stride = cols/32U + (cols%32U != 0U);
            result.dots.assign(static_cast<size_t>(rows)*dot_stride, INT32_MIN);
            dot_sink = result.dots.data();
        }
    } catch (...) { dot_sink = nullptr; fail("Cannot allocate activation snapshot"); }
    return result;
}
bool Session::boundary(int layer, const char* name, const float* values, uint32_t count) noexcept {
    if (!active_) return true;
    if (!good_) return false;
    FPGuard fp;
    try {
        const auto offset = tensors_.tellp();
        const auto summary = metrics(values, values, count);
        write_values(tensors_, values, count);
        records_ << "{\"type\":\"boundary\",\"token\":" << token_ << ",\"position\":" << position_
            << ",\"layer\":" << layer << ",\"name\":\"" << name << "\",\"count\":" << count
            << ",\"offset\":" << static_cast<int64_t>(offset) << ",\"mxcsr\":" << fp.saved
            << ",\"stats\":" << summary << "}\n";
        require(static_cast<bool>(records_) && static_cast<bool>(tensors_), "Cannot write I2 boundary");
    } catch (const std::exception&) { fail("I2 boundary failed"); }
    return good_;
}
bool Session::finish(const Capture& capture, const int8_t* w, const QuantDescriptor* s,
    const float* live_input, const float* actual, uint32_t rows, const PreparedW8A8Input* prepared) noexcept {
    dot_sink = nullptr;
    if (!active_) return true;
    if (!good_) return false;
    FPGuard fp;
    try {
        const uint32_t cols = static_cast<uint32_t>(capture.input.size());
        require(cols != 0, "Missing activation snapshot");
        require((capture.mxcsr & 0xffc0U) == (fp.saved & 0xffc0U), "FP control changed during production GEMV");
        const auto input_changed = mismatches(capture.input.data(), live_input, cols);
        auto e = Evaluate(capture.input.data(), cols, rows, w, s, actual, mode_, prepared,
                          capture.dots.empty() ? nullptr : capture.dots.data());
        const uint32_t blocks = cols/32U + (cols%32U != 0U);
        const uint32_t selected[2] = {0, e.worst_row};
        const uint32_t selected_count = e.worst_row == 0 ? 1U : 2U;
        const std::string replay = "gemv-"+std::to_string(sequence_++)+".i2r";
        std::ofstream file(directory_/replay, std::ios::out|std::ios::binary);
        const char magic[8] = {'M','I','C','I','2','R','1',0};
        const uint32_t header[7] = {cols, selected_count, rows, static_cast<uint32_t>(mode_), token_, position_, fp.saved};
        write_values(file, magic, 8); write_values(file, header, 7);
        write_values(file, capture.input.data(), cols);
        write_values(file, prepared ? prepared->values() : e.codes.data(), cols);
        write_values(file, prepared ? prepared->scales() : e.scales.data(), blocks);
        for (uint32_t i = 0; i < selected_count; ++i) {
            const uint32_t row = selected[i];
            write_values(file, &row, 1);
            write_values(file, w+static_cast<size_t>(row)*cols, cols);
            write_values(file, s+static_cast<size_t>(row)*blocks, blocks);
            write_values(file, actual+row, 1);
        }
        file.close(); require(static_cast<bool>(file), "Cannot write replay snapshot");
        records_ << "{\"type\":\"gemv\",\"token\":" << token_ << ",\"position\":" << position_
            << ",\"layer\":" << layer_ << ",\"operation\":\"" << operation_ << "\",\"rows\":" << rows
            << ",\"cols\":" << cols << ",\"row_begin\":0,\"row_end\":" << rows
            << ",\"blocks\":" << blocks << ",\"tail\":" << cols%32U << ",\"mxcsr_before\":" << capture.mxcsr
            << ",\"mxcsr_after\":" << fp.saved << ",\"input_version\":\"" << key(layer_, operation_)+".input"
            << "\",\"output_version\":\"" << key(layer_, operation_)+".output" << "\",\"input_changed\":" << input_changed
            << ",\"replay\":\"" << replay << "\",\"replay_rows\":[0";
        if (selected_count == 2) records_ << ',' << e.worst_row;
        records_ << "],\"evaluation\":" << e.json << "}\n";
        boundary(layer_, (operation_+".input").c_str(), capture.input.data(), cols);
        boundary(layer_, (operation_+".output").c_str(), actual, rows);
        // Retain local F64/I1 outputs for downstream isolation from propagation.
        boundary(layer_, (operation_+".local_f64").c_str(), e.f64.data(), rows);
        boundary(layer_, (operation_+".local_i1").c_str(), e.oracle.data(), rows);
        require(e.ok && input_changed == 0, "I2 same-input implementation mismatch");
        require(static_cast<bool>(records_), "Cannot write I2 record");
    } catch (const std::exception& e) { fail(e.what()); }
    return good_;
}
bool Session::complete() noexcept {
    if (!active_) return true;
    if (good_) {
        records_ << "{\"type\":\"complete\",\"gemv_count\":" << sequence_ << "}\n";
        records_.flush(); tensors_.flush();
        good_ = static_cast<bool>(records_) && static_cast<bool>(tensors_);
        complete_ = good_;
    }
    return good_;
}

void Replay(const std::filesystem::path& input, const std::filesystem::path& output) {
    FPGuard fp;
    require(!std::filesystem::exists(output), "Replay refuses to overwrite output");
    std::ifstream stream(input, std::ios::binary);
    require(static_cast<bool>(stream), "Cannot read replay file");
    char magic[8]; uint32_t h[7];
    read_values(stream, magic, 8); read_values(stream, h, 7);
    require(std::memcmp(magic, "MICI2R1", 8) == 0, "Invalid replay magic");
    const uint32_t cols=h[0], rows=h[1], total=h[2], mode=h[3];
    require(cols && cols <= 1048576 && rows && rows <= 2 && rows <= total && mode <= 2, "Invalid replay dimensions");
    require((h[6] & 0xffc0U) == 0x1f80U, "Invalid saved FP control");
    require((_mm_getcsr() & 0xffc0U) == 0x1f80U, "Unsupported replay FP control");
    const uint32_t blocks=cols/32U+(cols%32U != 0U);
    std::vector<float> x(cols), actual(rows);
    std::vector<int8_t> codes(cols), w(static_cast<size_t>(rows)*cols);
    std::vector<double> scales(blocks);
    std::vector<QuantDescriptor> descriptors(static_cast<size_t>(rows)*blocks);
    std::vector<uint32_t> ids(rows);
    read_values(stream, x.data(), cols); read_values(stream, codes.data(), cols); read_values(stream, scales.data(), blocks);
    for (uint32_t i=0;i<rows;++i) {
        read_values(stream, &ids[i], 1); require(ids[i] < total, "Invalid saved row");
        read_values(stream, w.data()+static_cast<size_t>(i)*cols, cols);
        read_values(stream, descriptors.data()+static_cast<size_t>(i)*blocks, blocks);
        read_values(stream, &actual[i], 1);
    }
    require(stream.peek() == std::char_traits<char>::eof(), "Trailing replay bytes");
    auto e=Evaluate(x.data(), cols, rows, w.data(), descriptors.data(), actual.data(), static_cast<int>(mode));
    const auto code_bad=mismatches(codes.data(),e.codes.data(),cols);
    const auto scale_bad=mismatches(scales.data(),e.scales.data(),blocks);
    std::ofstream out(output);
    out << "{\"schema\":1,\"scope\":\"selected rows, full activation; no forward pass\",\"token\":" << h[4]
        << ",\"position\":" << h[5] << ",\"saved_codes_mismatched\":" << code_bad
        << ",\"saved_scales_mismatched\":" << scale_bad << ",\"source_rows\":[";
    for (uint32_t i=0;i<rows;++i) { if(i) out << ','; out << ids[i]; }
    out << "],\"evaluation\":" << e.json << "}\n";
    out.close(); require(static_cast<bool>(out), "Cannot write replay result");
    require(e.ok && code_bad == 0 && scale_bad == 0, "Replay mismatch");
}
} // namespace micius::i2

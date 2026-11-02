#include "app/BackendInfo.h"
#include "micius_runtime.h"

#include <algorithm>
#include <atomic>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <pthread.h>
#include <sched.h>
#endif

#if defined(_MSC_VER)
#define MICIUS_NOINLINE __declspec(noinline)
#elif defined(__GNUC__) || defined(__clang__)
#define MICIUS_NOINLINE __attribute__((noinline))
#else
#define MICIUS_NOINLINE
#endif

namespace {

using Clock = std::chrono::steady_clock;

enum class RunMode {
    Throughput,
    Churn,
    Contracts
};

struct Options {
    RunMode mode = RunMode::Throughput;
    uint32_t external_threads = 1;
    uint64_t steps = 4096;
    uint64_t warmup_steps = 64;
    uint64_t verify_every = 256;
    uint32_t reset_interval = 0;
    uint32_t cpu_id = 0;

    uint32_t hidden_dim = 128;
    uint32_t vocab_size = 1024;
    uint32_t max_seq_len = 512;
    uint32_t kv_pool_size_mb = 4;
    uint32_t num_layers = 2;
    uint32_t num_heads = 4;
    uint32_t num_kv_heads = 2;
    uint32_t intermediate_dim = 256;
    uint64_t seed = UINT64_C(0x4D4943495553);

    bool steps_explicit = false;
    bool warmup_explicit = false;
    bool context_explicit = false;
    bool reset_explicit = false;
    bool cpu_explicit = false;
    bool show_help = false;
};

const char* ModeName(RunMode mode) noexcept
{
    switch (mode) {
        case RunMode::Throughput: return "throughput";
        case RunMode::Churn: return "churn";
        case RunMode::Contracts: return "contracts";
    }
    return "unknown";
}

int CurrentProcessor() noexcept
{
#if defined(_WIN32)
    return static_cast<int>(GetCurrentProcessorNumber());
#elif defined(__linux__)
    return sched_getcpu();
#else
    return -1;
#endif
}

bool PinCurrentThread(uint32_t cpu_id, int* observed_cpu, std::string* error)
{
    if (!observed_cpu || !error) return false;
#if defined(_WIN32)
    constexpr uint32_t kAffinityBits =
        static_cast<uint32_t>(std::numeric_limits<DWORD_PTR>::digits);
    if (cpu_id >= kAffinityBits) {
        *error = "--cpu exceeds the current Windows processor group mask";
        return false;
    }
    const DWORD_PTR mask = DWORD_PTR{1} << cpu_id;
    if (SetThreadAffinityMask(GetCurrentThread(), mask) == 0) {
        *error = "SetThreadAffinityMask failed with Win32 error " +
                 std::to_string(GetLastError());
        return false;
    }
#elif defined(__linux__)
    if (cpu_id >= static_cast<uint32_t>(CPU_SETSIZE)) {
        *error = "--cpu exceeds CPU_SETSIZE";
        return false;
    }
    cpu_set_t affinity;
    CPU_ZERO(&affinity);
    CPU_SET(static_cast<int>(cpu_id), &affinity);
    const int status = pthread_setaffinity_np(pthread_self(), sizeof(affinity), &affinity);
    if (status != 0) {
        *error = "pthread_setaffinity_np failed: " + std::string(std::strerror(status));
        return false;
    }
#else
    (void)cpu_id;
    *error = "--cpu is unsupported on this platform";
    return false;
#endif
    for (uint32_t attempt = 0; attempt < 1024U; ++attempt) {
        *observed_cpu = CurrentProcessor();
        if (*observed_cpu >= 0 && static_cast<uint32_t>(*observed_cpu) == cpu_id) {
            return true;
        }
        std::this_thread::yield();
    }
    *error = "thread affinity request did not take effect";
    return false;
}

std::string LastApiError(const char* operation)
{
    std::ostringstream stream;
    stream << operation << " failed (code="
           << static_cast<int>(Micius_GetLastErrorCode()) << "): ";
    const char* message = Micius_GetLastError();
    stream << (message && message[0] != '\0' ? message : "no diagnostic");
    return stream.str();
}

uint64_t ParseUnsigned(std::string_view text, const char* option)
{
    uint64_t value = 0;
    const char* begin = text.data();
    const char* end = begin + text.size();
    const auto parsed = std::from_chars(begin, end, value);
    if (parsed.ec != std::errc{} || parsed.ptr != end) {
        throw std::invalid_argument(std::string(option) + " requires an unsigned integer");
    }
    return value;
}

uint32_t ParseU32(std::string_view text, const char* option)
{
    const uint64_t value = ParseUnsigned(text, option);
    if (value > std::numeric_limits<uint32_t>::max()) {
        throw std::invalid_argument(std::string(option) + " exceeds uint32_t");
    }
    return static_cast<uint32_t>(value);
}

void PrintUsage(const char* executable)
{
    std::cout
        << "Usage: " << executable << " [options]\n"
        << "\n"
        << "Modes:\n"
        << "  --mode throughput   Reference decode hot path (default)\n"
        << "  --mode churn        Decode with frequent full-state resets\n"
        << "  --mode contracts    Transactional bind/error/lifecycle pressure\n"
        << "\n"
        << "Load controls:\n"
        << "  --threads N         Independent engine contexts (default 1)\n"
        << "  --steps N           Decode steps per thread, or contract cycles\n"
        << "  --warmup N          Untimed decode steps per thread\n"
        << "  --verify-every N    Timed logits/KV verification period; 0 disables\n"
        << "  --reset-interval N  Tokens between resets; must not exceed context\n"
        << "  --cpu N             Pin a single external worker to logical CPU N\n"
        << "\n"
        << "Synthetic model:\n"
        << "  --hidden N          Hidden dimension, divisible by 32\n"
        << "  --vocab N           Vocabulary size\n"
        << "  --context N         Maximum sequence length\n"
        << "  --pool-mb N         Per-context heterogeneous pool size\n"
        << "  --layers N          Transformer layers\n"
        << "  --heads N           Query heads\n"
        << "  --kv-heads N        KV heads; must divide query heads\n"
        << "  --intermediate N    FFN dimension, divisible by 32\n"
        << "  --seed N            Deterministic model seed\n"
        << "  --help              Show this text\n"
        << "\n"
        << "VTune examples:\n"
        << "  " << executable << " --mode throughput --steps 20000 --threads 1\n"
        << "  " << executable << " --mode churn --steps 10000 --reset-interval 8\n"
        << "  " << executable << " --mode contracts --steps 200 --threads 4\n";
}

Options ParseOptions(int argc, char** argv)
{
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string_view argument(argv[i]);
        if (argument == "--help") {
            options.show_help = true;
            continue;
        }

        if (i + 1 >= argc) {
            throw std::invalid_argument(std::string(argument) + " requires a value");
        }
        const std::string_view value(argv[++i]);

        if (argument == "--mode") {
            if (value == "throughput") options.mode = RunMode::Throughput;
            else if (value == "churn") options.mode = RunMode::Churn;
            else if (value == "contracts") options.mode = RunMode::Contracts;
            else throw std::invalid_argument("--mode must be throughput, churn, or contracts");
        } else if (argument == "--threads") {
            options.external_threads = ParseU32(value, "--threads");
        } else if (argument == "--steps") {
            options.steps = ParseUnsigned(value, "--steps");
            options.steps_explicit = true;
        } else if (argument == "--warmup") {
            options.warmup_steps = ParseUnsigned(value, "--warmup");
            options.warmup_explicit = true;
        } else if (argument == "--verify-every") {
            options.verify_every = ParseUnsigned(value, "--verify-every");
        } else if (argument == "--reset-interval") {
            options.reset_interval = ParseU32(value, "--reset-interval");
            options.reset_explicit = true;
        } else if (argument == "--cpu") {
            options.cpu_id = ParseU32(value, "--cpu");
            options.cpu_explicit = true;
        } else if (argument == "--hidden") {
            options.hidden_dim = ParseU32(value, "--hidden");
        } else if (argument == "--vocab") {
            options.vocab_size = ParseU32(value, "--vocab");
        } else if (argument == "--context") {
            options.max_seq_len = ParseU32(value, "--context");
            options.context_explicit = true;
        } else if (argument == "--pool-mb") {
            options.kv_pool_size_mb = ParseU32(value, "--pool-mb");
        } else if (argument == "--layers") {
            options.num_layers = ParseU32(value, "--layers");
        } else if (argument == "--heads") {
            options.num_heads = ParseU32(value, "--heads");
        } else if (argument == "--kv-heads") {
            options.num_kv_heads = ParseU32(value, "--kv-heads");
        } else if (argument == "--intermediate") {
            options.intermediate_dim = ParseU32(value, "--intermediate");
        } else if (argument == "--seed") {
            options.seed = ParseUnsigned(value, "--seed");
        } else {
            throw std::invalid_argument("unknown option: " + std::string(argument));
        }
    }

    if (options.mode == RunMode::Contracts) {
        if (!options.steps_explicit) options.steps = 64;
        if (!options.warmup_explicit) options.warmup_steps = 0;
        if (!options.context_explicit) options.max_seq_len = 32;
    }
    if (options.mode == RunMode::Churn && !options.reset_explicit) {
        options.reset_interval = 8;
    }
    if (options.reset_interval == 0) {
        options.reset_interval = options.max_seq_len;
    }

    if (options.external_threads == 0 || options.external_threads > 256) {
        throw std::invalid_argument("--threads must be in [1, 256]");
    }
    if (options.cpu_explicit && options.external_threads != 1U) {
        throw std::invalid_argument("--cpu currently requires --threads 1");
    }
    if (options.steps == 0) {
        throw std::invalid_argument("--steps must be non-zero");
    }
    if (options.hidden_dim == 0 || options.hidden_dim % 32U != 0) {
        throw std::invalid_argument("--hidden must be non-zero and divisible by 32");
    }
    if (options.intermediate_dim == 0 || options.intermediate_dim % 32U != 0) {
        throw std::invalid_argument("--intermediate must be non-zero and divisible by 32");
    }
    if (static_cast<uint64_t>(options.intermediate_dim) >
        static_cast<uint64_t>(options.hidden_dim) * UINT64_C(4)) {
        throw std::invalid_argument("--intermediate exceeds the engine limit of 4 * hidden");
    }
    if (options.vocab_size < 2 || options.max_seq_len == 0 ||
        options.kv_pool_size_mb == 0 || options.num_layers == 0 ||
        options.num_heads == 0 || options.num_kv_heads == 0) {
        throw std::invalid_argument("model sizes, context, pool, layers, and heads must be non-zero");
    }
    if (options.hidden_dim % options.num_heads != 0) {
        throw std::invalid_argument("--hidden must be divisible by --heads");
    }
    const uint32_t head_dim = options.hidden_dim / options.num_heads;
    if ((head_dim & 1U) != 0) {
        throw std::invalid_argument("derived head_dim must be even");
    }
    if (options.num_heads % options.num_kv_heads != 0) {
        throw std::invalid_argument("--kv-heads must divide --heads");
    }
    if (options.reset_interval > options.max_seq_len) {
        throw std::invalid_argument("--reset-interval cannot exceed --context");
    }
    return options;
}

size_t CheckedElements(uint64_t rows, uint64_t columns)
{
    if (rows == 0 || columns == 0 ||
        rows > std::numeric_limits<size_t>::max() / columns) {
        throw std::overflow_error("synthetic tensor size overflows size_t");
    }
    return static_cast<size_t>(rows * columns);
}

class SplitMix64 {
public:
    explicit SplitMix64(uint64_t seed) noexcept : state_(seed) {}

    uint64_t Next() noexcept
    {
        uint64_t value = (state_ += UINT64_C(0x9E3779B97F4A7C15));
        value = (value ^ (value >> 30U)) * UINT64_C(0xBF58476D1CE4E5B9);
        value = (value ^ (value >> 27U)) * UINT64_C(0x94D049BB133111EB);
        return value ^ (value >> 31U);
    }

private:
    uint64_t state_;
};

struct QuantMatrix {
    uint32_t rows = 0;
    uint32_t columns = 0;
    std::vector<int8_t> weights;
    std::vector<MiciusQ8Block32Scale> scales;

    void Initialize(uint32_t row_count, uint32_t column_count, SplitMix64& random)
    {
        if (column_count == 0 || column_count % 32U != 0) {
            throw std::invalid_argument("Q8 matrix columns must be divisible by 32");
        }
        rows = row_count;
        columns = column_count;
        weights.resize(CheckedElements(rows, columns));
        scales.resize(CheckedElements(rows, columns / 32U));

        for (int8_t& weight : weights) {
            const int32_t value = static_cast<int32_t>(random.Next() % UINT64_C(7)) - 3;
            weight = static_cast<int8_t>(value);
        }
        for (MiciusQ8Block32Scale& descriptor : scales) {
            const uint32_t bucket = static_cast<uint32_t>(random.Next() % UINT64_C(5));
            descriptor = MiciusQ8Block32Scale{
                0.0065f + static_cast<float>(bucket) * 0.00025f,
                0,
                {0, 0}};
        }
    }

    size_t ByteSize() const noexcept
    {
        return weights.size() * sizeof(int8_t) +
               scales.size() * sizeof(MiciusQ8Block32Scale);
    }
};

struct LayerStorage {
    QuantMatrix query;
    QuantMatrix key;
    QuantMatrix value;
    QuantMatrix output;
    QuantMatrix gate;
    QuantMatrix up;
    QuantMatrix down;
    std::vector<float> attention_norm;
    std::vector<float> ffn_norm;
};

class SyntheticModel {
public:
    explicit SyntheticModel(const Options& options)
    {
        if (!Micius_InitConfig(&config_, sizeof(config_))) {
            throw std::runtime_error(LastApiError("Micius_InitConfig"));
        }
        config_.hidden_dim = options.hidden_dim;
        config_.vocab_size = options.vocab_size;
        config_.max_seq_len = options.max_seq_len;
        config_.kv_pool_size_mb = options.kv_pool_size_mb;
        config_.worker_threads = 0;
        config_.num_layers = options.num_layers;
        config_.num_heads = options.num_heads;
        config_.num_kv_heads = options.num_kv_heads;
        config_.head_dim = options.hidden_dim / options.num_heads;
        config_.intermediate_dim = options.intermediate_dim;
        config_.execution_mode = MICIUS_EXECUTION_REFERENCE;
        config_.rope_dimension_count = config_.head_dim;

        SplitMix64 random(options.seed);
        embedding_.resize(CheckedElements(config_.vocab_size, config_.hidden_dim));
        final_norm_.resize(config_.hidden_dim);
        for (float& value : embedding_) {
            const int32_t bucket =
                static_cast<int32_t>(random.Next() % UINT64_C(2001)) - 1000;
            value = static_cast<float>(bucket) / 2000.0f;
        }
        FillNorm(final_norm_, random);
        output_.Initialize(config_.vocab_size, config_.hidden_dim, random);

        const uint32_t kv_dimension = config_.num_kv_heads * config_.head_dim;
        layers_.resize(static_cast<size_t>(config_.num_layers));
        for (LayerStorage& layer : layers_) {
            layer.query.Initialize(config_.hidden_dim, config_.hidden_dim, random);
            layer.key.Initialize(kv_dimension, config_.hidden_dim, random);
            layer.value.Initialize(kv_dimension, config_.hidden_dim, random);
            layer.output.Initialize(config_.hidden_dim, config_.hidden_dim, random);
            layer.gate.Initialize(config_.intermediate_dim, config_.hidden_dim, random);
            layer.up.Initialize(config_.intermediate_dim, config_.hidden_dim, random);
            layer.down.Initialize(config_.hidden_dim, config_.intermediate_dim, random);
            layer.attention_norm.resize(config_.hidden_dim);
            layer.ffn_norm.resize(config_.hidden_dim);
            FillNorm(layer.attention_norm, random);
            FillNorm(layer.ffn_norm, random);
        }

        const size_t tensor_count = static_cast<size_t>(config_.num_layers) * 16U + 4U;
        names_.reserve(tensor_count);
        entries_.reserve(tensor_count);
        AddF32(
            "token_embd.weight.f32",
            embedding_,
            config_.vocab_size,
            config_.hidden_dim);
        AddF32("output_norm.weight", final_norm_, config_.hidden_dim, 0);
        AddQ8("output", output_);
        for (uint32_t layer_index = 0; layer_index < config_.num_layers; ++layer_index) {
            LayerStorage& layer = layers_[layer_index];
            const std::string prefix = "blk." + std::to_string(layer_index) + ".";
            AddQ8(prefix + "attn_q", layer.query);
            AddQ8(prefix + "attn_k", layer.key);
            AddQ8(prefix + "attn_v", layer.value);
            AddQ8(prefix + "attn_output", layer.output);
            AddQ8(prefix + "ffn_gate", layer.gate);
            AddQ8(prefix + "ffn_up", layer.up);
            AddQ8(prefix + "ffn_down", layer.down);
            AddF32(prefix + "attn_norm.weight", layer.attention_norm, config_.hidden_dim, 0);
            AddF32(prefix + "ffn_norm.weight", layer.ffn_norm, config_.hidden_dim, 0);
        }
        if (entries_.size() != tensor_count) {
            throw std::logic_error("synthetic tensor manifest count is inconsistent");
        }
    }

    const MiciusConfig& Config() const noexcept { return config_; }
    const std::vector<MiciusTensorEntryV1>& Entries() const noexcept { return entries_; }
    size_t OutputWeightEntryIndex() const noexcept { return output_weight_entry_index_; }

    size_t ByteSize() const noexcept
    {
        size_t bytes =
            embedding_.size() * sizeof(float) +
            final_norm_.size() * sizeof(float) +
            output_.ByteSize();
        for (const LayerStorage& layer : layers_) {
            bytes += layer.query.ByteSize() + layer.key.ByteSize() +
                     layer.value.ByteSize() + layer.output.ByteSize() +
                     layer.gate.ByteSize() + layer.up.ByteSize() +
                     layer.down.ByteSize() +
                     (layer.attention_norm.size() + layer.ffn_norm.size()) * sizeof(float);
        }
        return bytes;
    }

private:
    static void FillNorm(std::vector<float>& values, SplitMix64& random)
    {
        for (float& value : values) {
            const uint32_t bucket = static_cast<uint32_t>(random.Next() % UINT64_C(17));
            value = 0.96f + static_cast<float>(bucket) * 0.005f;
        }
    }

    const char* StoreName(std::string name)
    {
        names_.push_back(std::move(name));
        return names_.back().c_str();
    }

    void AddF32(
        std::string name,
        const std::vector<float>& values,
        uint64_t rows,
        uint64_t columns)
    {
        MiciusTensorEntryV1 entry{};
        entry.tensor_name = StoreName(std::move(name));
        entry.tensor_data = values.data();
        entry.byte_size = values.size() * sizeof(float);
        entry.data_type = MICIUS_TENSOR_F32;
        entry.rank = columns == 0 ? 1U : 2U;
        entry.dimensions[0] = rows;
        entry.dimensions[1] = columns;
        entry.row_stride_bytes =
            columns == 0 ? 0 : static_cast<size_t>(columns) * sizeof(float);
        entries_.push_back(entry);
    }

    void AddQ8(const std::string& base_name, const QuantMatrix& matrix)
    {
        MiciusTensorEntryV1 weight_entry{};
        weight_entry.tensor_name = StoreName(base_name + ".weight");
        weight_entry.tensor_data = matrix.weights.data();
        weight_entry.byte_size = matrix.weights.size();
        weight_entry.data_type = MICIUS_TENSOR_Q8_32_WEIGHT;
        weight_entry.rank = 2;
        weight_entry.dimensions[0] = matrix.rows;
        weight_entry.dimensions[1] = matrix.columns;
        weight_entry.row_stride_bytes = matrix.columns;
        if (base_name == "output") output_weight_entry_index_ = entries_.size();
        entries_.push_back(weight_entry);

        MiciusTensorEntryV1 scale_entry{};
        scale_entry.tensor_name = StoreName(base_name + ".scales");
        scale_entry.tensor_data = matrix.scales.data();
        scale_entry.byte_size = matrix.scales.size() * sizeof(MiciusQ8Block32Scale);
        scale_entry.data_type = MICIUS_TENSOR_Q8_32_SCALE;
        scale_entry.rank = 2;
        scale_entry.dimensions[0] = matrix.rows;
        scale_entry.dimensions[1] = matrix.columns / 32U;
        scale_entry.row_stride_bytes =
            static_cast<size_t>(matrix.columns / 32U) * sizeof(MiciusQ8Block32Scale);
        entries_.push_back(scale_entry);
    }

    MiciusConfig config_{};
    std::vector<float> embedding_;
    std::vector<float> final_norm_;
    QuantMatrix output_;
    std::vector<LayerStorage> layers_;
    std::vector<std::string> names_;
    std::vector<MiciusTensorEntryV1> entries_;
    size_t output_weight_entry_index_ = 0;
};

struct ContextDeleter {
    void operator()(MiciusContext* context) const noexcept
    {
        Micius_DestroyEngine(context);
    }
};

using ContextPtr = std::unique_ptr<MiciusContext, ContextDeleter>;

ContextPtr CreateBoundContext(const SyntheticModel& model)
{
    const MiciusConfig& config = model.Config();
    ContextPtr context(Micius_CreateEngineV1(
        &config,
        sizeof(config),
        MICIUS_ABI_VERSION));
    if (!context) {
        throw std::runtime_error(LastApiError("Micius_CreateEngineV1"));
    }
    const auto& entries = model.Entries();
    if (entries.size() > std::numeric_limits<uint32_t>::max()) {
        throw std::overflow_error("tensor manifest count exceeds uint32_t");
    }
    if (!Micius_BindModelTensorsV1(
            context.get(),
            entries.data(),
            static_cast<uint32_t>(entries.size()),
            sizeof(MiciusTensorEntryV1),
            MICIUS_ABI_VERSION)) {
        throw std::runtime_error(LastApiError("Micius_BindModelTensorsV1"));
    }
    return context;
}

uint64_t MixHash(uint64_t hash, uint64_t value) noexcept
{
    hash ^= value + UINT64_C(0x9E3779B97F4A7C15) + (hash << 6U) + (hash >> 2U);
    return hash;
}

struct WorkerScratch {
    explicit WorkerScratch(const MiciusConfig& config)
        : logits(config.vocab_size),
          key(static_cast<size_t>(config.num_kv_heads) * config.head_dim),
          value(static_cast<size_t>(config.num_kv_heads) * config.head_dim)
    {
    }

    std::vector<float> logits;
    std::vector<float> key;
    std::vector<float> value;
};

struct WorkerResult {
    bool success = true;
    std::string error;
    uint64_t completed_units = 0;
    uint64_t successful_decodes = 0;
    uint64_t resets = 0;
    uint64_t verifications = 0;
    uint64_t checksum = UINT64_C(0xCBF29CE484222325);
    double elapsed_seconds = 0.0;
};

void SetFailure(WorkerResult& result, std::string message)
{
    if (result.success) {
        result.success = false;
        result.error = std::move(message);
    }
}

bool VerifyCurrentState(
    MiciusContext* context,
    const MiciusConfig& config,
    uint32_t expected_token,
    WorkerScratch& scratch,
    WorkerResult& result)
{
    const uint32_t sequence_length = Micius_GetSequenceLengthV1(context);
    if (Micius_GetLastErrorCode() != MICIUS_ERROR_NONE) {
        SetFailure(result, LastApiError("Micius_GetSequenceLengthV1"));
        return false;
    }
    if (sequence_length == 0) {
        SetFailure(result, "verification requires at least one committed token");
        return false;
    }
    if (!Micius_CopyLastLogitsV1(context, scratch.logits.data(), scratch.logits.size())) {
        SetFailure(result, LastApiError("Micius_CopyLastLogitsV1"));
        return false;
    }

    float best_logit = -std::numeric_limits<float>::infinity();
    uint32_t best_token = 0;
    for (uint32_t token = 0; token < config.vocab_size; ++token) {
        const float logit = scratch.logits[token];
        if (!std::isfinite(logit)) {
            SetFailure(result, "verification observed a non-finite logit");
            return false;
        }
        result.checksum = MixHash(result.checksum, std::bit_cast<uint32_t>(logit));
        if (logit > best_logit) {
            best_logit = logit;
            best_token = token;
        }
    }
    if (best_token != expected_token) {
        SetFailure(result, "public logits argmax disagrees with DecodeOneV1");
        return false;
    }

    if (!Micius_CopyReferenceKVV1(
            context,
            config.num_layers - 1U,
            sequence_length - 1U,
            scratch.key.data(),
            scratch.value.data(),
            scratch.key.size())) {
        SetFailure(result, LastApiError("Micius_CopyReferenceKVV1"));
        return false;
    }
    for (size_t index = 0; index < scratch.key.size(); ++index) {
        if (!std::isfinite(scratch.key[index]) || !std::isfinite(scratch.value[index])) {
            SetFailure(result, "verification observed non-finite reference KV data");
            return false;
        }
        result.checksum = MixHash(
            result.checksum,
            std::bit_cast<uint32_t>(scratch.key[index]));
        result.checksum = MixHash(
            result.checksum,
            std::bit_cast<uint32_t>(scratch.value[index]));
    }
    ++result.verifications;
    return true;
}

bool BasicPreflight(
    MiciusContext* context,
    const SyntheticModel& model,
    uint32_t start_token,
    std::string& error)
{
    const MiciusConfig& config = model.Config();
    if (Micius_GetWorkerCount(context) != 0 ||
        Micius_GetLastErrorCode() != MICIUS_ERROR_NONE) {
        error = "reference context unexpectedly exposes speculative workers";
        return false;
    }
    if (!Micius_ResetEngineV1(context)) {
        error = LastApiError("Micius_ResetEngineV1");
        return false;
    }

    uint32_t output_token = UINT32_C(0xFFFFFFFF);
    if (Micius_DecodeOneV1(context, start_token, &output_token) != MICIUS_STATE_YIELD_TOKEN) {
        error = LastApiError("Micius_DecodeOneV1 preflight");
        return false;
    }
    WorkerScratch scratch(config);
    WorkerResult result;
    if (!VerifyCurrentState(context, config, output_token, scratch, result)) {
        error = result.error;
        return false;
    }
    if (!Micius_ResetEngineV1(context)) {
        error = LastApiError("Micius_ResetEngineV1 preflight");
        return false;
    }
    if (Micius_CopyLastLogitsV1(context, scratch.logits.data(), scratch.logits.size()) ||
        Micius_GetLastErrorCode() != MICIUS_ERROR_NOT_READY) {
        error = "reset did not invalidate the last-logits snapshot";
        return false;
    }
    return true;
}

bool ContractPreflight(
    MiciusContext* context,
    const SyntheticModel& model,
    uint32_t start_token,
    std::string& error)
{
    const MiciusConfig& config = model.Config();
    std::vector<MiciusTensorEntryV1> invalid_entries = model.Entries();
    invalid_entries[model.OutputWeightEntryIndex()].byte_size -= 1U;
    if (Micius_BindModelTensorsV1(
            context,
            invalid_entries.data(),
            static_cast<uint32_t>(invalid_entries.size()),
            sizeof(MiciusTensorEntryV1),
            MICIUS_ABI_VERSION) ||
        Micius_GetLastErrorCode() != MICIUS_ERROR_INVALID_ARGUMENT) {
        error = "malformed replacement was not rejected";
        return false;
    }

    uint32_t output_token = UINT32_C(0xFFFFFFFF);
    if (Micius_DecodeOneV1(context, start_token, &output_token) != MICIUS_STATE_YIELD_TOKEN) {
        error = "failed replacement destroyed the previously bound model";
        return false;
    }
    if (!Micius_ResetEngineV1(context)) {
        error = LastApiError("Micius_ResetEngineV1 contract preflight");
        return false;
    }

    for (uint32_t position = 0; position < config.max_seq_len; ++position) {
        const uint32_t token =
            static_cast<uint32_t>((static_cast<uint64_t>(start_token) + position) %
                                  config.vocab_size);
        if (Micius_PrefillTokenV1(context, token) != MICIUS_STATE_TOKEN_ACCEPTED) {
            error = LastApiError("Micius_PrefillTokenV1 context-fill preflight");
            return false;
        }
    }
    output_token = UINT32_C(0xA5A5A5A5);
    if (Micius_DecodeOneV1(context, start_token, &output_token) !=
            MICIUS_STATE_CONTEXT_FULL ||
        output_token != UINT32_C(0xA5A5A5A5) ||
        Micius_GetLastErrorCode() != MICIUS_ERROR_CONTEXT_FULL) {
        error = "context-full contract or output preservation failed";
        return false;
    }
    std::vector<float> logits(config.vocab_size);
    if (!Micius_CopyLastLogitsV1(context, logits.data(), logits.size())) {
        error = "context-full failure erased the last successful logits";
        return false;
    }
    if (!Micius_ResetEngineV1(context)) {
        error = LastApiError("Micius_ResetEngineV1 after context-full");
        return false;
    }
    return true;
}

MICIUS_NOINLINE WorkerResult RunDecodeLoop(
    MiciusContext* context,
    const MiciusConfig& config,
    uint64_t steps,
    uint32_t reset_interval,
    uint64_t verify_every,
    uint32_t start_token,
    WorkerScratch& scratch,
    const std::atomic<bool>& abort_requested)
{
    WorkerResult result;
    uint32_t current_token = start_token;
    uint32_t sequence_length = Micius_GetSequenceLengthV1(context);
    if (Micius_GetLastErrorCode() != MICIUS_ERROR_NONE) {
        SetFailure(result, LastApiError("Micius_GetSequenceLengthV1"));
        return result;
    }

    for (uint64_t step = 0; step < steps; ++step) {
        if (abort_requested.load(std::memory_order_relaxed)) break;
        if (sequence_length >= reset_interval) {
            if (!Micius_ResetEngineV1(context)) {
                SetFailure(result, LastApiError("Micius_ResetEngineV1"));
                break;
            }
            sequence_length = 0;
            ++result.resets;
        }

        uint32_t output_token = UINT32_C(0xFFFFFFFF);
        const MiciusStatus status =
            Micius_DecodeOneV1(context, current_token, &output_token);
        if (status != MICIUS_STATE_YIELD_TOKEN) {
            SetFailure(result, LastApiError("Micius_DecodeOneV1"));
            break;
        }
        ++sequence_length;
        ++result.completed_units;
        ++result.successful_decodes;
        result.checksum = MixHash(result.checksum, output_token);
        current_token = output_token;

        if (verify_every != 0 && (step + 1U) % verify_every == 0) {
            if (!VerifyCurrentState(
                    context,
                    config,
                    output_token,
                    scratch,
                    result)) {
                break;
            }
        }
    }

    if (result.success) {
        const uint32_t public_length = Micius_GetSequenceLengthV1(context);
        if (Micius_GetLastErrorCode() != MICIUS_ERROR_NONE ||
            public_length != sequence_length) {
            SetFailure(result, "tracked sequence length disagrees with public runtime state");
        }
    }
    return result;
}

MICIUS_NOINLINE WorkerResult RunContractLoop(
    MiciusContext* context,
    const SyntheticModel& model,
    uint64_t cycles,
    uint32_t start_token,
    const std::atomic<bool>& abort_requested)
{
    WorkerResult result;
    const auto& valid_entries = model.Entries();
    std::vector<MiciusTensorEntryV1> invalid_entries = valid_entries;
    invalid_entries[model.OutputWeightEntryIndex()].byte_size -= 1U;
    const uint32_t entry_count = static_cast<uint32_t>(valid_entries.size());

    for (uint64_t cycle = 0; cycle < cycles; ++cycle) {
        if (abort_requested.load(std::memory_order_relaxed)) break;
        const bool use_short_stride = (cycle & UINT64_C(1)) != 0;
        const MiciusTensorEntryV1* candidate =
            use_short_stride ? valid_entries.data() : invalid_entries.data();
        const size_t entry_size =
            use_short_stride ? sizeof(MiciusTensorEntryV1) - 1U
                             : sizeof(MiciusTensorEntryV1);
        if (Micius_BindModelTensorsV1(
                context,
                candidate,
                entry_count,
                entry_size,
                MICIUS_ABI_VERSION) ||
            Micius_GetLastErrorCode() != MICIUS_ERROR_INVALID_ARGUMENT) {
            SetFailure(result, "contract loop accepted a malformed replacement");
            break;
        }

        uint32_t first_output = UINT32_C(0xFFFFFFFF);
        if (Micius_DecodeOneV1(context, start_token, &first_output) !=
            MICIUS_STATE_YIELD_TOKEN) {
            SetFailure(result, "transactional bind failure destroyed the active model");
            break;
        }
        ++result.successful_decodes;
        result.checksum = MixHash(result.checksum, first_output);
        if (!Micius_ResetEngineV1(context)) {
            SetFailure(result, LastApiError("Micius_ResetEngineV1 contract loop"));
            break;
        }
        ++result.resets;

        if (!Micius_BindModelTensorsV1(
                context,
                valid_entries.data(),
                entry_count,
                sizeof(MiciusTensorEntryV1),
                MICIUS_ABI_VERSION)) {
            SetFailure(result, LastApiError("valid transactional rebind"));
            break;
        }
        uint32_t second_output = UINT32_C(0xFFFFFFFF);
        if (Micius_DecodeOneV1(context, start_token, &second_output) !=
            MICIUS_STATE_YIELD_TOKEN) {
            SetFailure(result, LastApiError("post-rebind DecodeOneV1"));
            break;
        }
        ++result.successful_decodes;
        if (first_output != second_output) {
            SetFailure(result, "rebind changed deterministic output");
            break;
        }
        result.checksum = MixHash(result.checksum, second_output);
        if (!Micius_ResetEngineV1(context)) {
            SetFailure(result, LastApiError("final Micius_ResetEngineV1 contract loop"));
            break;
        }
        ++result.resets;
        ++result.completed_units;
    }
    return result;
}

struct AggregateResult {
    bool success = true;
    std::string error;
    uint64_t completed_units = 0;
    uint64_t successful_decodes = 0;
    uint64_t resets = 0;
    uint64_t verifications = 0;
    uint64_t checksum = 0;
    double wall_seconds = 0.0;
    double min_worker_seconds = std::numeric_limits<double>::infinity();
    double max_worker_seconds = 0.0;
    std::vector<int> worker_start_cpus;
};

AggregateResult RunWorkers(
    const Options& options,
    const SyntheticModel& model,
    std::vector<ContextPtr>& contexts)
{
    const MiciusConfig& config = model.Config();
    const uint32_t start_token = static_cast<uint32_t>(options.seed % config.vocab_size);
    const size_t thread_count = static_cast<size_t>(options.external_threads);
    std::vector<WorkerResult> results(thread_count);
    std::vector<std::thread> threads;
    threads.reserve(thread_count);
    std::atomic<uint32_t> ready{0};
    std::atomic<bool> begin{false};
    std::atomic<bool> abort_requested{false};
    std::vector<int> worker_start_cpus(thread_count, -1);

    for (uint32_t worker_index = 0;
         worker_index < options.external_threads;
         ++worker_index) {
        threads.emplace_back([&, worker_index]() {
            if (options.cpu_explicit) {
                std::string affinity_error;
                if (!PinCurrentThread(
                        options.cpu_id,
                        &worker_start_cpus[worker_index],
                        &affinity_error)) {
                    SetFailure(results[worker_index], std::move(affinity_error));
                    abort_requested.store(true, std::memory_order_release);
                }
            } else {
                worker_start_cpus[worker_index] = CurrentProcessor();
            }
            WorkerScratch scratch(config);
            WorkerResult warmup;
            if (results[worker_index].success &&
                options.mode != RunMode::Contracts && options.warmup_steps != 0) {
                warmup = RunDecodeLoop(
                    contexts[worker_index].get(),
                    config,
                    options.warmup_steps,
                    options.reset_interval,
                    0,
                    start_token,
                    scratch,
                    abort_requested);
                if (!warmup.success ||
                    !Micius_ResetEngineV1(contexts[worker_index].get())) {
                    results[worker_index] = std::move(warmup);
                    if (results[worker_index].success) {
                        SetFailure(
                            results[worker_index],
                            LastApiError("post-warmup Micius_ResetEngineV1"));
                    }
                    abort_requested.store(true, std::memory_order_release);
                }
            }

            MiciusRuntimeStats before{};
            if (results[worker_index].success) {
                Micius_GetRuntimeStats(contexts[worker_index].get(), &before);
                if (Micius_GetLastErrorCode() != MICIUS_ERROR_NONE) {
                    SetFailure(
                        results[worker_index],
                        LastApiError("Micius_GetRuntimeStats before workload"));
                    abort_requested.store(true, std::memory_order_release);
                }
            }

            ready.fetch_add(1U, std::memory_order_release);
            while (!begin.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            if (!results[worker_index].success ||
                abort_requested.load(std::memory_order_acquire)) {
                return;
            }

            const auto worker_start = Clock::now();
            WorkerResult timed;
            if (options.mode == RunMode::Contracts) {
                timed = RunContractLoop(
                    contexts[worker_index].get(),
                    model,
                    options.steps,
                    start_token,
                    abort_requested);
            } else {
                timed = RunDecodeLoop(
                    contexts[worker_index].get(),
                    config,
                    options.steps,
                    options.reset_interval,
                    options.verify_every,
                    start_token,
                    scratch,
                    abort_requested);
            }
            const auto worker_end = Clock::now();
            timed.elapsed_seconds =
                std::chrono::duration<double>(worker_end - worker_start).count();

            if (timed.success) {
                MiciusRuntimeStats after{};
                Micius_GetRuntimeStats(contexts[worker_index].get(), &after);
                const uint32_t generated_delta =
                    after.total_tokens_generated - before.total_tokens_generated;
                if (Micius_GetLastErrorCode() != MICIUS_ERROR_NONE ||
                    generated_delta != static_cast<uint32_t>(timed.successful_decodes)) {
                    SetFailure(timed, "runtime generated-token counter delta is inconsistent");
                }
            }
            if (!timed.success) {
                abort_requested.store(true, std::memory_order_release);
            }
            results[worker_index] = std::move(timed);
        });
    }

    while (ready.load(std::memory_order_acquire) < options.external_threads) {
        std::this_thread::yield();
    }
    std::cout << "VTUNE_ROI_BEGIN\n" << std::flush;
    const auto wall_start = Clock::now();
    begin.store(true, std::memory_order_release);
    for (std::thread& thread : threads) thread.join();
    const auto wall_end = Clock::now();
    std::cout << "VTUNE_ROI_END\n";

    AggregateResult aggregate;
    aggregate.worker_start_cpus = std::move(worker_start_cpus);
    aggregate.wall_seconds =
        std::chrono::duration<double>(wall_end - wall_start).count();
    bool has_reference_checksum = false;
    for (uint32_t worker_index = 0;
         worker_index < options.external_threads;
         ++worker_index) {
        const WorkerResult& result = results[worker_index];
        if (!result.success && aggregate.success) {
            aggregate.success = false;
            aggregate.error =
                "worker " + std::to_string(worker_index) + ": " + result.error;
        }
        aggregate.completed_units += result.completed_units;
        aggregate.successful_decodes += result.successful_decodes;
        aggregate.resets += result.resets;
        aggregate.verifications += result.verifications;
        aggregate.min_worker_seconds =
            std::min(aggregate.min_worker_seconds, result.elapsed_seconds);
        aggregate.max_worker_seconds =
            std::max(aggregate.max_worker_seconds, result.elapsed_seconds);
        if (result.success && !has_reference_checksum) {
            aggregate.checksum = result.checksum;
            has_reference_checksum = true;
        } else if (result.success && result.checksum != aggregate.checksum &&
                   aggregate.success) {
            aggregate.success = false;
            aggregate.error = "independent contexts produced different deterministic checksums";
        }
    }
    return aggregate;
}

void PrintConfiguration(const Options& options, const SyntheticModel& model, const MiciusContext* context)
{
    const MiciusConfig& config = model.Config();
    std::cout
        << "MICIUS reference stress sandbox\n"
        << "abi=0x" << std::hex << MICIUS_ABI_VERSION << std::dec << '\n'
        << "gemv_backend=" << micius::backend::GEMVBackend(context) << '\n'
#if defined(MICIUS_REFERENCE_GEMV_AVX2)
        << "gemv_fp_contract=off\n"
#else
        << "gemv_fp_contract=compiler_default_scalar_reference\n"
#endif
        << "gemv_input_preparation=" << micius::backend::InputPreparation() << '\n'

        << "gemv_scale_validation=" << micius::backend::ScaleValidation() << '\n'
        << "gemv_scale_storage=" << micius::backend::ScaleStorage(context) << '\n'
        << "attention_value_backend=" << micius::backend::AttentionValue() << '\n'
        << "attention_score_query_cache=" << micius::backend::QueryCache() << '\n'
        << "attention_score_time_tile=" << micius::backend::ScoreTile() << '\n'
        << "mode=" << ModeName(options.mode) << '\n'
        << "threads=" << options.external_threads << '\n'
        << "cpu_affinity="
        << (options.cpu_explicit ? static_cast<int64_t>(options.cpu_id) : int64_t{-1})
        << '\n'
        << "steps_per_thread=" << options.steps << '\n'
        << "warmup_per_thread=" << options.warmup_steps << '\n'
        << "verify_every=" << options.verify_every << '\n'
        << "reset_interval=" << options.reset_interval << '\n'
        << "hidden=" << config.hidden_dim << '\n'
        << "vocab=" << config.vocab_size << '\n'
        << "context=" << config.max_seq_len << '\n'
        << "pool_mb_per_context=" << config.kv_pool_size_mb << '\n'
        << "layers=" << config.num_layers << '\n'
        << "heads=" << config.num_heads << '\n'
        << "kv_heads=" << config.num_kv_heads << '\n'
        << "intermediate=" << config.intermediate_dim << '\n'
        << "model_bytes=" << model.ByteSize() << '\n';
}

void PrintResult(const Options& options, const AggregateResult& result)
{
    const double unit_rate =
        result.wall_seconds > 0.0
            ? static_cast<double>(result.completed_units) / result.wall_seconds
            : 0.0;
    const double decode_rate =
        result.wall_seconds > 0.0
            ? static_cast<double>(result.successful_decodes) / result.wall_seconds
            : 0.0;

    std::cout
        << std::fixed << std::setprecision(6)
        << "status=" << (result.success ? "PASS" : "FAIL") << '\n'
        << "wall_seconds=" << result.wall_seconds << '\n'
        << "completed_" << (options.mode == RunMode::Contracts ? "cycles" : "steps")
        << '=' << result.completed_units << '\n'
        << (options.mode == RunMode::Contracts ? "cycles_per_second=" : "steps_per_second=")
        << unit_rate << '\n'
        << "successful_decodes=" << result.successful_decodes << '\n'
        << "decodes_per_second=" << decode_rate << '\n'
        << "resets=" << result.resets << '\n'
        << "verifications=" << result.verifications << '\n'
        << "worker_seconds_min=" << result.min_worker_seconds << '\n'
        << "worker_seconds_max=" << result.max_worker_seconds << '\n';
    for (size_t worker = 0; worker < result.worker_start_cpus.size(); ++worker) {
        std::cout << "worker_start_cpu_" << worker << '='
                  << result.worker_start_cpus[worker] << '\n';
    }
    std::cout << "checksum=0x" << std::hex << result.checksum << std::dec << '\n';
    if (!result.success) std::cout << "error=" << result.error << '\n';
}

} // namespace

int main(int argc, char** argv)
{
    try {
        const Options options = ParseOptions(argc, argv);
        if (options.show_help) {
            PrintUsage(argv[0]);
            return 0;
        }

        SyntheticModel model(options);

        std::vector<ContextPtr> contexts;
        contexts.reserve(options.external_threads);
        const uint32_t start_token =
            static_cast<uint32_t>(options.seed % model.Config().vocab_size);
        for (uint32_t index = 0; index < options.external_threads; ++index) {
            contexts.push_back(CreateBoundContext(model));
            std::string error;
            if (!BasicPreflight(contexts.back().get(), model, start_token, error)) {
                std::cerr << "preflight failed for context " << index << ": " << error << '\n';
                return 3;
            }
            if (options.mode == RunMode::Contracts &&
                !ContractPreflight(contexts.back().get(), model, start_token, error)) {
                std::cerr << "contract preflight failed for context " << index
                          << ": " << error << '\n';
                return 3;
            }
        }

        PrintConfiguration(options, model, contexts.front().get());
        const AggregateResult result = RunWorkers(options, model, contexts);
        PrintResult(options, result);
        return result.success ? 0 : 4;
    } catch (const std::invalid_argument& error) {
        std::cerr << "argument error: " << error.what() << '\n';
        return 2;
    } catch (const std::exception& error) {
        std::cerr << "sandbox setup failed: " << error.what() << '\n';
        return 3;
    } catch (...) {
        std::cerr << "sandbox setup failed: unknown exception\n";
        return 3;
    }
}

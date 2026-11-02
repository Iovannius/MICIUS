#include "../micius_runtime.h"
#include "AICoreEngine.h"
#include "AI/AICompactScaleInternal.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <exception>
#include <cmath>
#include <limits>
#include <memory>
#include <new>
#include <string>
#include <unordered_map>
#include <vector>

struct MiciusContext {
    MiciusConfig config{};
    std::unique_ptr<AIEngineFront> core_engine;
};

micius::internal::PreparedWeightStats micius::internal::GetPreparedWeightStats(
    const MiciusContext* context) noexcept
{
    return context && context->core_engine
        ? context->core_engine->GetPreparedWeightStats() : PreparedWeightStats{};
}

namespace {

thread_local MiciusErrorCode g_last_error_code = MICIUS_ERROR_NONE;
thread_local char g_last_error_message[512] = "";

void ClearError() noexcept
{
    g_last_error_code = MICIUS_ERROR_NONE;
    g_last_error_message[0] = '\0';
}

void SetError(MiciusErrorCode code, const char* message) noexcept
{
    g_last_error_code = code;
    std::snprintf(
        g_last_error_message,
        sizeof(g_last_error_message),
        "%s",
        message ? message : "unknown error");
}

bool CheckedProduct(size_t lhs, size_t rhs, size_t* result) noexcept
{
    if (!result || (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

bool ValidateConfig(const MiciusConfig& config) noexcept
{
    if (config.hidden_dim == 0 || config.vocab_size == 0 ||
        config.max_seq_len == 0 || config.kv_pool_size_mb == 0 ||
        config.num_layers == 0 || config.num_heads == 0 ||
        config.num_kv_heads == 0 || config.head_dim == 0 ||
        config.intermediate_dim == 0) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "all model dimensions and pool sizes must be non-zero");
        return false;
    }
    if (config.worker_threads < 0) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "worker_threads cannot be negative");
        return false;
    }
    if (config.execution_mode != MICIUS_EXECUTION_REFERENCE &&
        config.execution_mode != MICIUS_EXECUTION_SPECULATIVE) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "execution_mode is not a recognized value");
        return false;
    }
    if (config.execution_mode == MICIUS_EXECUTION_SPECULATIVE) {
        SetError(
            MICIUS_ERROR_UNSUPPORTED_MODE,
            "speculative execution is disabled until the verifier and scheduler are rebuilt");
        return false;
    }
    if (!std::isfinite(config.rms_norm_epsilon) || config.rms_norm_epsilon <= 0.0f) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "rms_norm_epsilon must be finite and positive");
        return false;
    }
    if (!std::isfinite(config.rope_freq_base) || config.rope_freq_base <= 0.0f) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "rope_freq_base must be finite and positive");
        return false;
    }

    const uint64_t projected_hidden =
        static_cast<uint64_t>(config.num_heads) * config.head_dim;
    if (projected_hidden != config.hidden_dim) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "hidden_dim must equal num_heads * head_dim");
        return false;
    }
    if (config.num_heads % config.num_kv_heads != 0) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "num_kv_heads must divide num_heads");
        return false;
    }
    if ((config.head_dim & 1U) != 0) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "head_dim must be even for rotary embeddings");
        return false;
    }
    if (config.rope_dimension_count != config.head_dim) {
        SetError(
            MICIUS_ERROR_UNSUPPORTED_MODE,
            "partial rotary dimensions are not implemented; rope_dimension_count must equal head_dim");
        return false;
    }
    if (config.hidden_dim % 32U != 0) {
        SetError(
            MICIUS_ERROR_INVALID_ARGUMENT,
            "the exact Q8_0 bridge requires hidden_dim to be divisible by 32");
        return false;
    }
    if (config.intermediate_dim % 32U != 0) {
        SetError(
            MICIUS_ERROR_INVALID_ARGUMENT,
            "the exact row-wise Q8_0 contract requires intermediate_dim to be divisible by 32");
        return false;
    }

    const uint64_t kv_dimension =
        static_cast<uint64_t>(config.num_kv_heads) * config.head_dim;
    const uint64_t qkv_dimension =
        static_cast<uint64_t>(config.hidden_dim) + 2U * kv_dimension;
    if (qkv_dimension > std::numeric_limits<uint32_t>::max()) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "combined QKV dimension exceeds uint32_t");
        return false;
    }

    const uint64_t ffn_capacity = static_cast<uint64_t>(config.hidden_dim) * 4U;
    if (config.intermediate_dim > ffn_capacity) {
        SetError(
            MICIUS_ERROR_INVALID_ARGUMENT,
            "intermediate_dim exceeds the current checked workspace capacity (4 * hidden_dim)");
        return false;
    }

    size_t pool_bytes = 0;
    size_t layer_positions = 0;
    size_t kv_elements = 0;
    size_t kv_fp32_bytes = 0;
    size_t all_kv_bytes = 0;
    size_t rope_elements = 0;
    size_t rope_bytes = 0;
    if (!CheckedProduct(static_cast<size_t>(config.kv_pool_size_mb), 1024U * 1024U, &pool_bytes) ||
        pool_bytes < 2U * sizeof(DraftProposal)) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "kv_pool_size_mb is too small or overflows size_t");
        return false;
    }
    if (kv_dimension > std::numeric_limits<size_t>::max() ||
        !CheckedProduct(config.num_layers, config.max_seq_len, &layer_positions) ||
        !CheckedProduct(layer_positions, static_cast<size_t>(kv_dimension), &kv_elements) ||
        !CheckedProduct(kv_elements, sizeof(float), &kv_fp32_bytes) ||
        !CheckedProduct(kv_fp32_bytes, 2U, &all_kv_bytes)) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "KV cache allocation size overflows size_t");
        return false;
    }
    if (!CheckedProduct(config.max_seq_len, config.head_dim, &rope_elements) ||
        !CheckedProduct(rope_elements, sizeof(float), &rope_bytes)) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "RoPE cache allocation size overflows size_t");
        return false;
    }
    (void)all_kv_bytes;
    (void)rope_bytes;
    return true;
}

struct LegacyConfigPrefix {
    uint32_t hidden_dim;
    uint32_t vocab_size;
    uint32_t max_seq_len;
    uint32_t kv_pool_size_mb;
    int worker_threads;
};

MiciusConfig MakeLegacyConfig(const LegacyConfigPrefix& legacy) noexcept
{
    MiciusConfig config{};
    config.hidden_dim = legacy.hidden_dim;
    config.vocab_size = legacy.vocab_size;
    config.max_seq_len = legacy.max_seq_len;
    config.kv_pool_size_mb = legacy.kv_pool_size_mb;
    config.worker_threads = legacy.worker_threads;

    // The unversioned ABI can only safely rely on its original five-field prefix.
    // These defaults create a bounded reference-mode context; production callers
    // must use Micius_CreateEngineV1 and provide the complete topology.
    config.num_layers = 1;
    config.num_heads = 1;
    config.num_kv_heads = 1;
    config.head_dim = legacy.hidden_dim;
    config.intermediate_dim = legacy.hidden_dim;
    config.execution_mode = MICIUS_EXECUTION_REFERENCE;
    config.rms_norm_epsilon = 1.0e-5f;
    config.rope_freq_base = 10000.0f;
    config.rope_dimension_count = legacy.hidden_dim;
    return config;
}

MiciusContext* CreateEngineChecked(const MiciusConfig& config) noexcept
{
    if (!ValidateConfig(config)) return nullptr;

    try {
        auto context = std::make_unique<MiciusContext>();
        context->config = config;
        context->core_engine = std::make_unique<AIEngineFront>(context->config);
        ClearError();
        return context.release();
    } catch (const std::bad_alloc&) {
        SetError(MICIUS_ERROR_OUT_OF_MEMORY, "engine allocation failed");
    } catch (const std::exception& error) {
        SetError(MICIUS_ERROR_INTERNAL, error.what());
    } catch (...) {
        SetError(MICIUS_ERROR_INTERNAL, "unknown exception while creating the engine");
    }
    return nullptr;
}

MiciusStatus TranslateReferenceResult(
    ReferenceStepResult result,
    bool prefill) noexcept
{
    switch (result) {
        case ReferenceStepResult::SUCCESS:
            ClearError();
            return prefill ? MICIUS_STATE_TOKEN_ACCEPTED : MICIUS_STATE_YIELD_TOKEN;
        case ReferenceStepResult::MODEL_NOT_BOUND:
            SetError(MICIUS_ERROR_MODEL_NOT_BOUND, "model tensors have not been bound successfully");
            return MICIUS_STATE_NOT_READY;
        case ReferenceStepResult::SEQUENCE_LIMIT_REACHED:
            SetError(MICIUS_ERROR_CONTEXT_FULL, "the authoritative context has reached max_seq_len");
            return MICIUS_STATE_CONTEXT_FULL;
        case ReferenceStepResult::INVALID_MODEL_DATA:
            SetError(
                MICIUS_ERROR_INVALID_MODEL_DATA,
                "model pointers, symmetric Q8 descriptors, or reference buffers are invalid");
            return MICIUS_STATE_NOT_READY;
        case ReferenceStepResult::NUMERIC_ERROR:
            SetError(MICIUS_ERROR_NUMERIC, "reference forward pass produced a non-finite value");
            return MICIUS_STATE_NUMERIC_ERROR;
    }
    SetError(MICIUS_ERROR_INTERNAL, "unknown reference step result");
    return MICIUS_STATE_NOT_READY;
}

} // namespace

extern "C" {

MiciusContext* Micius_CreateEngine(const MiciusConfig* config)
{
    if (!config) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "config is null");
        return nullptr;
    }

    try {
        LegacyConfigPrefix prefix{};
        std::memcpy(&prefix, config, sizeof(prefix));
        return CreateEngineChecked(MakeLegacyConfig(prefix));
    } catch (...) {
        SetError(MICIUS_ERROR_INTERNAL, "failed to read the legacy configuration prefix");
        return nullptr;
    }
}

MiciusContext* Micius_CreateEngineV1(
    const MiciusConfig* config,
    size_t config_size,
    uint32_t abi_version)
{
    if (abi_version != MICIUS_ABI_VERSION) {
        SetError(MICIUS_ERROR_UNSUPPORTED_ABI, "ABI version does not match MICIUS_ABI_VERSION");
        return nullptr;
    }
    if (!config || config_size < sizeof(MiciusConfig)) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "config is null or smaller than MiciusConfig");
        return nullptr;
    }

    MiciusConfig local{};
    std::memcpy(&local, config, sizeof(local));
    return CreateEngineChecked(local);
}

bool Micius_InitConfig(MiciusConfig* config, size_t config_size)
{
    if (!config || config_size < sizeof(MiciusConfig)) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "config buffer is null or too small");
        return false;
    }

    MiciusConfig defaults{};
    defaults.kv_pool_size_mb = 16;
    defaults.worker_threads = 0;
    defaults.execution_mode = MICIUS_EXECUTION_REFERENCE;
    defaults.rms_norm_epsilon = 1.0e-5f;
    defaults.rope_freq_base = 10000.0f;
    std::memcpy(config, &defaults, sizeof(defaults));
    ClearError();
    return true;
}

bool Micius_BindModelTensors(
    MiciusContext* context,
    const MiciusTensorEntry* entries,
    uint32_t count)
{
    if (!context || !context->core_engine || !entries || count == 0) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "context, entries, and a non-zero count are required");
        return false;
    }

    SetError(
        MICIUS_ERROR_UNSUPPORTED_ABI,
        "legacy tensor ABI has no size/type/shape contract; use Micius_BindModelTensorsV1");
    return false;
}

bool Micius_BindModelTensorsV1(
    MiciusContext* context,
    const MiciusTensorEntryV1* entries,
    uint32_t count,
    size_t entry_size,
    uint32_t abi_version)
{
    static_assert(sizeof(MiciusQ8Block32Scale) == sizeof(QuantDescriptor));
    if (abi_version != MICIUS_ABI_VERSION) {
        SetError(MICIUS_ERROR_UNSUPPORTED_ABI, "tensor ABI version does not match MICIUS_ABI_VERSION");
        return false;
    }
    if (!context || !context->core_engine || !entries || count == 0 ||
        entry_size < sizeof(MiciusTensorEntryV1) ||
        static_cast<size_t>(count) > std::numeric_limits<size_t>::max() / entry_size) {
        SetError(
            MICIUS_ERROR_INVALID_ARGUMENT,
            "context, tensor entries, and the complete V1 entry size are required");
        return false;
    }

    try {
        std::unordered_map<std::string, MiciusTensorEntryV1> provided;
        provided.reserve(count);
        const auto* entry_bytes = reinterpret_cast<const unsigned char*>(entries);
        for (uint32_t i = 0; i < count; ++i) {
            MiciusTensorEntryV1 entry{};
            std::memcpy(
                &entry,
                entry_bytes + static_cast<size_t>(i) * entry_size,
                sizeof(entry));
            if (!entry.tensor_name || entry.tensor_name[0] == '\0' || !entry.tensor_data ||
                entry.flags != 0 || entry.rank == 0 || entry.rank > 2 ||
                entry.dimensions[0] == 0 ||
                (entry.rank == 2 && entry.dimensions[1] == 0) ||
                entry.data_type > MICIUS_TENSOR_Q8_32_SCALE) {
                SetError(MICIUS_ERROR_INVALID_ARGUMENT, "a V1 tensor entry has invalid basic metadata");
                return false;
            }
            if (!provided.emplace(entry.tensor_name, entry).second) {
                SetError(MICIUS_ERROR_INVALID_ARGUMENT, "duplicate tensor name in V1 tensor set");
                return false;
            }
        }

        std::vector<MiciusTensorEntry> legacy_entries;
        std::vector<AIEngineFront::OwnedScaleBuffer> scale_buffers;
        std::vector<std::string> validated_names;
        const size_t required_capacity =
            static_cast<size_t>(context->config.num_layers) * 16U + 8U;
        legacy_entries.reserve(required_capacity);
        validated_names.reserve(required_capacity);
        scale_buffers.reserve(static_cast<size_t>(context->config.num_layers) * 7U + 2U);

        auto require_tensor = [&](
            const std::string& name,
            MiciusTensorDataType type,
            uint64_t rows,
            uint64_t columns) -> bool {
            const auto found = provided.find(name);
            if (found == provided.end()) {
                SetError(MICIUS_ERROR_INVALID_ARGUMENT, ("missing required tensor: " + name).c_str());
                return false;
            }
            const MiciusTensorEntryV1& entry = found->second;
            const bool is_vector = columns == 0;
            const uint32_t expected_rank = is_vector ? 1U : 2U;
            if (entry.data_type != static_cast<uint32_t>(type) ||
                entry.rank != expected_rank || entry.dimensions[0] != rows ||
                (!is_vector && entry.dimensions[1] != columns)) {
                SetError(
                    MICIUS_ERROR_INVALID_ARGUMENT,
                    ("type or shape mismatch for required tensor: " + name).c_str());
                return false;
            }

            size_t expected_bytes = 0;
            size_t expected_stride = 0;
            size_t element_count = 0;
            if (rows > std::numeric_limits<size_t>::max() ||
                (!is_vector && columns > std::numeric_limits<size_t>::max())) {
                SetError(MICIUS_ERROR_INVALID_ARGUMENT, "tensor dimensions exceed size_t");
                return false;
            }

            if (type == MICIUS_TENSOR_F32) {
                const size_t row_count = static_cast<size_t>(rows);
                const size_t column_count = is_vector ? 1U : static_cast<size_t>(columns);
                if (!CheckedProduct(row_count, column_count, &element_count) ||
                    !CheckedProduct(element_count, sizeof(float), &expected_bytes) ||
                    (!is_vector &&
                     !CheckedProduct(column_count, sizeof(float), &expected_stride))) {
                    SetError(MICIUS_ERROR_INVALID_ARGUMENT, "F32 tensor byte size overflows");
                    return false;
                }
            } else if (type == MICIUS_TENSOR_Q8_32_WEIGHT) {
                if (is_vector || !CheckedProduct(
                        static_cast<size_t>(rows),
                        static_cast<size_t>(columns),
                        &expected_bytes)) {
                    SetError(MICIUS_ERROR_INVALID_ARGUMENT, "Q8 weight byte size overflows");
                    return false;
                }
                expected_stride = static_cast<size_t>(columns);
            } else {
                if (is_vector || columns > std::numeric_limits<size_t>::max()) {
                    SetError(MICIUS_ERROR_INVALID_ARGUMENT, "Q8 scale dimensions are invalid");
                    return false;
                }
                const size_t descriptor_columns = static_cast<size_t>(columns);
                if (!CheckedProduct(
                        static_cast<size_t>(rows),
                        descriptor_columns,
                        &element_count) ||
                    !CheckedProduct(element_count, sizeof(MiciusQ8Block32Scale), &expected_bytes) ||
                    !CheckedProduct(
                        descriptor_columns,
                        sizeof(MiciusQ8Block32Scale),
                        &expected_stride)) {
                    SetError(MICIUS_ERROR_INVALID_ARGUMENT, "Q8 scale byte size overflows");
                    return false;
                }
            }

            if (entry.byte_size != expected_bytes ||
                entry.row_stride_bytes != expected_stride) {
                SetError(
                    MICIUS_ERROR_INVALID_ARGUMENT,
                    ("byte size or row stride mismatch for required tensor: " + name).c_str());
                return false;
            }

            const size_t required_alignment =
                type == MICIUS_TENSOR_F32
                    ? alignof(float)
                    : (type == MICIUS_TENSOR_Q8_32_SCALE
                           ? alignof(MiciusQ8Block32Scale)
                           : 1U);
            if (reinterpret_cast<std::uintptr_t>(entry.tensor_data) %
                    required_alignment != 0U) {
                SetError(
                    MICIUS_ERROR_INVALID_ARGUMENT,
                    ("misaligned tensor data pointer: " + name).c_str());
                return false;
            }

            if (type == MICIUS_TENSOR_F32) {
                const auto* values = static_cast<const float*>(entry.tensor_data);
                for (size_t i = 0; i < element_count; ++i) {
                    if (!std::isfinite(values[i])) {
                        SetError(
                            MICIUS_ERROR_INVALID_MODEL_DATA,
                            ("non-finite F32 tensor value: " + name).c_str());
                        return false;
                    }
                }
                validated_names.push_back(name);
                legacy_entries.push_back({validated_names.back().c_str(), entry.tensor_data});
            } else if (type == MICIUS_TENSOR_Q8_32_SCALE) {
                const auto* public_scales =
                    static_cast<const MiciusQ8Block32Scale*>(entry.tensor_data);
                AIEngineFront::OwnedScaleBuffer owned_scales;
                owned_scales.count = element_count;
                for (size_t i = 0; i < element_count; ++i) {
                    if (!std::isfinite(public_scales[i].scale) ||
                        public_scales[i].zero_point != 0 ||
                        public_scales[i].padding[0] != 0 ||
                        public_scales[i].padding[1] != 0) {
                        SetError(
                            MICIUS_ERROR_INVALID_MODEL_DATA,
                            ("non-symmetric or invalid Q8 scale descriptor: " + name).c_str());
                        return false;
                    }
                }
#if defined(MICIUS_REFERENCE_GEMV_COMPACT_SCALES)
                owned_scales.compact_data = std::make_unique<float[]>(element_count);
                micius::internal::CopyCompactScaleBits(
                    public_scales, owned_scales.compact_data.get(), element_count);
#else
                owned_scales.data = std::make_unique<QuantDescriptor[]>(element_count);
                for (size_t i = 0; i < element_count; ++i) {
                    owned_scales.data[i] = QuantDescriptor(public_scales[i].scale, 0);
                }
#endif
                validated_names.push_back(name);
#if defined(MICIUS_REFERENCE_GEMV_COMPACT_SCALES)
                legacy_entries.push_back(
                    {validated_names.back().c_str(), owned_scales.compact_data.get()});
#else
                legacy_entries.push_back(
                    {validated_names.back().c_str(), owned_scales.data.get()});
#endif
                scale_buffers.push_back(std::move(owned_scales));
            } else {
                validated_names.push_back(name);
                legacy_entries.push_back({validated_names.back().c_str(), entry.tensor_data});
            }
            return true;
        };

        const uint64_t hidden = context->config.hidden_dim;
        const uint64_t vocabulary = context->config.vocab_size;
        const uint64_t intermediate = context->config.intermediate_dim;
        const uint64_t kv_dimension =
            static_cast<uint64_t>(context->config.num_kv_heads) * context->config.head_dim;
        const uint64_t hidden_blocks = hidden / 32U;
        const uint64_t intermediate_blocks = intermediate / 32U;

        if (!require_tensor("token_embd.weight.f32", MICIUS_TENSOR_F32, vocabulary, hidden) ||
            !require_tensor("output_norm.weight", MICIUS_TENSOR_F32, hidden, 0)) {
            return false;
        }

        const bool has_output = provided.count("output.weight") != 0 ||
                                provided.count("output.scales") != 0;
        if (has_output) {
            if (!require_tensor("output.weight", MICIUS_TENSOR_Q8_32_WEIGHT, vocabulary, hidden) ||
                !require_tensor("output.scales", MICIUS_TENSOR_Q8_32_SCALE, vocabulary, hidden_blocks)) {
                return false;
            }
        } else if (!require_tensor(
                       "token_embd.weight",
                       MICIUS_TENSOR_Q8_32_WEIGHT,
                       vocabulary,
                       hidden) ||
                   !require_tensor(
                       "token_embd.scales",
                       MICIUS_TENSOR_Q8_32_SCALE,
                       vocabulary,
                       hidden_blocks)) {
            return false;
        }

        for (uint32_t layer = 0; layer < context->config.num_layers; ++layer) {
            const std::string prefix = "blk." + std::to_string(layer) + ".";
            if (!require_tensor(prefix + "attn_q.weight", MICIUS_TENSOR_Q8_32_WEIGHT, hidden, hidden) ||
                !require_tensor(prefix + "attn_q.scales", MICIUS_TENSOR_Q8_32_SCALE, hidden, hidden_blocks) ||
                !require_tensor(prefix + "attn_k.weight", MICIUS_TENSOR_Q8_32_WEIGHT, kv_dimension, hidden) ||
                !require_tensor(prefix + "attn_k.scales", MICIUS_TENSOR_Q8_32_SCALE, kv_dimension, hidden_blocks) ||
                !require_tensor(prefix + "attn_v.weight", MICIUS_TENSOR_Q8_32_WEIGHT, kv_dimension, hidden) ||
                !require_tensor(prefix + "attn_v.scales", MICIUS_TENSOR_Q8_32_SCALE, kv_dimension, hidden_blocks) ||
                !require_tensor(prefix + "attn_output.weight", MICIUS_TENSOR_Q8_32_WEIGHT, hidden, hidden) ||
                !require_tensor(prefix + "attn_output.scales", MICIUS_TENSOR_Q8_32_SCALE, hidden, hidden_blocks) ||
                !require_tensor(prefix + "ffn_gate.weight", MICIUS_TENSOR_Q8_32_WEIGHT, intermediate, hidden) ||
                !require_tensor(prefix + "ffn_gate.scales", MICIUS_TENSOR_Q8_32_SCALE, intermediate, hidden_blocks) ||
                !require_tensor(prefix + "ffn_up.weight", MICIUS_TENSOR_Q8_32_WEIGHT, intermediate, hidden) ||
                !require_tensor(prefix + "ffn_up.scales", MICIUS_TENSOR_Q8_32_SCALE, intermediate, hidden_blocks) ||
                !require_tensor(prefix + "ffn_down.weight", MICIUS_TENSOR_Q8_32_WEIGHT, hidden, intermediate) ||
                !require_tensor(prefix + "ffn_down.scales", MICIUS_TENSOR_Q8_32_SCALE, hidden, intermediate_blocks) ||
                !require_tensor(prefix + "attn_norm.weight", MICIUS_TENSOR_F32, hidden, 0) ||
                !require_tensor(prefix + "ffn_norm.weight", MICIUS_TENSOR_F32, hidden, 0)) {
                return false;
            }
        }

        if (legacy_entries.size() > std::numeric_limits<uint32_t>::max()) {
            SetError(MICIUS_ERROR_INVALID_ARGUMENT, "validated tensor count exceeds uint32_t");
            return false;
        }
        if (!context->core_engine->BindValidatedTensors(
                legacy_entries.data(),
                static_cast<uint32_t>(legacy_entries.size()),
                std::move(scale_buffers))) {
            SetError(MICIUS_ERROR_INVALID_MODEL_DATA, "core rejected the validated tensor set");
            return false;
        }
        ClearError();
        return true;
    } catch (const std::bad_alloc&) {
        SetError(MICIUS_ERROR_OUT_OF_MEMORY, "V1 tensor validation or scale ownership allocation failed");
    } catch (const std::exception& error) {
        SetError(MICIUS_ERROR_INTERNAL, error.what());
    } catch (...) {
        SetError(MICIUS_ERROR_INTERNAL, "unknown exception while binding V1 tensors");
    }
    return false;
}

void Micius_DestroyEngine(MiciusContext* context)
{
    delete context;
    ClearError();
}

void Micius_ResetEngine(MiciusContext* context)
{
    (void)Micius_ResetEngineV1(context);
}

bool Micius_ResetEngineV1(MiciusContext* context)
{
    if (!context || !context->core_engine) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "context is null");
        return false;
    }
    if (!context->core_engine->ResetState()) {
        SetError(MICIUS_ERROR_UNSUPPORTED_MODE, "state reset is unsafe while speculative workers are active");
        return false;
    }
    ClearError();
    return true;
}

MiciusErrorCode Micius_GetLastErrorCode(void)
{
    return g_last_error_code;
}

const char* Micius_GetLastError(void)
{
    return g_last_error_message;
}

bool Micius_LoadQuantWeights(
    MiciusContext* context,
    const int8_t* quant_weights,
    const void* scales)
{
    if (!context || !context->core_engine || !quant_weights || !scales) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "context, quant_weights, and scales are required");
        return false;
    }
    SetError(
        MICIUS_ERROR_UNSUPPORTED_ABI,
        "legacy raw weight ABI has no buffer sizes; use Micius_BindModelTensorsV1");
    return false;
}

void Micius_RegisterVerifier(
    MiciusContext* context,
    MiciusVerifierCallback verifier_callback,
    void* user_data)
{
    if (!context || !context->core_engine) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "context is null");
        return;
    }
    (void)verifier_callback;
    (void)user_data;
    SetError(
        MICIUS_ERROR_UNSUPPORTED_MODE,
        "verifier registration is unavailable while speculative execution is disabled");
}

MiciusStatus Micius_Step(
    MiciusContext* context,
    uint32_t input_token,
    const float* hidden_state,
    uint32_t* out_accepted_token)
{
    if (!context || !context->core_engine || !hidden_state || !out_accepted_token) {
        SetError(
            MICIUS_ERROR_INVALID_ARGUMENT,
            "context, hidden_state, and out_accepted_token are required");
        return MICIUS_STATE_INVALID_ARGUMENT;
    }
    if (input_token >= context->config.vocab_size) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "input_token is outside the configured vocabulary");
        return MICIUS_STATE_INVALID_ARGUMENT;
    }
    if (!context->core_engine->IsModelBound()) {
        SetError(MICIUS_ERROR_MODEL_NOT_BOUND, "model tensors have not been bound successfully");
        return MICIUS_STATE_NOT_READY;
    }
    if (!context->core_engine->IsSpeculativeEnabled()) {
        return TranslateReferenceResult(
            context->core_engine->DecodeReferenceToken(input_token, out_accepted_token),
            false);
    }

    try {
        PumpResult result = context->core_engine->PumpPipeline(
            input_token,
            hidden_state,
            out_accepted_token);
        if (result == PumpResult::YIELD_TOKEN) {
            ClearError();
            return MICIUS_STATE_YIELD_TOKEN;
        }
        if (result == PumpResult::TREE_REJECTED) {
            ClearError();
            return MICIUS_STATE_TREE_REJECTED;
        }
        SetError(MICIUS_ERROR_OUT_OF_MEMORY, "the internal pipeline exhausted its KV slots");
        return MICIUS_STATE_OOM_ERROR;
    } catch (const std::bad_alloc&) {
        SetError(MICIUS_ERROR_OUT_OF_MEMORY, "step allocation failed");
    } catch (const std::exception& error) {
        SetError(MICIUS_ERROR_INTERNAL, error.what());
    } catch (...) {
        SetError(MICIUS_ERROR_INTERNAL, "unknown exception while stepping the engine");
    }
    return MICIUS_STATE_OOM_ERROR;
}

MiciusStatus Micius_PrefillTokenV1(MiciusContext* context, uint32_t input_token)
{
    if (!context || !context->core_engine) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "context is null");
        return MICIUS_STATE_INVALID_ARGUMENT;
    }
    if (input_token >= context->config.vocab_size) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "input_token is outside the configured vocabulary");
        return MICIUS_STATE_INVALID_ARGUMENT;
    }
    if (context->core_engine->IsSpeculativeEnabled()) {
        SetError(MICIUS_ERROR_UNSUPPORTED_MODE, "authoritative prefill requires reference mode");
        return MICIUS_STATE_NOT_READY;
    }
    return TranslateReferenceResult(
        context->core_engine->PrefillReferenceToken(input_token),
        true);
}

MiciusStatus Micius_DecodeOneV1(
    MiciusContext* context,
    uint32_t input_token,
    uint32_t* out_token)
{
    if (!context || !context->core_engine || !out_token) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "context and out_token are required");
        return MICIUS_STATE_INVALID_ARGUMENT;
    }
    if (input_token >= context->config.vocab_size) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "input_token is outside the configured vocabulary");
        return MICIUS_STATE_INVALID_ARGUMENT;
    }
    if (context->core_engine->IsSpeculativeEnabled()) {
        SetError(MICIUS_ERROR_UNSUPPORTED_MODE, "DecodeOneV1 requires reference mode");
        return MICIUS_STATE_NOT_READY;
    }
    return TranslateReferenceResult(
        context->core_engine->DecodeReferenceToken(input_token, out_token),
        false);
}

uint32_t Micius_GetSequenceLengthV1(MiciusContext* context)
{
    if (!context || !context->core_engine) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "context is null");
        return 0;
    }
    ClearError();
    return context->core_engine->GetReferenceSequenceLength();
}

bool Micius_CopyLastLogitsV1(
    MiciusContext* context,
    float* out_logits,
    size_t logits_count)
{
    if (!context || !context->core_engine || !out_logits) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "context and out_logits are required");
        return false;
    }
    if (logits_count < context->config.vocab_size) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "logits buffer is smaller than vocab_size");
        return false;
    }
    if (!context->core_engine->CopyLastReferenceLogits(out_logits, logits_count)) {
        SetError(MICIUS_ERROR_NOT_READY, "no successful reference logits are available");
        return false;
    }
    ClearError();
    return true;
}

bool Micius_CopyReferenceKVV1(
    MiciusContext* context,
    uint32_t layer,
    uint32_t position,
    float* out_key,
    float* out_value,
    size_t element_count)
{
    if (!context || !context->core_engine || !out_key || !out_value) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "context, out_key, and out_value are required");
        return false;
    }
    const size_t required =
        static_cast<size_t>(context->config.num_kv_heads) * context->config.head_dim;
    if (element_count < required) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "KV output buffers are smaller than kv_dim");
        return false;
    }
    if (!context->core_engine->CopyReferenceKVSlot(
            layer,
            position,
            out_key,
            out_value,
            element_count)) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "reference KV layer or position is not available");
        return false;
    }
    ClearError();
    return true;
}

uint32_t Micius_GetTreeValidCount(MiciusDraftTree* tree)
{
    if (!tree) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "tree is null");
        return 0;
    }
    SetError(
        MICIUS_ERROR_UNSUPPORTED_MODE,
        "draft-tree inspection is unavailable while speculative execution is disabled");
    return 0;
}

void Micius_AcceptTreeNode(MiciusDraftTree* tree, int32_t node_index)
{
    if (!tree) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "tree is null");
        return;
    }
    (void)node_index;
    SetError(
        MICIUS_ERROR_UNSUPPORTED_MODE,
        "draft-tree acceptance is unavailable while speculative execution is disabled");
}

void Micius_GetRuntimeStats(MiciusContext* context, MiciusRuntimeStats* out_stats)
{
    if (!context || !context->core_engine || !out_stats) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "context and out_stats are required");
        return;
    }
    out_stats->total_tokens_generated =
        context->core_engine->total_tokens_generated.load(std::memory_order_relaxed);
    out_stats->tree_reject_count =
        context->core_engine->tree_reject_count.load(std::memory_order_relaxed);
    out_stats->kv_slots_total = context->config.max_seq_len;
    out_stats->kv_slots_used = context->core_engine->GetUsedKVSlots();
    ClearError();
}

uint32_t Micius_GetWorkerCount(MiciusContext* context)
{
    if (!context || !context->core_engine) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "context is null");
        return 0;
    }
    const size_t count = context->core_engine->GetWorkerCount();
    if (count > std::numeric_limits<uint32_t>::max()) {
        SetError(MICIUS_ERROR_INTERNAL, "worker count exceeds the public ABI range");
        return 0;
    }
    ClearError();
    return static_cast<uint32_t>(count);
}

bool Micius_GetWorkerInfo(
    MiciusContext* context,
    uint32_t worker_index,
    MiciusWorkerInfo* out_info)
{
    if (!context || !context->core_engine || !out_info) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "context and out_info are required");
        return false;
    }
    if (worker_index >= context->core_engine->GetWorkerCount()) {
        SetError(MICIUS_ERROR_INVALID_ARGUMENT, "worker_index is out of range");
        return false;
    }

    WorkerContext* worker = context->core_engine->GetWorkerContext(worker_index);
    if (!worker) {
        SetError(MICIUS_ERROR_INTERNAL, "worker context is null");
        return false;
    }
    out_info->core_id = context->core_engine->GetWorkerCoreId(worker_index);
    out_info->tasks_processed = worker->tasks_processed.load(std::memory_order_relaxed);
    out_info->tasks_stolen = worker->tasks_stolen.load(std::memory_order_relaxed);
    ClearError();
    return true;
}

} // extern "C"

micius::i25::Stats micius::i25::BindingStats(const MiciusContext* c) noexcept {
    return c && c->core_engine ? c->core_engine->I25Stats() : Stats{};
}
std::span<const micius::i25::RetiredWeight> micius::i25::RetirableWeights(const MiciusContext* c) noexcept {
    return c && c->core_engine ? c->core_engine->I25Retired() : std::span<const RetiredWeight>{};
}

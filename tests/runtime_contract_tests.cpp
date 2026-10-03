#include "micius_runtime.h"
#include "AICoreEngine.h"
#include "AIPrimitives.h"
#include "AI/PreparedGEMVWeights.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#if defined(NDEBUG)
#error "runtime_contract_tests requires assert() checks; build this target without NDEBUG"
#endif

template <typename T>
concept ExposesRawTensorBinding = requires(T& engine, const MiciusTensorEntry* entries) {
    engine.BindTensors(entries, 1U);
};

template <typename T>
concept ExposesRawScaleReplacement =
    requires(T& engine, const int8_t* weights, const QuantDescriptor* scales) {
        engine.LoadWeights(weights, scales);
    };

static_assert(!ExposesRawTensorBinding<AIEngineFront>);
static_assert(!ExposesRawScaleReplacement<AIEngineFront>);

namespace {

void CheckPreparedWeightBinding(MiciusContext* context, const MiciusConfig& cfg)
{
    const auto stats=micius::internal::GetPreparedWeightStats(context);
    const uint64_t rows=stats.selection==1
        ? uint64_t{cfg.hidden_dim}+2U*cfg.num_kv_heads*cfg.head_dim : cfg.hidden_dim;
    const uint64_t expected=uint64_t{cfg.num_layers}*rows*cfg.hidden_dim*sizeof(float);
    assert(stats.selection<=2 && stats.preparation_seconds>=0);
    if (!stats.selection) {
        assert(stats.extra_bytes==0 && stats.matrices==0 && std::strcmp(stats.status,"disabled")==0);
    } else if (stats.budget_bytes>=expected) {
        assert(stats.extra_bytes==expected && stats.matrices==cfg.num_layers);
        assert(std::strcmp(stats.status,"ready")==0);
    } else {
        assert(stats.extra_bytes==0 && stats.matrices==0 && std::strcmp(stats.status,"fallback_budget")==0);
    }
}

MiciusConfig MakeMinimalConfig()
{
    MiciusConfig config{};
    assert(Micius_InitConfig(&config, sizeof(config)));
    config.hidden_dim = 64;
    config.vocab_size = 128;
    config.max_seq_len = 8;
    config.kv_pool_size_mb = 1;
    config.worker_threads = 0;
    config.num_layers = 1;
    config.num_heads = 1;
    config.num_kv_heads = 1;
    config.head_dim = 64;
    config.intermediate_dim = 64;
    config.execution_mode = MICIUS_EXECUTION_REFERENCE;
    config.rope_dimension_count = config.head_dim;
    return config;
}

void ExpectError(MiciusErrorCode expected, const char* fragment)
{
    assert(Micius_GetLastErrorCode() == expected);
    const char* message = Micius_GetLastError();
    assert(message != nullptr);
    if (fragment) assert(std::strstr(message, fragment) != nullptr);
}

void TestConfigAndAbiValidation()
{
    MiciusConfig config{};
    assert(!Micius_InitConfig(nullptr, sizeof(config)));
    ExpectError(MICIUS_ERROR_INVALID_ARGUMENT, "null");

    assert(!Micius_InitConfig(&config, sizeof(config) - 1));
    ExpectError(MICIUS_ERROR_INVALID_ARGUMENT, "too small");

    config = MakeMinimalConfig();
    assert(Micius_CreateEngineV1(&config, sizeof(config), MICIUS_ABI_VERSION + 1) == nullptr);
    ExpectError(MICIUS_ERROR_UNSUPPORTED_ABI, "ABI version");

    MiciusConfig invalid = config;
    invalid.hidden_dim = 128;
    assert(Micius_CreateEngineV1(&invalid, sizeof(invalid), MICIUS_ABI_VERSION) == nullptr);
    ExpectError(MICIUS_ERROR_INVALID_ARGUMENT, "num_heads * head_dim");

    invalid = config;
    invalid.execution_mode = MICIUS_EXECUTION_SPECULATIVE;
    assert(Micius_CreateEngineV1(&invalid, sizeof(invalid), MICIUS_ABI_VERSION) == nullptr);
    ExpectError(MICIUS_ERROR_UNSUPPORTED_MODE, "speculative execution");

    invalid = config;
    invalid.num_layers = std::numeric_limits<uint32_t>::max();
    invalid.max_seq_len = std::numeric_limits<uint32_t>::max();
    assert(Micius_CreateEngineV1(&invalid, sizeof(invalid), MICIUS_ABI_VERSION) == nullptr);
    ExpectError(MICIUS_ERROR_INVALID_ARGUMENT, "KV cache allocation size overflows");
}

void TestReferenceLifecycleAndFailClosedStep()
{
    MiciusConfig config = MakeMinimalConfig();
    MiciusContext* context = Micius_CreateEngineV1(
        &config,
        sizeof(config),
        MICIUS_ABI_VERSION);
    assert(context != nullptr);
    ExpectError(MICIUS_ERROR_NONE, nullptr);

    assert(Micius_GetWorkerCount(context) == 0);
    MiciusRuntimeStats stats{};
    Micius_GetRuntimeStats(context, &stats);
    assert(stats.total_tokens_generated == 0);
    assert(stats.tree_reject_count == 0);
    assert(stats.kv_slots_used == 0);
    assert(stats.kv_slots_total == config.max_seq_len);

    std::vector<float> hidden(config.hidden_dim, 0.0f);
    uint32_t output_token = 0xA5A5A5A5U;
    MiciusStatus status = Micius_Step(context, 1, hidden.data(), &output_token);
    assert(status == MICIUS_STATE_NOT_READY);
    assert(output_token == 0xA5A5A5A5U);
    ExpectError(MICIUS_ERROR_MODEL_NOT_BOUND, "not been bound");

    const MiciusTensorEntry incomplete[] = {{"output_norm.weight", hidden.data()}};
    assert(!Micius_BindModelTensors(context, incomplete, 1));
    ExpectError(MICIUS_ERROR_UNSUPPORTED_ABI, "legacy tensor ABI");

    int8_t legacy_weight = 0;
    MiciusQ8Block32Scale legacy_scale{1.0f, 0, {0, 0}};
    assert(!Micius_LoadQuantWeights(context, &legacy_weight, &legacy_scale));
    ExpectError(MICIUS_ERROR_UNSUPPORTED_ABI, "legacy raw weight ABI");

    Micius_RegisterVerifier(context, nullptr, nullptr);
    ExpectError(MICIUS_ERROR_UNSUPPORTED_MODE, "speculative execution");

    assert(Micius_ResetEngineV1(context));
    ExpectError(MICIUS_ERROR_NONE, nullptr);
    Micius_DestroyEngine(context);
    ExpectError(MICIUS_ERROR_NONE, nullptr);
}

void TestDeterministicReferenceDecodeAndContextOwnership()
{
    MiciusConfig config = MakeMinimalConfig();
    config.num_layers = 2;
    MiciusContext* context = Micius_CreateEngineV1(
        &config,
        sizeof(config),
        MICIUS_ABI_VERSION);
    assert(context != nullptr);

    const size_t matrix_elements =
        static_cast<size_t>(config.vocab_size) * config.hidden_dim;
    std::vector<int8_t> output_weights(matrix_elements, int8_t{0});
    std::vector<int8_t> layer_weights(
        static_cast<size_t>(config.hidden_dim) * config.hidden_dim,
        int8_t{0});
    std::vector<int8_t> key_weights_layer0(
        static_cast<size_t>(config.hidden_dim) * config.hidden_dim,
        int8_t{0});
    std::vector<int8_t> key_weights_layer1(
        static_cast<size_t>(config.hidden_dim) * config.hidden_dim,
        int8_t{0});
    std::vector<MiciusQ8Block32Scale> quant_scales(
        static_cast<size_t>(config.vocab_size) * (config.hidden_dim / 32U),
        MiciusQ8Block32Scale{1.0f, 0, {0, 0}});
    std::vector<float> token_embedding(matrix_elements, 0.0f);
    std::vector<float> norm(config.hidden_dim, 1.0f);

    constexpr uint32_t kExpectedToken = 7;
    for (uint32_t i = 0; i < config.hidden_dim; ++i) {
        token_embedding[static_cast<size_t>(1) * config.hidden_dim + i] = 1.0f;
        output_weights[static_cast<size_t>(kExpectedToken) * config.hidden_dim + i] = 1;
    }
    for (uint32_t row = 0; row < config.hidden_dim; ++row) {
        const int8_t layer0_value = static_cast<int8_t>(static_cast<int32_t>(row % 9U) - 4);
        const int8_t layer1_value = static_cast<int8_t>(static_cast<int32_t>(row % 7U) + 1);
        std::fill_n(
            key_weights_layer0.data() + static_cast<size_t>(row) * config.hidden_dim,
            config.hidden_dim,
            layer0_value);
        std::fill_n(
            key_weights_layer1.data() + static_cast<size_t>(row) * config.hidden_dim,
            config.hidden_dim,
            layer1_value);
    }

    std::vector<MiciusTensorEntryV1> entries;
    auto add_f32 = [&](const char* name, const float* data, uint64_t rows, uint64_t columns) {
        MiciusTensorEntryV1 entry{};
        entry.tensor_name = name;
        entry.tensor_data = data;
        entry.data_type = MICIUS_TENSOR_F32;
        entry.rank = columns == 0 ? 1U : 2U;
        entry.dimensions[0] = rows;
        entry.dimensions[1] = columns;
        entry.byte_size = static_cast<size_t>(rows) *
                          static_cast<size_t>(columns == 0 ? 1U : columns) * sizeof(float);
        entry.row_stride_bytes = columns == 0
            ? 0
            : static_cast<size_t>(columns) * sizeof(float);
        entries.push_back(entry);
    };
    auto add_q8_weight = [&](const char* name, const int8_t* data, uint64_t rows, uint64_t columns) {
        MiciusTensorEntryV1 entry{};
        entry.tensor_name = name;
        entry.tensor_data = data;
        entry.byte_size = static_cast<size_t>(rows * columns);
        entry.data_type = MICIUS_TENSOR_Q8_32_WEIGHT;
        entry.rank = 2;
        entry.dimensions[0] = rows;
        entry.dimensions[1] = columns;
        entry.row_stride_bytes = static_cast<size_t>(columns);
        entries.push_back(entry);
    };
    auto add_q8_scale = [&](const char* name, uint64_t rows, uint64_t blocks) {
        MiciusTensorEntryV1 entry{};
        entry.tensor_name = name;
        entry.tensor_data = quant_scales.data();
        entry.byte_size = static_cast<size_t>(rows * blocks) * sizeof(MiciusQ8Block32Scale);
        entry.data_type = MICIUS_TENSOR_Q8_32_SCALE;
        entry.rank = 2;
        entry.dimensions[0] = rows;
        entry.dimensions[1] = blocks;
        entry.row_stride_bytes = static_cast<size_t>(blocks) * sizeof(MiciusQ8Block32Scale);
        entries.push_back(entry);
    };

    add_f32("token_embd.weight.f32", token_embedding.data(), config.vocab_size, config.hidden_dim);
    add_f32("output_norm.weight", norm.data(), config.hidden_dim, 0);
    add_q8_weight("output.weight", output_weights.data(), config.vocab_size, config.hidden_dim);
    add_q8_scale("output.scales", config.vocab_size, 2);
    add_q8_weight("blk.0.attn_q.weight", layer_weights.data(), config.hidden_dim, config.hidden_dim);
    add_q8_scale("blk.0.attn_q.scales", config.hidden_dim, 2);
    add_q8_weight("blk.0.attn_k.weight", key_weights_layer0.data(), config.hidden_dim, config.hidden_dim);
    add_q8_scale("blk.0.attn_k.scales", config.hidden_dim, 2);
    add_q8_weight("blk.0.attn_v.weight", layer_weights.data(), config.hidden_dim, config.hidden_dim);
    add_q8_scale("blk.0.attn_v.scales", config.hidden_dim, 2);
    add_q8_weight("blk.0.attn_output.weight", layer_weights.data(), config.hidden_dim, config.hidden_dim);
    add_q8_scale("blk.0.attn_output.scales", config.hidden_dim, 2);
    add_q8_weight("blk.0.ffn_gate.weight", layer_weights.data(), config.intermediate_dim, config.hidden_dim);
    add_q8_scale("blk.0.ffn_gate.scales", config.intermediate_dim, 2);
    add_q8_weight("blk.0.ffn_up.weight", layer_weights.data(), config.intermediate_dim, config.hidden_dim);
    add_q8_scale("blk.0.ffn_up.scales", config.intermediate_dim, 2);
    add_q8_weight("blk.0.ffn_down.weight", layer_weights.data(), config.hidden_dim, config.intermediate_dim);
    add_q8_scale("blk.0.ffn_down.scales", config.hidden_dim, 2);
    add_f32("blk.0.attn_norm.weight", norm.data(), config.hidden_dim, 0);
    add_f32("blk.0.ffn_norm.weight", norm.data(), config.hidden_dim, 0);
    add_q8_weight("blk.1.attn_q.weight", layer_weights.data(), config.hidden_dim, config.hidden_dim);
    add_q8_scale("blk.1.attn_q.scales", config.hidden_dim, 2);
    add_q8_weight("blk.1.attn_k.weight", key_weights_layer1.data(), config.hidden_dim, config.hidden_dim);
    add_q8_scale("blk.1.attn_k.scales", config.hidden_dim, 2);
    add_q8_weight("blk.1.attn_v.weight", layer_weights.data(), config.hidden_dim, config.hidden_dim);
    add_q8_scale("blk.1.attn_v.scales", config.hidden_dim, 2);
    add_q8_weight("blk.1.attn_output.weight", layer_weights.data(), config.hidden_dim, config.hidden_dim);
    add_q8_scale("blk.1.attn_output.scales", config.hidden_dim, 2);
    add_q8_weight("blk.1.ffn_gate.weight", layer_weights.data(), config.intermediate_dim, config.hidden_dim);
    add_q8_scale("blk.1.ffn_gate.scales", config.intermediate_dim, 2);
    add_q8_weight("blk.1.ffn_up.weight", layer_weights.data(), config.intermediate_dim, config.hidden_dim);
    add_q8_scale("blk.1.ffn_up.scales", config.intermediate_dim, 2);
    add_q8_weight("blk.1.ffn_down.weight", layer_weights.data(), config.hidden_dim, config.intermediate_dim);
    add_q8_scale("blk.1.ffn_down.scales", config.hidden_dim, 2);
    add_f32("blk.1.attn_norm.weight", norm.data(), config.hidden_dim, 0);
    add_f32("blk.1.ffn_norm.weight", norm.data(), config.hidden_dim, 0);

    assert(!Micius_BindModelTensorsV1(
        context,
        entries.data(),
        static_cast<uint32_t>(entries.size()),
        sizeof(MiciusTensorEntryV1) - 1,
        MICIUS_ABI_VERSION));
    ExpectError(MICIUS_ERROR_INVALID_ARGUMENT, "entry size");

    entries[2].byte_size -= 1;
    assert(!Micius_BindModelTensorsV1(
        context,
        entries.data(),
        static_cast<uint32_t>(entries.size()),
        sizeof(MiciusTensorEntryV1),
        MICIUS_ABI_VERSION));
    ExpectError(MICIUS_ERROR_INVALID_ARGUMENT, "byte size");
    entries[2].byte_size += 1;

    const MiciusQ8Block32Scale valid_scale = quant_scales[0];
    quant_scales[0].zero_point = 1;
    assert(!Micius_BindModelTensorsV1(
        context,
        entries.data(),
        static_cast<uint32_t>(entries.size()),
        sizeof(MiciusTensorEntryV1),
        MICIUS_ABI_VERSION));
    ExpectError(MICIUS_ERROR_INVALID_MODEL_DATA, "non-symmetric");
    quant_scales[0] = valid_scale;

    quant_scales[0].scale = std::numeric_limits<float>::quiet_NaN();
    assert(!Micius_BindModelTensorsV1(
        context,
        entries.data(),
        static_cast<uint32_t>(entries.size()),
        sizeof(MiciusTensorEntryV1),
        MICIUS_ABI_VERSION));
    ExpectError(MICIUS_ERROR_INVALID_MODEL_DATA, "invalid Q8 scale");
    quant_scales[0] = valid_scale;

    quant_scales[0].padding[1] = 1;
    assert(!Micius_BindModelTensorsV1(
        context,
        entries.data(),
        static_cast<uint32_t>(entries.size()),
        sizeof(MiciusTensorEntryV1),
        MICIUS_ABI_VERSION));
    ExpectError(MICIUS_ERROR_INVALID_MODEL_DATA, "invalid Q8 scale");
    quant_scales[0] = valid_scale;

    token_embedding[0] = std::numeric_limits<float>::infinity();
    assert(!Micius_BindModelTensorsV1(
        context,
        entries.data(),
        static_cast<uint32_t>(entries.size()),
        sizeof(MiciusTensorEntryV1),
        MICIUS_ABI_VERSION));
    ExpectError(MICIUS_ERROR_INVALID_MODEL_DATA, "non-finite F32");
    token_embedding[0] = 0.0f;

    const void* aligned_embedding = entries[0].tensor_data;
    entries[0].tensor_data =
        static_cast<const unsigned char*>(aligned_embedding) + 1U;
    assert(!Micius_BindModelTensorsV1(
        context,
        entries.data(),
        static_cast<uint32_t>(entries.size()),
        sizeof(MiciusTensorEntryV1),
        MICIUS_ABI_VERSION));
    ExpectError(MICIUS_ERROR_INVALID_ARGUMENT, "misaligned tensor data pointer");
    entries[0].tensor_data = aligned_embedding;

    const void* aligned_scales = entries[3].tensor_data;
    entries[3].tensor_data =
        static_cast<const unsigned char*>(aligned_scales) + 1U;
    assert(!Micius_BindModelTensorsV1(
        context,
        entries.data(),
        static_cast<uint32_t>(entries.size()),
        sizeof(MiciusTensorEntryV1),
        MICIUS_ABI_VERSION));
    ExpectError(MICIUS_ERROR_INVALID_ARGUMENT, "misaligned tensor data pointer");
    entries[3].tensor_data = aligned_scales;

    assert(Micius_BindModelTensorsV1(
        context,
        entries.data(),
        static_cast<uint32_t>(entries.size()),
        sizeof(MiciusTensorEntryV1),
        MICIUS_ABI_VERSION));

    struct ExtendedTensorEntry {
        MiciusTensorEntryV1 base;
        uint64_t extension_sentinel;
    };
    std::vector<ExtendedTensorEntry> extended_entries;
    extended_entries.reserve(entries.size());
    for (const auto& entry : entries) {
        extended_entries.push_back({entry, UINT64_C(0x1122334455667788)});
    }
    assert(Micius_BindModelTensorsV1(
        context,
        reinterpret_cast<const MiciusTensorEntryV1*>(extended_entries.data()),
        static_cast<uint32_t>(extended_entries.size()),
        sizeof(ExtendedTensorEntry),
        MICIUS_ABI_VERSION));

    CheckPreparedWeightBinding(context, config);
    const auto prepared_before_rejection=micius::internal::GetPreparedWeightStats(context);
    // A rejected replacement is transactional: the last valid model remains
    // bound and usable.
    entries[2].byte_size -= 1U;
    assert(!Micius_BindModelTensorsV1(
        context,
        entries.data(),
        static_cast<uint32_t>(entries.size()),
        sizeof(MiciusTensorEntryV1),
        MICIUS_ABI_VERSION));
    entries[2].byte_size += 1U;
    const auto prepared_after_rejection=micius::internal::GetPreparedWeightStats(context);
    assert(prepared_before_rejection.extra_bytes==prepared_after_rejection.extra_bytes &&
        prepared_before_rejection.preparation_seconds==prepared_after_rejection.preparation_seconds);
    CheckPreparedWeightBinding(context, config);

    // Scale descriptors are copied into the engine at the validated boundary.
    // Invalidating the caller's source after a successful bind must neither
    // change the active model nor make the prevalidated path observe bad data.
    quant_scales[0].scale = std::numeric_limits<float>::quiet_NaN();
    quant_scales[0].zero_point = 1;
    quant_scales[0].padding[0] = 1;

    assert(Micius_GetSequenceLengthV1(context) == 0);
    assert(Micius_PrefillTokenV1(context, 1) == MICIUS_STATE_TOKEN_ACCEPTED);
    assert(Micius_GetSequenceLengthV1(context) == 1);

    std::vector<float> layer0_key(config.hidden_dim, 0.0f);
    std::vector<float> layer0_value(config.hidden_dim, -1.0f);
    std::vector<float> layer1_key(config.hidden_dim, 0.0f);
    std::vector<float> layer1_value(config.hidden_dim, -1.0f);
    assert(Micius_CopyReferenceKVV1(
        context,
        0,
        0,
        layer0_key.data(),
        layer0_value.data(),
        layer0_key.size()));
    assert(Micius_CopyReferenceKVV1(
        context,
        1,
        0,
        layer1_key.data(),
        layer1_value.data(),
        layer1_key.size()));
    const float normalized_sum =
        static_cast<float>(config.hidden_dim) /
        std::sqrt(1.0f + config.rms_norm_epsilon);
    for (uint32_t i = 0; i < config.hidden_dim; ++i) {
        const float expected_layer0_key = normalized_sum *
            static_cast<float>(static_cast<int32_t>(i % 9U) - 4);
        const float expected_layer1_key = normalized_sum *
            static_cast<float>(static_cast<int32_t>(i % 7U) + 1);
        assert(std::fabs(layer0_key[i] - expected_layer0_key) < 0.01f);
        assert(std::fabs(layer1_key[i] - expected_layer1_key) < 0.02f);
        assert(layer0_value[i] == 0.0f);
        assert(layer1_value[i] == 0.0f);
    }
    assert(!Micius_CopyReferenceKVV1(
        context,
        config.num_layers,
        0,
        layer0_key.data(),
        layer0_value.data(),
        layer0_key.size()));
    ExpectError(MICIUS_ERROR_INVALID_ARGUMENT, "not available");

    std::vector<float> logits(config.vocab_size, -1.0f);
    assert(Micius_CopyLastLogitsV1(context, logits.data(), logits.size()));
    assert(logits[kExpectedToken] > 63.0f);
    assert(logits[0] == 0.0f);

    uint32_t output_token = 0xDEADBEEFU;
    assert(Micius_DecodeOneV1(context, 1, &output_token) == MICIUS_STATE_YIELD_TOKEN);
    assert(output_token == kExpectedToken);
    assert(Micius_GetSequenceLengthV1(context) == 2);

    std::vector<float> rotated_key(config.hidden_dim, 0.0f);
    std::vector<float> rotated_value(config.hidden_dim, -1.0f);
    assert(Micius_CopyReferenceKVV1(
        context,
        0,
        1,
        rotated_key.data(),
        rotated_value.data(),
        rotated_key.size()));
    for (uint32_t i = 0; i < config.head_dim; i += 2U) {
        const float raw0 = normalized_sum *
            static_cast<float>(static_cast<int32_t>(i % 9U) - 4);
        const float raw1 = normalized_sum *
            static_cast<float>(static_cast<int32_t>((i + 1U) % 9U) - 4);
        const float theta = 1.0f / std::pow(
            config.rope_freq_base,
            static_cast<float>(i) / static_cast<float>(config.head_dim));
        const float expected0 = raw0 * std::cos(theta) - raw1 * std::sin(theta);
        const float expected1 = raw1 * std::cos(theta) + raw0 * std::sin(theta);
        assert(std::fabs(rotated_key[i] - expected0) < 0.05f);
        assert(std::fabs(rotated_key[i + 1U] - expected1) < 0.05f);
        assert(rotated_value[i] == 0.0f);
        assert(rotated_value[i + 1U] == 0.0f);
    }

    std::vector<float> hidden(config.hidden_dim, 0.0f);
    output_token = 0xA5A5A5A5U;
    assert(Micius_Step(context, 1, hidden.data(), &output_token) == MICIUS_STATE_YIELD_TOKEN);
    assert(output_token == kExpectedToken);
    assert(Micius_GetSequenceLengthV1(context) == 3);

    assert(Micius_Step(context, config.vocab_size, hidden.data(), &output_token) ==
           MICIUS_STATE_INVALID_ARGUMENT);
    ExpectError(MICIUS_ERROR_INVALID_ARGUMENT, "outside");

    assert(Micius_ResetEngineV1(context));
    assert(Micius_GetSequenceLengthV1(context) == 0);
    assert(!Micius_CopyLastLogitsV1(context, logits.data(), logits.size()));
    ExpectError(MICIUS_ERROR_NOT_READY, "no successful");

    std::vector<float> first_logits(config.vocab_size, 0.0f);
    std::vector<float> second_logits(config.vocab_size, 0.0f);
    assert(Micius_DecodeOneV1(context, 1, &output_token) == MICIUS_STATE_YIELD_TOKEN);
    assert(output_token == kExpectedToken);
    assert(Micius_CopyLastLogitsV1(context, first_logits.data(), first_logits.size()));
    assert(Micius_ResetEngineV1(context));
    assert(Micius_DecodeOneV1(context, 1, &output_token) == MICIUS_STATE_YIELD_TOKEN);
    assert(output_token == kExpectedToken);
    assert(Micius_CopyLastLogitsV1(context, second_logits.data(), second_logits.size()));
    assert(first_logits == second_logits);

    assert(Micius_ResetEngineV1(context));
    for (uint32_t i = 0; i < config.max_seq_len; ++i) {
        assert(Micius_PrefillTokenV1(context, 1) == MICIUS_STATE_TOKEN_ACCEPTED);
    }
    assert(Micius_GetSequenceLengthV1(context) == config.max_seq_len);
    output_token = 0x12345678U;
    assert(Micius_DecodeOneV1(context, 1, &output_token) == MICIUS_STATE_CONTEXT_FULL);
    assert(output_token == 0x12345678U);
    ExpectError(MICIUS_ERROR_CONTEXT_FULL, "max_seq_len");
    assert(Micius_CopyLastLogitsV1(context, logits.data(), logits.size()));

    MiciusRuntimeStats stats{};
    Micius_GetRuntimeStats(context, &stats);
    assert(stats.kv_slots_used == config.max_seq_len);

    Micius_DestroyEngine(context);
}

void TestAsymmetricGQAReferenceOracle()
{
    MiciusConfig config = MakeMinimalConfig();
    config.vocab_size = 16;
    config.max_seq_len = 4;
    config.num_layers = 1;
    config.num_heads = 2;
    config.num_kv_heads = 1;
    config.head_dim = 32;
    config.rope_dimension_count = 32;

    MiciusContext* context = Micius_CreateEngineV1(
        &config,
        sizeof(config),
        MICIUS_ABI_VERSION);
    assert(context != nullptr);

    const size_t hidden = config.hidden_dim;
    const size_t kv_dim = static_cast<size_t>(config.num_kv_heads) * config.head_dim;
    const size_t intermediate = config.intermediate_dim;
    std::vector<float> embedding(static_cast<size_t>(config.vocab_size) * hidden, 0.0f);
    std::vector<float> norm(hidden, 1.0f);
    std::vector<int8_t> q(hidden * hidden, int8_t{0});
    std::vector<int8_t> k(kv_dim * hidden, int8_t{0});
    std::vector<int8_t> v(kv_dim * hidden, int8_t{0});
    std::vector<int8_t> o(hidden * hidden, int8_t{0});
    std::vector<int8_t> gate(intermediate * hidden, int8_t{0});
    std::vector<int8_t> up(intermediate * hidden, int8_t{0});
    std::vector<int8_t> down(hidden * intermediate, int8_t{0});
    std::vector<int8_t> output(static_cast<size_t>(config.vocab_size) * hidden, int8_t{0});
    std::vector<MiciusQ8Block32Scale> scales(
        hidden * (hidden / 32U),
        MiciusQ8Block32Scale{1.0f, 0, {0, 0}});

    constexpr uint32_t kToken0 = 1;
    constexpr uint32_t kToken1 = 2;
    embedding[static_cast<size_t>(kToken0) * hidden + 0U] = 1.0f;
    embedding[static_cast<size_t>(kToken0) * hidden + 1U] = 2.0f;
    embedding[static_cast<size_t>(kToken0) * hidden + 2U] = 1.0f;
    embedding[static_cast<size_t>(kToken1) * hidden + 0U] = 2.0f;
    embedding[static_cast<size_t>(kToken1) * hidden + 1U] = -1.0f;
    embedding[static_cast<size_t>(kToken1) * hidden + 2U] = 3.0f;

    q[0U * hidden + 0U] = 1;
    q[32U * hidden + 1U] = 1;
    k[0U * hidden + 0U] = 1;
    k[1U * hidden + 1U] = 1;
    v[0U * hidden + 2U] = 1;
    o[0U * hidden + 0U] = 1;
    o[1U * hidden + 32U] = 1;
    gate[0U * hidden + 0U] = 1;
    up[0U * hidden + 1U] = 1;
    down[2U * intermediate + 0U] = 1;
    output[5U * hidden + 0U] = 1;
    output[6U * hidden + 1U] = 1;
    output[7U * hidden + 2U] = 1;

    std::vector<MiciusTensorEntryV1> entries;
    entries.reserve(20);
    auto add_f32 = [&](const char* name, const float* data, uint64_t rows, uint64_t columns) {
        MiciusTensorEntryV1 entry{};
        entry.tensor_name = name;
        entry.tensor_data = data;
        entry.byte_size = static_cast<size_t>(rows) *
            static_cast<size_t>(columns == 0 ? 1U : columns) * sizeof(float);
        entry.data_type = MICIUS_TENSOR_F32;
        entry.rank = columns == 0 ? 1U : 2U;
        entry.dimensions[0] = rows;
        entry.dimensions[1] = columns;
        entry.row_stride_bytes = columns == 0
            ? 0U
            : static_cast<size_t>(columns) * sizeof(float);
        entries.push_back(entry);
    };
    auto add_weight = [&](const char* name, const int8_t* data, uint64_t rows, uint64_t columns) {
        MiciusTensorEntryV1 entry{};
        entry.tensor_name = name;
        entry.tensor_data = data;
        entry.byte_size = static_cast<size_t>(rows * columns);
        entry.data_type = MICIUS_TENSOR_Q8_32_WEIGHT;
        entry.rank = 2;
        entry.dimensions[0] = rows;
        entry.dimensions[1] = columns;
        entry.row_stride_bytes = static_cast<size_t>(columns);
        entries.push_back(entry);
    };
    auto add_scales = [&](const char* name, uint64_t rows, uint64_t columns) {
        MiciusTensorEntryV1 entry{};
        entry.tensor_name = name;
        entry.tensor_data = scales.data();
        entry.byte_size = static_cast<size_t>(rows * columns) * sizeof(MiciusQ8Block32Scale);
        entry.data_type = MICIUS_TENSOR_Q8_32_SCALE;
        entry.rank = 2;
        entry.dimensions[0] = rows;
        entry.dimensions[1] = columns;
        entry.row_stride_bytes = static_cast<size_t>(columns) * sizeof(MiciusQ8Block32Scale);
        entries.push_back(entry);
    };

    add_f32("token_embd.weight.f32", embedding.data(), config.vocab_size, config.hidden_dim);
    add_f32("output_norm.weight", norm.data(), config.hidden_dim, 0);
    add_weight("output.weight", output.data(), config.vocab_size, config.hidden_dim);
    add_scales("output.scales", config.vocab_size, 2);
    add_weight("blk.0.attn_q.weight", q.data(), config.hidden_dim, config.hidden_dim);
    add_scales("blk.0.attn_q.scales", config.hidden_dim, 2);
    add_weight("blk.0.attn_k.weight", k.data(), kv_dim, config.hidden_dim);
    add_scales("blk.0.attn_k.scales", kv_dim, 2);
    add_weight("blk.0.attn_v.weight", v.data(), kv_dim, config.hidden_dim);
    add_scales("blk.0.attn_v.scales", kv_dim, 2);
    add_weight("blk.0.attn_output.weight", o.data(), config.hidden_dim, config.hidden_dim);
    add_scales("blk.0.attn_output.scales", config.hidden_dim, 2);
    add_weight("blk.0.ffn_gate.weight", gate.data(), config.intermediate_dim, config.hidden_dim);
    add_scales("blk.0.ffn_gate.scales", config.intermediate_dim, 2);
    add_weight("blk.0.ffn_up.weight", up.data(), config.intermediate_dim, config.hidden_dim);
    add_scales("blk.0.ffn_up.scales", config.intermediate_dim, 2);
    add_weight("blk.0.ffn_down.weight", down.data(), config.hidden_dim, config.intermediate_dim);
    add_scales("blk.0.ffn_down.scales", config.hidden_dim, 2);
    add_f32("blk.0.attn_norm.weight", norm.data(), config.hidden_dim, 0);
    add_f32("blk.0.ffn_norm.weight", norm.data(), config.hidden_dim, 0);

    assert(Micius_BindModelTensorsV1(
        context,
        entries.data(),
        static_cast<uint32_t>(entries.size()),
        sizeof(MiciusTensorEntryV1),
        MICIUS_ABI_VERSION));
    assert(Micius_PrefillTokenV1(context, kToken0) == MICIUS_STATE_TOKEN_ACCEPTED);
    uint32_t generated = 0;
    assert(Micius_DecodeOneV1(context, kToken1, &generated) == MICIUS_STATE_YIELD_TOKEN);

    auto rms_inverse = [&](double square_sum) {
        return 1.0 / std::sqrt(square_sum / static_cast<double>(hidden) +
                               config.rms_norm_epsilon);
    };
    const double inv0 = rms_inverse(1.0 + 4.0 + 1.0);
    const double inv1 = rms_inverse(4.0 + 1.0 + 9.0);
    const double n0_x = 1.0 * inv0;
    const double n0_y = 2.0 * inv0;
    const double n0_v = 1.0 * inv0;
    const double n1_x = 2.0 * inv1;
    const double n1_y = -1.0 * inv1;
    const double n1_v = 3.0 * inv1;
    const double cosine = std::cos(1.0);
    const double sine = std::sin(1.0);
    const double k1_x = n1_x * cosine - n1_y * sine;
    const double k1_y = n1_y * cosine + n1_x * sine;
    const double attention_scale = 1.0 / std::sqrt(static_cast<double>(config.head_dim));

    auto attention_value = [&](double query_raw) {
        const double qx = query_raw * cosine;
        const double qy = query_raw * sine;
        const double score0 = (qx * n0_x + qy * n0_y) * attention_scale;
        const double score1 = (qx * k1_x + qy * k1_y) * attention_scale;
        const double maximum = std::max(score0, score1);
        const double probability0 = std::exp(score0 - maximum);
        const double probability1 = std::exp(score1 - maximum);
        return (probability0 * n0_v + probability1 * n1_v) /
               (probability0 + probability1);
    };

    std::vector<double> residual(hidden, 0.0);
    residual[0] = 2.0 + attention_value(n1_x);
    residual[1] = -1.0 + attention_value(n1_y);
    residual[2] = 3.0;
    double ffn_square_sum = 0.0;
    for (double value : residual) ffn_square_sum += value * value;
    const double ffn_inverse = rms_inverse(ffn_square_sum);
    const double gate_value = residual[0] * ffn_inverse;
    const double up_value = residual[1] * ffn_inverse;
    residual[2] += gate_value / (1.0 + std::exp(-gate_value)) * up_value;

    double final_square_sum = 0.0;
    for (double value : residual) final_square_sum += value * value;
    const double final_inverse = rms_inverse(final_square_sum);
    std::vector<float> logits(config.vocab_size, 0.0f);
    assert(Micius_CopyLastLogitsV1(context, logits.data(), logits.size()));
    assert(std::fabs(logits[5] - static_cast<float>(residual[0] * final_inverse)) < 1.0e-4f);
    assert(std::fabs(logits[6] - static_cast<float>(residual[1] * final_inverse)) < 1.0e-4f);
    assert(std::fabs(logits[7] - static_cast<float>(residual[2] * final_inverse)) < 1.0e-4f);

    Micius_DestroyEngine(context);
}

void TestLegacyPrefixCreation()
{
    MiciusConfig config = MakeMinimalConfig();
    MiciusContext* context = Micius_CreateEngine(&config);
    assert(context != nullptr);
    assert(Micius_GetWorkerCount(context) == 0);
    Micius_DestroyEngine(context);
}

void TestDraftProposalBoundedReservation()
{
    DraftProposal proposal;
    for (uint32_t i = 0; i < DraftProposal::MAX_K_DRAFTS + 16U; ++i) {
        const int32_t index = proposal.SpawnNodeConcurrent(i, -1, i, i == 0);
        if (i < DraftProposal::MAX_K_DRAFTS) {
            assert(index == static_cast<int32_t>(i));
        } else {
            assert(index == -1);
        }
    }
    assert(proposal.num_valid_drafts.load(std::memory_order_acquire) ==
           DraftProposal::MAX_K_DRAFTS);
    assert(proposal.commit_counter.load(std::memory_order_acquire) ==
           DraftProposal::MAX_K_DRAFTS);

    auto* opaque = reinterpret_cast<MiciusDraftTree*>(&proposal);
    assert(Micius_GetTreeValidCount(opaque) == 0);
    ExpectError(MICIUS_ERROR_UNSUPPORTED_MODE, "speculative execution");
    Micius_AcceptTreeNode(opaque, 0);
    ExpectError(MICIUS_ERROR_UNSUPPORTED_MODE, "speculative execution");
}

} // namespace

int main()
{
    TestConfigAndAbiValidation();
    TestReferenceLifecycleAndFailClosedStep();
    TestDeterministicReferenceDecodeAndContextOwnership();
    TestAsymmetricGQAReferenceOracle();
    TestLegacyPrefixCreation();
    TestDraftProposalBoundedReservation();
    std::cout << "runtime contract tests passed\n";
    return 0;
}

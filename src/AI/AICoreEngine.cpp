#include "AICoreEngine.h"
#include "i25/ModelPanels.h"
#if defined(MICIUS_I19_SCORE_GROUP3)
#include "I19AttentionScore.h"
#endif
#if defined(MICIUS_I17_SCORE_REUSE)
#include "I17AttentionScore.h"
#endif
#if defined(MICIUS_I11_SCORE64)
#include "I11AttentionScore.h"
#endif
#if defined(MICIUS_I15_VALUE_TILE32)
#include "I15AttentionValue.h"
#endif
#if defined(MICIUS_I10_VALUE64)
#include "I10AttentionValue.h"
#endif
#include "AIComputKrnlInternal.h"
#include "PreparedGEMVInput.h"
#if MICIUS_I4_MODE
#include "I4MixedPrecision.h"
#if MICIUS_I1_MODE != 2 || defined(MICIUS_I2_DIAGNOSTICS) || defined(MICIUS_I3_INTERVENTION)
#error "I4 mixed precision requires I1 AVX2 without I2/I3 instrumentation"
#endif
#endif
#if defined(MICIUS_I3_INTERVENTION)
#include "I3Intervention.h"
#if MICIUS_I1_MODE != 2 || defined(MICIUS_I2_DIAGNOSTICS)
#error "I3 requires I1 AVX2 and I2 diagnostics OFF"
#endif
#endif
#if defined(MICIUS_I2_DIAGNOSTICS)
#include "I2Diagnostic.h"
#endif
#include "PreparedW8A8Input.h"
#ifndef MICIUS_I1_MODE
#define MICIUS_I1_MODE 0
#endif
#include "DecodeProfile.h"
#include <chrono>
#ifndef MICIUS_PW2_MODE
#define MICIUS_PW2_MODE 0
#endif
#ifndef MICIUS_PW2_BUDGET_MIB
#define MICIUS_PW2_BUDGET_MIB 512
#endif
#if MICIUS_PW2_MODE != 0 && (!defined(MICIUS_REFERENCE_GEMV_PREPARED_F64) || !defined(MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES) || defined(MICIUS_REFERENCE_GEMV_COMPACT_SCALES))
#error "PW2 requires r1 prepared input and prevalidated descriptor scales"
#endif

#include "HardwareTopology.h"

#include <iostream>
#include <cstring>
#include <cmath>
#include <cfloat>
#include <algorithm>
#include <unordered_map>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <sched.h>
    #include <pthread.h>
    #include <fstream>
    #include <string>
    #include <set>
#endif

namespace {

#if defined(MICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD)
constexpr uint32_t kAttentionScoreTimeQuadWidth = 4U;
#endif

#if defined(MICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE)
constexpr uint32_t kAttentionQueryF64CacheMinimumRows = 8U;
#endif

enum class ReferenceMathStatus
{
    OK,
    INVALID_MODEL,
    NUMERIC_ERROR
};

void FreePackedLayerStorage(std::vector<MiciusLayerWeights>& layers) noexcept
{
    for (auto& layer : layers) {
        if (layer.qkv_packed_weights) _mm_free(layer.qkv_packed_weights);
        if (layer.qkv_packed_scales) _mm_free(layer.qkv_packed_scales);
        if (layer.qkv_packed_compact_scales) {
            _mm_free(layer.qkv_packed_compact_scales);
        }
        layer.qkv_packed_weights = nullptr;
        layer.qkv_packed_scales = nullptr;
        layer.qkv_packed_compact_scales = nullptr;
    }
}

class PackedLayerCandidateGuard
{
public:
    explicit PackedLayerCandidateGuard(
        std::vector<MiciusLayerWeights>& layers) noexcept
        : layers_(layers)
    {
    }

    ~PackedLayerCandidateGuard()
    {
        FreePackedLayerStorage(layers_);
    }

    PackedLayerCandidateGuard(const PackedLayerCandidateGuard&) = delete;
    PackedLayerCandidateGuard& operator=(const PackedLayerCandidateGuard&) = delete;

private:
    std::vector<MiciusLayerWeights>& layers_;
};

[[maybe_unused]] ReferenceMathStatus ReferenceGEMV(
    const int8_t* weights,
    const QuantDescriptor* scales,
    const float* compact_scales,
    const float* input,
    float* output,
    uint32_t input_dim,
    uint32_t output_dim,
    bool scales_prevalidated,
    double* prepared_scratch,
    size_t prepared_capacity,
    const micius::internal::PreparedF64GEMVInput* supplied_prepared = nullptr,
    const micius::internal::PreparedF32GEMVWeights* prepared_weights = nullptr) noexcept
{
    MICIUS_D1_SCOPE(micius::d1::Current());
#if MICIUS_PW2_MODE == 0
    (void)prepared_weights;
#endif
#if !defined(MICIUS_REFERENCE_GEMV_PREPARED_F64)
    (void)prepared_scratch;
    (void)prepared_capacity;
    (void)supplied_prepared;
#endif
#if defined(MICIUS_REFERENCE_GEMV_AVX2)
#if defined(MICIUS_REFERENCE_GEMV_PREPARED_F64)
    using namespace micius::internal;
    if (!input || input_dim == 0 || output_dim == 0)
        return ReferenceMathStatus::INVALID_MODEL;
    PreparedF64GEMVInput local_prepared;
    const PreparedF64GEMVInput* prepared = supplied_prepared;
    if (!prepared) {
        MICIUS_D1_SCOPE(micius::d1::Preparation());
        if (!prepared_scratch || prepared_capacity < input_dim)
            return ReferenceMathStatus::INVALID_MODEL;
        const auto status = PrepareF64GEMVInput(
            {input, input_dim}, {prepared_scratch, prepared_capacity}, local_prepared);
        if (status != Q8_32_GEMVStatus::OK)
            return status == Q8_32_GEMVStatus::NUMERIC_ERROR
                ? ReferenceMathStatus::NUMERIC_ERROR : ReferenceMathStatus::INVALID_MODEL;
        prepared = &local_prepared;
    }
    if (prepared->size() != input_dim) return ReferenceMathStatus::INVALID_MODEL;
#if defined(MICIUS_REFERENCE_GEMV_COMPACT_SCALES)
    const auto gemv_status = scales_prevalidated
        ? GEMV_Q8_32_PreparedF64RowsCompactScales(
            weights, compact_scales, *prepared, output, output_dim, 0U, output_dim)
        : GEMV_Q8_32_PreparedF64Rows(
            weights, scales, *prepared, output, output_dim, 0U, output_dim);
#elif defined(MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES)
    (void)compact_scales;
    const auto gemv_status =
#if MICIUS_PW2_MODE != 0
        prepared_weights
        ? (scales_prevalidated
            ? GEMV_F32WeightsPreparedRowsPrevalidatedScales(
                *prepared_weights, scales, *prepared, output, 0U, output_dim)
            : GEMV_F32WeightsPreparedRows(
                *prepared_weights, scales, *prepared, output, 0U, output_dim))
        :
#endif
        scales_prevalidated
        ? GEMV_Q8_32_PreparedF64RowsPrevalidatedScales(
            weights, scales, *prepared, output, output_dim, 0U, output_dim)
        : GEMV_Q8_32_PreparedF64Rows(
            weights, scales, *prepared, output, output_dim, 0U, output_dim);
#else
    (void)compact_scales;
    (void)scales_prevalidated;
    const auto gemv_status = GEMV_Q8_32_PreparedF64Rows(
        weights, scales, *prepared, output, output_dim, 0U, output_dim);
#endif
#else
#if defined(MICIUS_REFERENCE_GEMV_COMPACT_SCALES)
#if defined(MICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2)
    const Q8_32_GEMVStatus gemv_status = scales_prevalidated
        ? micius::internal::GEMV_Q8_32_F32_AVX2_RowPairCompactScales(
            weights, compact_scales, input, output, input_dim, output_dim)
        : micius::internal::GEMV_Q8_32_F32_AVX2_RowPair(
            weights, scales, input, output, input_dim, output_dim);
#else
    const Q8_32_GEMVStatus gemv_status = scales_prevalidated
        ? micius::internal::GEMV_Q8_32_F32_AVX2_CompactScales(
            weights, compact_scales, input, output, input_dim, output_dim)
        : GEMV_Q8_32_F32_AVX2(
            weights, scales, input, output, input_dim, output_dim);
#endif
#else
    (void)compact_scales;
#if defined(MICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2)
#if defined(MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES)
    const Q8_32_GEMVStatus gemv_status = scales_prevalidated
        ? micius::internal::GEMV_Q8_32_F32_AVX2_RowPairPrevalidatedScales(
            weights, scales, input, output, input_dim, output_dim)
        : micius::internal::GEMV_Q8_32_F32_AVX2_RowPair(
            weights, scales, input, output, input_dim, output_dim);
#else
    (void)scales_prevalidated;
    const Q8_32_GEMVStatus gemv_status =
        micius::internal::GEMV_Q8_32_F32_AVX2_RowPair(
            weights, scales, input, output, input_dim, output_dim);
#endif
#else
#if defined(MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES)
    const Q8_32_GEMVStatus gemv_status = scales_prevalidated
        ? micius::internal::GEMV_Q8_32_F32_AVX2_PrevalidatedScales(
            weights, scales, input, output, input_dim, output_dim)
        : GEMV_Q8_32_F32_AVX2(
            weights, scales, input, output, input_dim, output_dim);
#else
    (void)scales_prevalidated;
    const Q8_32_GEMVStatus gemv_status = GEMV_Q8_32_F32_AVX2(
        weights,
        scales,
        input,
        output,
        input_dim,
        output_dim);
#endif
#endif
#endif
#endif // MICIUS_REFERENCE_GEMV_PREPARED_F64
    switch (gemv_status) {
        case Q8_32_GEMVStatus::OK:
            return ReferenceMathStatus::OK;
        case Q8_32_GEMVStatus::INVALID_ARGUMENT:
        case Q8_32_GEMVStatus::INVALID_MODEL:
            return ReferenceMathStatus::INVALID_MODEL;
        case Q8_32_GEMVStatus::NUMERIC_ERROR:
            return ReferenceMathStatus::NUMERIC_ERROR;
    }
    return ReferenceMathStatus::NUMERIC_ERROR;
#else
    (void)scales_prevalidated;
    (void)compact_scales;
    if (!weights || !scales || !input || !output || input_dim == 0 || output_dim == 0) {
        return ReferenceMathStatus::INVALID_MODEL;
    }

    const uint32_t blocks_per_row = input_dim / 32U + (input_dim % 32U != 0 ? 1U : 0U);
    for (uint32_t row = 0; row < output_dim; ++row) {
        double sum = 0.0;
        const size_t weight_row = static_cast<size_t>(row) * input_dim;
        const size_t scale_row = static_cast<size_t>(row) * blocks_per_row;
        for (uint32_t block = 0; block < blocks_per_row; ++block) {
            const QuantDescriptor& descriptor = scales[scale_row + block];
            if (descriptor.zero_point != 0 || !std::isfinite(descriptor.scale)) {
                return ReferenceMathStatus::INVALID_MODEL;
            }
            const uint32_t begin = block * 32U;
            const uint32_t end = std::min(begin + 32U, input_dim);
            for (uint32_t column = begin; column < end; ++column) {
                if (!std::isfinite(input[column])) {
                    return ReferenceMathStatus::NUMERIC_ERROR;
                }
                sum += static_cast<double>(input[column]) *
                       static_cast<double>(weights[weight_row + column]) *
                       static_cast<double>(descriptor.scale);
            }
        }
        if (!std::isfinite(sum) ||
            sum > std::numeric_limits<float>::max() ||
            sum < -std::numeric_limits<float>::max()) {
            return ReferenceMathStatus::NUMERIC_ERROR;
        }
        output[row] = static_cast<float>(sum);
    }
    return ReferenceMathStatus::OK;
#endif
}

bool HasReferenceScales(
    const QuantDescriptor* descriptors,
    const float* compact_scales) noexcept
{
#if defined(MICIUS_REFERENCE_GEMV_COMPACT_SCALES)
    return descriptors == nullptr && compact_scales != nullptr;
#else
    return descriptors != nullptr && compact_scales == nullptr;
#endif
}

ReferenceMathStatus ReferenceRMSNorm(
    float* output,
    const float* input,
    const float* weight,
    uint32_t dimension,
    float epsilon) noexcept
{
    if (!output || !input || !weight || dimension == 0 ||
        !std::isfinite(epsilon) || epsilon <= 0.0f) {
        return ReferenceMathStatus::INVALID_MODEL;
    }

    double square_sum = 0.0;
    for (uint32_t i = 0; i < dimension; ++i) {
        if (!std::isfinite(input[i]) || !std::isfinite(weight[i])) {
            return ReferenceMathStatus::NUMERIC_ERROR;
        }
        const double value = input[i];
        square_sum += value * value;
    }
    const double denominator = square_sum / static_cast<double>(dimension) + epsilon;
    if (!std::isfinite(denominator) || denominator <= 0.0) {
        return ReferenceMathStatus::NUMERIC_ERROR;
    }
    const double inverse_rms = 1.0 / std::sqrt(denominator);
    for (uint32_t i = 0; i < dimension; ++i) {
        const double value = static_cast<double>(input[i]) * inverse_rms * weight[i];
        if (!std::isfinite(value) ||
            value > std::numeric_limits<float>::max() ||
            value < -std::numeric_limits<float>::max()) {
            return ReferenceMathStatus::NUMERIC_ERROR;
        }
        output[i] = static_cast<float>(value);
    }
    return ReferenceMathStatus::OK;
}

ReferenceMathStatus ReferenceApplyRoPE(
    float* values,
    const RoPEFreqCache& cache,
    uint32_t position,
    uint32_t heads, bool split_half) noexcept
{
    if (!values || !cache.cos_table || !cache.sin_table ||
        cache.head_dim == 0 || (cache.head_dim & 1U) != 0 ||
        position >= cache.max_seq_len) {
        return ReferenceMathStatus::INVALID_MODEL;
    }

    const float* cos_row = cache.cos_table + static_cast<size_t>(position) * cache.head_dim;
    const float* sin_row = cache.sin_table + static_cast<size_t>(position) * cache.head_dim;
    for (uint32_t head = 0; head < heads; ++head) {
        float* head_values = values + static_cast<size_t>(head) * cache.head_dim;
        for (uint32_t i = 0; i < cache.head_dim; i += 2U) {
            const uint32_t first = split_half ? i / 2U : i;
            const uint32_t second = split_half ? first + cache.head_dim / 2U : i + 1U;
            const float x0 = head_values[first];
            const float x1 = head_values[second];
            const float cosine = cos_row[i];
            const float sine = sin_row[i];
            if (!std::isfinite(x0) || !std::isfinite(x1) ||
                !std::isfinite(cosine) || !std::isfinite(sine)) {
                return ReferenceMathStatus::NUMERIC_ERROR;
            }
            head_values[first] = x0 * cosine - x1 * sine;
            head_values[second] = x1 * cosine + x0 * sine;
        }
    }
    return ReferenceMathStatus::OK;
}

ReferenceStepResult ToStepResult(ReferenceMathStatus status) noexcept
{
    switch (status) {
        case ReferenceMathStatus::OK: return ReferenceStepResult::SUCCESS;
        case ReferenceMathStatus::INVALID_MODEL: return ReferenceStepResult::INVALID_MODEL_DATA;
        case ReferenceMathStatus::NUMERIC_ERROR: return ReferenceStepResult::NUMERIC_ERROR;
    }
    return ReferenceStepResult::NUMERIC_ERROR;
}

} // namespace

AIEngineFront::AIEngineFront(const MiciusConfig& cfg)
    : config(cfg), scheduler(this) // 初始化调度器
{
    auto checked_product = [](size_t a, size_t b) -> size_t {
        if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
            throw std::overflow_error("engine allocation size overflow");
        }
        return a * b;
    };

    try {
        hidden_dim = config.hidden_dim;
        speculative_enabled = config.execution_mode == MICIUS_EXECUTION_SPECULATIVE;

        if (!InitRoPECache(
                &rope_cache,
                config.max_seq_len,
                config.rope_dimension_count,
                config.rope_freq_base)) {
            throw std::bad_alloc();
        }

        if (speculative_enabled) {
            const size_t ffn_elements = checked_product(
                static_cast<size_t>(hidden_dim), config.intermediate_dim);
            const size_t output_elements = checked_product(
                static_cast<size_t>(config.vocab_size), hidden_dim);
            const size_t dummy_elements = std::max(ffn_elements, output_elements);
            safe_dummy_scale_count =
                dummy_elements / 64U + (dummy_elements % 64U != 0 ? 1U : 0U);
            const size_t dummy_scale_bytes = checked_product(
                safe_dummy_scale_count,
                sizeof(QuantDescriptor));
            const size_t dummy_fp32_bytes = checked_product(
                static_cast<size_t>(hidden_dim),
                sizeof(float));

            safe_dummy_weights = static_cast<int8_t*>(_mm_malloc(dummy_elements, 64));
            safe_dummy_scales = static_cast<QuantDescriptor*>(
                _mm_malloc(dummy_scale_bytes, 64));
            safe_dummy_fp32 = static_cast<float*>(
                _mm_malloc(dummy_fp32_bytes, 64));
            if (!safe_dummy_weights || !safe_dummy_scales || !safe_dummy_fp32) {
                throw std::bad_alloc();
            }
            std::fill_n(safe_dummy_weights, dummy_elements, int8_t{0});
            for (size_t i = 0; i < safe_dummy_scale_count; ++i) {
                std::construct_at(safe_dummy_scales + i, 0.0f, 0);
            }
            std::fill_n(safe_dummy_fp32, hidden_dim, 1.0f);
            engine_quant_weights = safe_dummy_weights;
            engine_scales = safe_dummy_scales;

            const size_t pool_bytes = checked_product(
                checked_product(static_cast<size_t>(config.kv_pool_size_mb), 1024), 1024);
            accepted_slots_history = new uint32_t[config.max_seq_len]{};
            mem_pool = new HeteroMemPool(pool_bytes);
            kv_cache = new KVCacheRingBuffer(config.max_seq_len, hidden_dim);
            shared_proposal[0] = mem_pool->GetDraftBuffer(0);
            shared_proposal[1] = mem_pool->GetDraftBuffer(1);
            if (!shared_proposal[0] || !shared_proposal[1]) {
                throw std::runtime_error("proposal buffers were not constructed");
            }
            data_front = new AiDataFront(kv_cache, shared_proposal[0]);

            topology_plan = HardwareTopology::GenerateDeploymentPlan();
            worker_cores = topology_plan.worker_cores;
            if (config.worker_threads > 0 &&
                static_cast<size_t>(config.worker_threads) < worker_cores.size()) {
                worker_cores.resize(static_cast<size_t>(config.worker_threads));
            }
            if (worker_cores.empty()) {
                throw std::runtime_error("speculative mode requires at least one worker");
            }

            topology_plan.worker_cores = worker_cores;
            recommended_shadow_core = topology_plan.recommended_shadow_core;
            local_queues = std::vector<ChaseLevDeque<FrontierTask, 1024>>(worker_cores.size());
            worker_ctxs.reserve(worker_cores.size());
            worker_threads.reserve(worker_cores.size());

            for (size_t i = 0; i < worker_cores.size(); ++i) {
                WorkerContext* ctx = static_cast<WorkerContext*>(_mm_malloc(sizeof(WorkerContext), 64));
                if (!ctx) throw std::bad_alloc();
                try {
                    std::construct_at(
                        ctx,
                        hidden_dim,
                        config.vocab_size,
                        config.max_seq_len,
                        static_cast<uint32_t>(1337U + i * 19U));
                } catch (...) {
                    _mm_free(ctx);
                    throw;
                }
                worker_ctxs.push_back(ctx);
            }

            for (size_t i = 0; i < worker_cores.size(); ++i) {
                worker_threads.emplace_back(&AIEngineFront::WorkerRoutine, this, static_cast<int>(i));
            }
        } else {
            const uint64_t kv_dim64 =
                static_cast<uint64_t>(config.num_kv_heads) * config.head_dim;
            const uint64_t qkv_dim64 =
                static_cast<uint64_t>(hidden_dim) + 2U * kv_dim64;
            if (kv_dim64 > std::numeric_limits<uint32_t>::max() ||
                qkv_dim64 > std::numeric_limits<uint32_t>::max()) {
                throw std::overflow_error("reference QKV dimensions exceed uint32_t");
            }
            reference_kv_dim = static_cast<uint32_t>(kv_dim64);
            reference_qkv_dim = static_cast<uint32_t>(qkv_dim64);
            reference_kv_elements = checked_product(
                checked_product(config.num_layers, config.max_seq_len),
                reference_kv_dim);

            reference_k_cache = MakeAlignedArray<float>(reference_kv_elements);
            reference_v_cache = MakeAlignedArray<float>(reference_kv_elements);
            reference_residual = MakeAlignedArray<float>(hidden_dim);
            reference_normed = MakeAlignedArray<float>(hidden_dim);
            reference_qkv = MakeAlignedArray<float>(reference_qkv_dim);
            reference_attention = MakeAlignedArray<float>(hidden_dim);
            reference_projection = MakeAlignedArray<float>(hidden_dim);
            reference_gate = MakeAlignedArray<float>(config.intermediate_dim);
            reference_up = MakeAlignedArray<float>(config.intermediate_dim);
            reference_logits = MakeAlignedArray<float>(config.vocab_size);
#if defined(MICIUS_I19_SCORE_GROUP3)
            const size_t i19_copies = config.head_dim==64 && config.num_heads/config.num_kv_heads==3 &&
                config.max_seq_len>=128 && config.max_seq_len<=8192 ? 4U : 1U;
            reference_scores = MakeAlignedArray<float>(checked_product(config.max_seq_len,i19_copies));
#else
            reference_scores = MakeAlignedArray<float>(config.max_seq_len);
#endif
#if MICIUS_I1_MODE
            const auto i1_capacity=std::max(hidden_dim,config.intermediate_dim);
            i1_input=MakeAlignedArray<int8_t>(i1_capacity);
            i1_scales=MakeAlignedArray<double>((static_cast<size_t>(i1_capacity)+31U)/32U);
#endif
#if defined(MICIUS_REFERENCE_GEMV_PREPARED_F64)
            reference_gemv_input = MakeAlignedArray<double>(
                std::max(hidden_dim, config.intermediate_dim));
#endif
#if defined(MICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE)
            reference_query_f64 = MakeAlignedArray<double>(config.head_dim);
#endif

            std::fill_n(reference_k_cache.get(), reference_kv_elements, 0.0f);
            std::fill_n(reference_v_cache.get(), reference_kv_elements, 0.0f);
            std::fill_n(reference_residual.get(), hidden_dim, 0.0f);
            std::fill_n(reference_normed.get(), hidden_dim, 0.0f);
            std::fill_n(reference_qkv.get(), reference_qkv_dim, 0.0f);
            std::fill_n(reference_attention.get(), hidden_dim, 0.0f);
            std::fill_n(reference_projection.get(), hidden_dim, 0.0f);
            std::fill_n(reference_gate.get(), config.intermediate_dim, 0.0f);
            std::fill_n(reference_up.get(), config.intermediate_dim, 0.0f);
            std::fill_n(reference_logits.get(), config.vocab_size, 0.0f);
            std::fill_n(reference_scores.get(), config.max_seq_len, 0.0f);
        }
    } catch (...) {
        global_terminate.store(true, std::memory_order_release);
        for (auto& thread : worker_threads) {
            if (thread.joinable()) thread.join();
        }
        for (WorkerContext* ctx : worker_ctxs) {
            std::destroy_at(ctx);
            _mm_free(ctx);
        }
        worker_ctxs.clear();
        delete data_front;
        delete kv_cache;
        delete mem_pool;
        data_front = nullptr;
        kv_cache = nullptr;
        mem_pool = nullptr;
        DestroyRoPECache(&rope_cache);
        delete[] accepted_slots_history;
        accepted_slots_history = nullptr;
        if (safe_dummy_fp32) _mm_free(safe_dummy_fp32);
        if (safe_dummy_scales) {
            std::destroy_n(safe_dummy_scales, safe_dummy_scale_count);
            _mm_free(safe_dummy_scales);
        }
        if (safe_dummy_weights) _mm_free(safe_dummy_weights);
        safe_dummy_fp32 = nullptr;
        safe_dummy_scales = nullptr;
        safe_dummy_weights = nullptr;
        throw;
    }
}

AIEngineFront::~AIEngineFront()
{
    global_terminate.store(true, std::memory_order_release);

    for (auto& t : worker_threads) {
        if (t.joinable()) t.join();
    }

    //释放RoPE物理页
    DestroyRoPECache(&rope_cache);

    if (safe_dummy_weights) _mm_free(safe_dummy_weights);
    if (safe_dummy_scales) {
        std::destroy_n(safe_dummy_scales, safe_dummy_scale_count);
        _mm_free(safe_dummy_scales);
    }
    if (safe_dummy_fp32) _mm_free(safe_dummy_fp32);
    delete[] accepted_slots_history;

    for (auto ctx : worker_ctxs) {
        //显式调用析构函数，智能指针成员会自动触发AlignedDeleter
        ctx->~WorkerContext();
        //释放placement new所占用的外层内存
        _mm_free(ctx);
    }

    delete data_front;
    delete kv_cache;
    delete mem_pool;

    FreePackedLayerStorage(model_layers);
}

// Validated V1 weight binding and packing. This entry is private and receives
// ownership of every checked private scale buffer in the same transaction.
bool AIEngineFront::BindValidatedTensors(
    const MiciusTensorEntry* entries,
    uint32_t count,
    std::vector<OwnedScaleBuffer>&& scale_buffers, bool split_half_rope)
{
    // The public v0.6.3 tensor contract is reference-only and Q8_32. The
    // dormant speculative kernels still use their historical Q8_64 layout and
    // therefore must not be rebound through this path.
    if (speculative_enabled || !entries || count == 0) return false;
    if (static_cast<size_t>(config.num_layers) >
        (std::numeric_limits<size_t>::max() - 1U) / 7U) {
        return false;
    }
    const size_t expected_scale_buffer_count =
        static_cast<size_t>(config.num_layers) * 7U + 1U;
    if (scale_buffers.size() != expected_scale_buffer_count) return false;

    std::unordered_map<const void*, size_t> unconsumed_scale_buffers;
    unconsumed_scale_buffers.reserve(scale_buffers.size());
    for (const OwnedScaleBuffer& buffer : scale_buffers) {
#if defined(MICIUS_REFERENCE_GEMV_COMPACT_SCALES)
        const void* scale_storage = buffer.compact_data.get();
        const bool has_wrong_storage = buffer.data != nullptr;
#else
        const void* scale_storage = buffer.data.get();
        const bool has_wrong_storage = buffer.compact_data != nullptr;
#endif
        if (!scale_storage || has_wrong_storage || buffer.count == 0U ||
            !unconsumed_scale_buffers.emplace(scale_storage, buffer.count).second) {
            return false;
        }
    }
    auto consume_scale_buffer = [&](const void* pointer, size_t expected_count) {
        const auto found = unconsumed_scale_buffers.find(pointer);
        if (found == unconsumed_scale_buffers.end() || found->second != expected_count) {
            return false;
        }
        unconsumed_scale_buffers.erase(found);
        return true;
    };

    std::unordered_map<std::string, const void*> dict;
    dict.reserve(count);
    for (uint32_t i = 0; i < count; ++i) {
        if (!entries[i].tensor_name || !entries[i].tensor_data) return false;
        if (!dict.emplace(entries[i].tensor_name, entries[i].tensor_data).second) return false;
    }

    auto has = [&](const std::string& name) { return dict.find(name) != dict.end(); };
    if (!has("token_embd.weight.f32") || !has("output_norm.weight")) return false;
    const bool has_output = has("output.weight") && has("output.scales");
    const bool has_tied_output = has("token_embd.weight") && has("token_embd.scales");
    if (!has_output && !has_tied_output) return false;

    for (uint32_t l = 0; l < config.num_layers; ++l) {
        const std::string pfx = "blk." + std::to_string(l) + ".";
        const char* required[] = {
            "attn_q.weight", "attn_q.scales", "attn_k.weight", "attn_k.scales",
            "attn_v.weight", "attn_v.scales", "attn_output.weight", "attn_output.scales",
            "ffn_gate.weight", "ffn_gate.scales", "ffn_up.weight", "ffn_up.scales",
            "ffn_down.weight", "ffn_down.scales", "attn_norm.weight", "ffn_norm.weight"
        };
        for (const char* suffix : required) {
            if (!has(pfx + suffix)) return false;
        }
    }

    // Build an entirely separate candidate. Validation or allocation failure
    // leaves the previously bound model and its session untouched. The guard
    // also covers allocation failures thrown by strings and hash containers.
    std::vector<MiciusLayerWeights> candidate_layers(config.num_layers);
    PackedLayerCandidateGuard candidate_guard(candidate_layers);

    uint32_t q_dim = config.num_heads * config.head_dim;
    uint32_t kv_dim = config.num_kv_heads * config.head_dim;
    uint32_t total_qkv_dim = q_dim + kv_dim * 2;
    uint32_t num_blocks = (config.hidden_dim + 31U) / 32U;

    auto fetch_tensor = [&](const std::string& name) -> const void* {
        auto it = dict.find(name);
        return it != dict.end() ? it->second : nullptr;
    };

    for (uint32_t l = 0; l < config.num_layers; ++l) {
        std::string pfx = "blk." + std::to_string(l) + ".";
        
        candidate_layers[l].split_half_rope = split_half_rope;
        candidate_layers[l].q_bias = static_cast<const float*>(fetch_tensor(pfx + "attn_q.bias"));
        candidate_layers[l].k_bias = static_cast<const float*>(fetch_tensor(pfx + "attn_k.bias"));
        candidate_layers[l].v_bias = static_cast<const float*>(fetch_tensor(pfx + "attn_v.bias"));
        //64字节对齐
        if (static_cast<size_t>(total_qkv_dim) >
                std::numeric_limits<size_t>::max() / config.hidden_dim ||
            static_cast<size_t>(total_qkv_dim) >
                std::numeric_limits<size_t>::max() / num_blocks) {
            return false;
        }
        const size_t packed_weight_count =
            static_cast<size_t>(total_qkv_dim) * config.hidden_dim;
        const size_t packed_scale_count =
            static_cast<size_t>(total_qkv_dim) * num_blocks;
        if (packed_scale_count > std::numeric_limits<size_t>::max() /
#if defined(MICIUS_REFERENCE_GEMV_COMPACT_SCALES)
                sizeof(float)) {
#else
                sizeof(QuantDescriptor)) {
#endif
            return false;
        }
        int8_t* p_w = static_cast<int8_t*>(
            _mm_malloc(packed_weight_count * sizeof(int8_t), 64));
        if (!p_w) throw std::bad_alloc();
        candidate_layers[l].qkv_packed_weights = p_w;

#if defined(MICIUS_REFERENCE_GEMV_COMPACT_SCALES)
        float* p_s = static_cast<float*>(
            _mm_malloc(packed_scale_count * sizeof(float), 64));
        if (!p_s) throw std::bad_alloc();
        candidate_layers[l].qkv_packed_compact_scales = p_s;
#else
        QuantDescriptor* p_s = static_cast<QuantDescriptor*>(
            _mm_malloc(packed_scale_count * sizeof(QuantDescriptor), 64));
        if (!p_s) throw std::bad_alloc();
        candidate_layers[l].qkv_packed_scales = p_s;
#endif

        const int8_t* w_q = (const int8_t*)fetch_tensor(pfx + "attn_q.weight");
        const int8_t* w_k = (const int8_t*)fetch_tensor(pfx + "attn_k.weight");
        const int8_t* w_v = (const int8_t*)fetch_tensor(pfx + "attn_v.weight");
#if defined(MICIUS_REFERENCE_GEMV_COMPACT_SCALES)
        const float* s_q = (const float*)fetch_tensor(pfx + "attn_q.scales");
        const float* s_k = (const float*)fetch_tensor(pfx + "attn_k.scales");
        const float* s_v = (const float*)fetch_tensor(pfx + "attn_v.scales");
        constexpr size_t scale_element_size = sizeof(float);
#else
        const QuantDescriptor* s_q = (const QuantDescriptor*)fetch_tensor(pfx + "attn_q.scales");
        const QuantDescriptor* s_k = (const QuantDescriptor*)fetch_tensor(pfx + "attn_k.scales");
        const QuantDescriptor* s_v = (const QuantDescriptor*)fetch_tensor(pfx + "attn_v.scales");
        constexpr size_t scale_element_size = sizeof(QuantDescriptor);
#endif

        const size_t q_scale_count = static_cast<size_t>(q_dim) * num_blocks;
        const size_t kv_scale_count = static_cast<size_t>(kv_dim) * num_blocks;
        if (w_q && w_k && w_v && s_q && s_k && s_v &&
            consume_scale_buffer(s_q, q_scale_count) &&
            consume_scale_buffer(s_k, kv_scale_count) &&
            consume_scale_buffer(s_v, kv_scale_count)) {
            // GGUF LLaMA Q/K tensors are already converted to the adjacent-pair
            // row order consumed by ReferenceApplyRoPE. A second permutation
            // here corrupts every non-symmetric Q/K projection.
            std::memcpy(p_w, w_q, static_cast<size_t>(q_dim) * config.hidden_dim);
            std::memcpy(
                p_s,
                s_q,
                static_cast<size_t>(q_dim) * num_blocks * scale_element_size);
            std::memcpy(
                p_w + static_cast<size_t>(q_dim) * config.hidden_dim,
                w_k,
                static_cast<size_t>(kv_dim) * config.hidden_dim);
            std::memcpy(
                p_s + static_cast<size_t>(q_dim) * num_blocks,
                s_k,
                static_cast<size_t>(kv_dim) * num_blocks * scale_element_size);
            std::memcpy(
                p_w + static_cast<size_t>(q_dim + kv_dim) * config.hidden_dim,
                w_v,
                static_cast<size_t>(kv_dim) * config.hidden_dim);
            std::memcpy(
                p_s + static_cast<size_t>(q_dim + kv_dim) * num_blocks,
                s_v,
                static_cast<size_t>(kv_dim) * num_blocks * scale_element_size);
        } else {
            return false;
        }

        //对连续的张量保持零拷贝
        candidate_layers[l].o_weight = (const int8_t*)fetch_tensor(pfx + "attn_output.weight");
#if defined(MICIUS_REFERENCE_GEMV_COMPACT_SCALES)
        candidate_layers[l].o_compact_scales = (const float*)fetch_tensor(pfx + "attn_output.scales");
#else
        candidate_layers[l].o_scales = (const QuantDescriptor*)fetch_tensor(pfx + "attn_output.scales");
#endif
        candidate_layers[l].gate_weight = (const int8_t*)fetch_tensor(pfx + "ffn_gate.weight");
#if defined(MICIUS_REFERENCE_GEMV_COMPACT_SCALES)
        candidate_layers[l].gate_compact_scales = (const float*)fetch_tensor(pfx + "ffn_gate.scales");
#else
        candidate_layers[l].gate_scales = (const QuantDescriptor*)fetch_tensor(pfx + "ffn_gate.scales");
#endif
        candidate_layers[l].up_weight = (const int8_t*)fetch_tensor(pfx + "ffn_up.weight");
#if defined(MICIUS_REFERENCE_GEMV_COMPACT_SCALES)
        candidate_layers[l].up_compact_scales = (const float*)fetch_tensor(pfx + "ffn_up.scales");
#else
        candidate_layers[l].up_scales = (const QuantDescriptor*)fetch_tensor(pfx + "ffn_up.scales");
#endif
        candidate_layers[l].down_weight = (const int8_t*)fetch_tensor(pfx + "ffn_down.weight");
#if defined(MICIUS_REFERENCE_GEMV_COMPACT_SCALES)
        candidate_layers[l].down_compact_scales = (const float*)fetch_tensor(pfx + "ffn_down.scales");
#else
        candidate_layers[l].down_scales = (const QuantDescriptor*)fetch_tensor(pfx + "ffn_down.scales");
#endif

        const uint32_t intermediate_blocks =
            config.intermediate_dim / 32U +
            (config.intermediate_dim % 32U != 0U ? 1U : 0U);
        if (!consume_scale_buffer(
#if defined(MICIUS_REFERENCE_GEMV_COMPACT_SCALES)
                candidate_layers[l].o_compact_scales,
#else
                candidate_layers[l].o_scales,
#endif
                static_cast<size_t>(config.hidden_dim) * num_blocks) ||
            !consume_scale_buffer(
#if defined(MICIUS_REFERENCE_GEMV_COMPACT_SCALES)
                candidate_layers[l].gate_compact_scales,
#else
                candidate_layers[l].gate_scales,
#endif
                static_cast<size_t>(config.intermediate_dim) * num_blocks) ||
            !consume_scale_buffer(
#if defined(MICIUS_REFERENCE_GEMV_COMPACT_SCALES)
                candidate_layers[l].up_compact_scales,
#else
                candidate_layers[l].up_scales,
#endif
                static_cast<size_t>(config.intermediate_dim) * num_blocks) ||
            !consume_scale_buffer(
#if defined(MICIUS_REFERENCE_GEMV_COMPACT_SCALES)
                candidate_layers[l].down_compact_scales,
#else
                candidate_layers[l].down_scales,
#endif
                static_cast<size_t>(config.hidden_dim) * intermediate_blocks)) {
            return false;
        }

        candidate_layers[l].rms_attn_weight = (const float*)fetch_tensor(pfx + "attn_norm.weight");
        candidate_layers[l].rms_ffn_weight = (const float*)fetch_tensor(pfx + "ffn_norm.weight");
    }

    const float* candidate_norm = (const float*)fetch_tensor("output_norm.weight");
    const float* candidate_embedding = (const float*)fetch_tensor("token_embd.weight.f32");

    const int8_t* w_out = (const int8_t*)fetch_tensor("output.weight");
#if defined(MICIUS_REFERENCE_GEMV_COMPACT_SCALES)
    const float* s_out = (const float*)fetch_tensor("output.scales");
#else
    const QuantDescriptor* s_out = (const QuantDescriptor*)fetch_tensor("output.scales");
#endif
    
    //如果没有找到output.weight，在Tied Embedding复用token_embd
    if (!w_out || !s_out) {
        w_out = (const int8_t*)fetch_tensor("token_embd.weight");
#if defined(MICIUS_REFERENCE_GEMV_COMPACT_SCALES)
        s_out = (const float*)fetch_tensor("token_embd.scales");
#else
        s_out = (const QuantDescriptor*)fetch_tensor("token_embd.scales");
#endif
    }

    if (!w_out || !s_out || !candidate_norm || !candidate_embedding ||
        !consume_scale_buffer(
            s_out,
            static_cast<size_t>(config.vocab_size) * num_blocks) ||
        !unconsumed_scale_buffers.empty()) {
        return false;
    }

    std::vector<micius::internal::PreparedF32GEMVWeights> candidate_prepared_weights;
    micius::internal::PreparedWeightStats candidate_weight_stats;
    candidate_weight_stats.selection = MICIUS_PW2_MODE;
    candidate_weight_stats.budget_bytes = uint64_t{MICIUS_PW2_BUDGET_MIB} * 1024U * 1024U;
#if MICIUS_PW2_MODE != 0
    const auto weight_prepare_begin = std::chrono::steady_clock::now();
    try {
        std::vector<micius::internal::WeightSource> sources;
        sources.reserve(candidate_layers.size());
        for (const auto& layer : candidate_layers) {
#if MICIUS_PW2_MODE == 1
            sources.push_back({{layer.qkv_packed_weights,
                static_cast<size_t>(total_qkv_dim) * hidden_dim}, total_qkv_dim, hidden_dim});
#else
            sources.push_back({{layer.o_weight,
                static_cast<size_t>(hidden_dim) * hidden_dim}, hidden_dim, hidden_dim});
#endif
        }
        using Result = micius::internal::WeightPrepareResult;
        const auto result = micius::internal::PrepareF32WeightGroup(
            sources, candidate_weight_stats.budget_bytes, candidate_prepared_weights,
            candidate_weight_stats.extra_bytes);
        // INVALID is never treated as an optional resource failure.
        if (result == Result::INVALID) return false;
        candidate_weight_stats.status = result == Result::READY ? "ready"
            : result == Result::BUDGET ? "fallback_budget" : "fallback_allocation";
        candidate_weight_stats.matrices = static_cast<uint32_t>(candidate_prepared_weights.size());
    } catch (const std::bad_alloc&) {
        candidate_weight_stats.status = "fallback_allocation";
    }
    candidate_weight_stats.preparation_seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - weight_prepare_begin).count();
#endif

    std::unique_ptr<micius::i25::Panels> candidate_panels;
#if defined(MICIUS_I25_PANEL_GEMV)
    // Full-model shape decision is made before any ownership transition.
    // Vocab tails retain the complete I22 path; there is no mixed partial bind.
    if(config.vocab_size%4U==0 && total_qkv_dim%4U==0) {
        candidate_panels=std::make_unique<micius::i25::Panels>();
        auto& p=*candidate_panels;
        p.matrices.reserve(size_t(config.num_layers)*5U+1U);
        p.retired.reserve(size_t(config.num_layers)*7U+1U);
        p.scratch.resize(size_t(std::max(hidden_dim,config.intermediate_dim))*4U);
        p.stats.scratch_bytes=p.scratch.size();
        for(uint32_t i=0;i<config.num_layers;++i) {
            const auto& layer=candidate_layers[i];
            p.Pack(layer.qkv_packed_weights,layer.qkv_packed_scales,total_qkv_dim,hidden_dim);
            p.Pack(layer.o_weight,layer.o_scales,hidden_dim,hidden_dim);
            p.Pack(layer.gate_weight,layer.gate_scales,config.intermediate_dim,hidden_dim);
            p.Pack(layer.up_weight,layer.up_scales,config.intermediate_dim,hidden_dim);
            p.Pack(layer.down_weight,layer.down_scales,hidden_dim,config.intermediate_dim);
            const std::string prefix="blk."+std::to_string(i)+".";
            p.Retire(dict.at(prefix+"attn_q.weight"),size_t(hidden_dim)*hidden_dim);
            p.Retire(dict.at(prefix+"attn_k.weight"),size_t(kv_dim)*hidden_dim);
            p.Retire(dict.at(prefix+"attn_v.weight"),size_t(kv_dim)*hidden_dim);
            p.Retire(layer.o_weight,size_t(hidden_dim)*hidden_dim);
            p.Retire(layer.gate_weight,size_t(config.intermediate_dim)*hidden_dim);
            p.Retire(layer.up_weight,size_t(config.intermediate_dim)*hidden_dim);
            p.Retire(layer.down_weight,size_t(hidden_dim)*config.intermediate_dim);
        }
        p.Pack(w_out,s_out,config.vocab_size,hidden_dim);
        p.Retire(w_out,size_t(config.vocab_size)*hidden_dim);
        p.stats.active=true;
    }
#endif

    const float* old_norm = norm_weight_final;
    const float* old_embedding = token_embd_weight;
    const int8_t* old_output_weights = engine_quant_weights;
    const QuantDescriptor* old_output_scales = engine_scales;
    const float* old_output_compact_scales = engine_compact_scales;
    const bool old_model_bound = model_bound;
    const bool old_scales_prevalidated = reference_scales_prevalidated;

    model_layers.swap(candidate_layers);
    i25_panels.swap(candidate_panels);
    norm_weight_final = candidate_norm;
    token_embd_weight = candidate_embedding;
    engine_quant_weights = w_out;
#if defined(MICIUS_REFERENCE_GEMV_COMPACT_SCALES)
    engine_scales = nullptr;
    engine_compact_scales = s_out;
#else
    engine_scales = s_out;
    engine_compact_scales = nullptr;
#endif
    model_bound = true;
    reference_scales_prevalidated = true;
    if (!ResetState()) {
        model_layers.swap(candidate_layers);
        i25_panels.swap(candidate_panels);
        norm_weight_final = old_norm;
        token_embd_weight = old_embedding;
        engine_quant_weights = old_output_weights;
        engine_scales = old_output_scales;
        engine_compact_scales = old_output_compact_scales;
        model_bound = old_model_bound;
        reference_scales_prevalidated = old_scales_prevalidated;
        return false;
    }

    // candidate_layers now holds the previous model's owned packed QKV data;
    // candidate_guard releases it on every exit path.
    if(i25_panels) {
        // No old-consumer pointer remains reachable after this commit. Both
        // scale copies and the staging row-major QKV copy can now disappear.
        FreePackedLayerStorage(model_layers);
        for(auto& l:model_layers) {
            l.o_weight=l.gate_weight=l.up_weight=l.down_weight=nullptr;
            l.o_scales=l.gate_scales=l.up_scales=l.down_scales=nullptr;
            l.o_compact_scales=l.gate_compact_scales=l.up_compact_scales=l.down_compact_scales=nullptr;
        }
        engine_quant_weights=nullptr;engine_scales=nullptr;engine_compact_scales=nullptr;
        scale_buffers.clear();owned_scale_buffers.clear();
    } else owned_scale_buffers = std::move(scale_buffers);
    reference_prepared_weights.swap(candidate_prepared_weights);
    reference_weight_stats = candidate_weight_stats;
    return true;
}

inline bool AIEngineFront::ComputeStateTransition(WorkerContext* ctx, const FrontierTask& task, const float* parent_state, float* new_state)
{
    (void)parent_state;
    uint32_t num_layers = config.num_layers;       
    uint32_t n_heads = config.num_heads;
    uint32_t n_kv_heads = config.num_kv_heads;
    uint32_t head_dim = config.head_dim;
    uint32_t intermediate_dim = config.intermediate_dim; //FFN扩展维度

    //进行指针追踪前校验纪元防止主线程触发Drift Fallback重置shared_proposal，导致树路径追溯越界与读取脏数据
    if (task.gen_id != (uint32_t)scheduler.GetBarrier().GetCurrentEpoch()) return false;

    //指针追踪
    uint32_t seq_len = 0;
    uint32_t kv_offset = hidden_dim / 2; 

    for (uint32_t i = 0; i < history_len; ++i) {
        if (seq_len < config.max_seq_len) {
            ctx->k_ptrs[seq_len] = kv_cache->GetFP32Base() + (accepted_slots_history[i] * hidden_dim);
            ctx->v_ptrs[seq_len] = kv_cache->GetFP32Base() + (accepted_slots_history[i] * hidden_dim) + kv_offset;
            seq_len++;
        }
    }

    //接入当前Prefill/Draft计算管线的即时上下文
    if (task.parent_tree_idx != -1) {
        //先压入本批次Seed Task生成的KV槽位
        if (seq_len < config.max_seq_len) {
            uint32_t base_slot = pipeline_base_slots[task.tree_idx];
            ctx->k_ptrs[seq_len] = kv_cache->GetFP32Base() + (base_slot * hidden_dim);
            ctx->v_ptrs[seq_len] = kv_cache->GetFP32Base() + (base_slot * hidden_dim) + kv_offset;
            seq_len++;
        }

        //回溯草稿树路径不包括未生成的槽位
        uint32_t draft_len = 0;
        int32_t curr_idx = shared_proposal[task.tree_idx]->GetParent(task.parent_tree_idx);
        uint32_t temp_slots[DraftProposal::MAX_K_DRAFTS];
    
        while (curr_idx != -1) {
            temp_slots[draft_len++] = shared_proposal[task.tree_idx]->GetSlot(curr_idx);
            curr_idx = shared_proposal[task.tree_idx]->GetParent(curr_idx);
        }

        for (int i = (int)draft_len - 1; i >= 0; --i) {
            if (seq_len < config.max_seq_len) {
                ctx->k_ptrs[seq_len] = kv_cache->GetFP32Base() + (temp_slots[i] * hidden_dim);
                ctx->v_ptrs[seq_len] = kv_cache->GetFP32Base() + (temp_slots[i] * hidden_dim) + kv_offset;
                seq_len++;
            }
        }
    }

    uint32_t absolute_pos = seq_len; 

    //初始映射修正应当是全新的Embedding，而不是累加至无关的parent_state
    if (token_embd_weight) {
        const float* emb_row = token_embd_weight + (task.input_token * hidden_dim);
        for (uint32_t i = 0; i < hidden_dim; ++i) {
            new_state[i] = emb_row[i];
        }
    } else {
        for (uint32_t i = 0; i < hidden_dim; ++i) {
            new_state[i] = static_cast<float>(task.input_token) * 0.0001f;
        }
    }

    /* Transformer Layer Stack */
    for (uint32_t l = 0; l < num_layers; ++l) {
        
        //每4层检查一次全局Epoch状态，若管线已被推翻立即熔断循环
        //将极大避免长尾慢速Worker拖累整体的吞吐
        if ((l & 3) == 0 && l > 0) {
            if (task.gen_id != (uint32_t)scheduler.GetBarrier().GetCurrentEpoch()) {
                return false; 
            }
        }
        
        //提前将下一层QKV权重缓存至L2/L3，缓解内存瓶颈
        if (l + 1 < num_layers) {
            _mm_prefetch((const char*)model_layers[l+1].qkv_packed_weights, _MM_HINT_T0);
            _mm_prefetch((const char*)model_layers[l+1].gate_weight, _MM_HINT_T0);
        }

        const MiciusLayerWeights& lw = model_layers[l];

        //保持原始的安全假权重回退机制
        const float* rms_w_attn = lw.rms_attn_weight ? lw.rms_attn_weight : safe_dummy_fp32;
        const float* rms_w_ffn  = lw.rms_ffn_weight ? lw.rms_ffn_weight : safe_dummy_fp32;
        const int8_t* w_o = lw.o_weight ? lw.o_weight : safe_dummy_weights;
        const QuantDescriptor* s_o = lw.o_scales ? lw.o_scales : safe_dummy_scales;
        const int8_t* w_gate = lw.gate_weight ? lw.gate_weight : safe_dummy_weights;
        const QuantDescriptor* s_gate = lw.gate_scales ? lw.gate_scales : safe_dummy_scales;
        const int8_t* w_up = lw.up_weight ? lw.up_weight : safe_dummy_weights;
        const QuantDescriptor* s_up = lw.up_scales ? lw.up_scales : safe_dummy_scales;
        const int8_t* w_down = lw.down_weight ? lw.down_weight : safe_dummy_weights;
        const QuantDescriptor* s_down = lw.down_scales ? lw.down_scales : safe_dummy_scales;

        float* normed_state = ctx->qkv_workspace.get(); 
        RMSNorm_AVX2(normed_state, new_state, rms_w_attn, hidden_dim, 1e-5f);

        float* qkv_buffer = ctx->qkv_workspace.get() + hidden_dim; 
        ForwardQKV_Proj_RoPE_AVX2(
            lw.qkv_packed_weights, lw.qkv_packed_scales, normed_state, qkv_buffer, ctx->workspace_i8.get(),
            hidden_dim, n_heads, n_kv_heads, head_dim, absolute_pos, &rope_cache
        );

        float* curr_q = qkv_buffer;
        float* curr_k = qkv_buffer + (n_heads * head_dim);
        float* curr_v = curr_k + (n_kv_heads * head_dim);
        float* slot_base = kv_cache->GetFP32Base() + (task.alloc_slot * hidden_dim); 
        std::memcpy(slot_base, curr_k, (n_kv_heads * head_dim) * sizeof(float));
        std::memcpy(slot_base + kv_offset, curr_v, (n_kv_heads * head_dim) * sizeof(float));

        if (seq_len < config.max_seq_len) {
            ctx->k_ptrs[seq_len] = slot_base;
            ctx->v_ptrs[seq_len] = slot_base + kv_offset;
            seq_len++; 
        }

        float* attn_out = qkv_buffer; 
        TreeAttention_GQA_AVX2(
            curr_q, ctx->k_ptrs.get(), ctx->v_ptrs.get(), attn_out, 
            ctx->attn_score_workspace.get(), seq_len, n_heads, n_kv_heads, head_dim
        );

        float* o_proj_out = ctx->qkv_workspace.get(); 
        DraftGEMV_INT8_AVX2(w_o, s_o, attn_out, o_proj_out, ctx->workspace_i8.get(), hidden_dim, hidden_dim);
        for(uint32_t i = 0; i < hidden_dim; ++i) new_state[i] += o_proj_out[i];

        RMSNorm_AVX2(normed_state, new_state, rms_w_ffn, hidden_dim, 1e-5f);

        float* gate_out = ctx->ffn_workspace.get();
        float* up_out = ctx->ffn_workspace.get() + intermediate_dim;

        DraftGEMV_INT8_AVX2(w_gate, s_gate, normed_state, gate_out, ctx->workspace_i8.get(), hidden_dim, intermediate_dim);
        DraftGEMV_INT8_AVX2(w_up, s_up, normed_state, up_out, ctx->workspace_i8.get(), hidden_dim, intermediate_dim);

        SiLU_Mul_AVX2(gate_out, up_out, intermediate_dim);

        float* down_proj_out = ctx->qkv_workspace.get(); 
        DraftGEMV_INT8_AVX2(w_down, s_down, gate_out, down_proj_out, ctx->workspace_i8.get(), intermediate_dim, hidden_dim);
        
        for(uint32_t i = 0; i < hidden_dim; ++i) new_state[i] += down_proj_out[i];
    }

    if (norm_weight_final) {
        RMSNorm_AVX2(new_state, new_state, norm_weight_final, hidden_dim, 1e-5f);
    } else {
        RMSNorm_AVX2(new_state, new_state, safe_dummy_fp32, hidden_dim, 1e-5f);
    }
    
    return true;
}

void AIEngineFront::WorkerRoutine(int w_idx)
{
    //用独立的拓扑结构绑核
    HardwareTopology::SetThreadAffinity(topology_plan.worker_cores[w_idx]);
    
    WorkerContext* ctx = worker_ctxs[w_idx];
    int idle_spins = 0;

    while(!global_terminate.load(std::memory_order_acquire)) {
        
        //获取最新的全局Epoch
        uint32_t current_epoch = (uint32_t)scheduler.GetBarrier().GetCurrentEpoch();

        FrontierTask task;
        bool got = false;
        
        //尝试获取Seed Task
        if (seed_claimed.load(std::memory_order_acquire) == 0) {
            int expected = 0;
            if (seed_claimed.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
                task = global_seed_task;
                got = true;
            }
        }

        //尝试从本地队列弹出任务
        if(!got) {
            got = local_queues[w_idx].pop_bottom(task);
            if (got) ctx->tasks_processed++; 
        }
        
        //尝试任务窃取
        if(!got && topology_plan.worker_cores.size() > 1) {
            /* Xorshift32 极速伪随机发生器 */
            ctx->prng_state ^= ctx->prng_state << 13;
            ctx->prng_state ^= ctx->prng_state >> 17;
            ctx->prng_state ^= ctx->prng_state << 5;
            
            size_t victim = ctx->prng_state % topology_plan.worker_cores.size();
            if (victim != (size_t)w_idx) {
                got = local_queues[victim].steal_top(task);
                if (got) ctx->tasks_stolen++; 
            }
        }

        if(!got) {
            if (idle_spins < 50) {
                _mm_pause();
            } else if (idle_spins < 200) {
                for(int p = 0; p < 8; ++p) _mm_pause();
            } else {
                int power = std::min((idle_spins - 200) / 50 + 4, 12);
                int pause_count = 1 << power;
                for(int p = 0; p < pause_count; ++p) _mm_pause();
            }
            idle_spins++;
            continue; 
        }
        
        idle_spins = 0;
        
        //丢弃非当前Epoch的脏任务
        if (task.gen_id != current_epoch) {
            continue; 
        }
        
        DraftProposal* active_prop = shared_proposal[task.tree_idx];
        
        //如果提案容量已满，停止扩张
        if (active_prop->num_valid_drafts.load(std::memory_order_relaxed) >= DraftProposal::MAX_K_DRAFTS) {
            continue;
        }

        task.fusion_depth = 4;
        bool fusion_continue = true;

        //TLAB闭包逻辑不变
        auto tlab_alloc = [&]() -> uint32_t {
            if (ctx->tlab_count == 0) {
                ctx->tlab_count = kv_cache->BulkAllocateSlots(ctx->tlab_slots, 16);
                if (ctx->tlab_count == 0) return 0xFFFFFFFF;
            }
            return ctx->tlab_slots[--ctx->tlab_count];
        };
        
        auto tlab_free = [&](uint32_t slot) {
            if (ctx->tlab_count < 16) {
                ctx->tlab_slots[ctx->tlab_count++] = slot;
            } else {
                kv_cache->FreeSlot(slot);
            }
        };
        
        //融合循环
        while(fusion_continue && task.fusion_depth > 0) {

            fusion_continue = false; 
            
            //循环前二次校验纪元，防止长尾任务时大管线已经发生Tree Reject
            if(task.gen_id != (uint32_t)scheduler.GetBarrier().GetCurrentEpoch()) break; 
            
            float *p_state = kv_cache->GetFP32Base()+(task.parent_slot*hidden_dim), *m_state = kv_cache->GetFP32Base()+(task.alloc_slot*hidden_dim);
            //函数内部检测到Epoch跃迁并返回false时中止循环
            if (!ComputeStateTransition(ctx, task, p_state, m_state)) {
                break;
            }
            
            DraftGEMV_INT8_AVX2(this->engine_quant_weights, this->engine_scales, m_state, ctx->logits.get(), ctx->workspace_i8.get(), hidden_dim, config.vocab_size);
            
            if(task.gen_id != (uint32_t)scheduler.GetBarrier().GetCurrentEpoch()) break;
            kv_cache->WriteTreeSlotFP32(task.alloc_slot, m_state);
            
            uint32_t top_k[3];
            BranchlessTop3_AVX2(ctx->logits.get(), config.vocab_size, top_k);
            
            bool golden_survived = false;
            FrontierTask child_golden;
            
            uint32_t slot_0 = tlab_alloc();
            if (slot_0 == 0xFFFFFFFF) {
                active_prop->state_flags.store(3, std::memory_order_release);
                fusion_continue = false;
                break;
            }
            
            //处理k=0
            int32_t tree_idx_0 = active_prop->SpawnNodeConcurrent(top_k[0], task.parent_tree_idx, slot_0, task.is_golden);            
            
            if(tree_idx_0 != -1) {
                child_golden.gen_id = task.gen_id;
                child_golden.parent_slot = task.alloc_slot;
                child_golden.alloc_slot = slot_0;
                child_golden.input_token = top_k[0];
                child_golden.parent_tree_idx = tree_idx_0; 
                child_golden.branch_id = (task.branch_id == 0) ? tree_idx_0 : task.branch_id; 
                child_golden.fusion_depth = task.fusion_depth - 1;
                child_golden.tree_idx = task.tree_idx;
                child_golden.is_golden = task.is_golden;
                
                golden_survived = true;
            } else {
                tlab_free(slot_0);
            }

            //处理k=2和k=1
            for(int k=2; k>=1; --k) { 
                uint32_t new_slot = tlab_alloc();
                if (new_slot == 0xFFFFFFFF) {
                    active_prop->state_flags.store(3, std::memory_order_release);
                    fusion_continue = false;
                    break;
                }
                
                int32_t new_tree_idx = active_prop->SpawnNodeConcurrent(top_k[k], task.parent_tree_idx, new_slot, false);
                if(new_tree_idx == -1) {
                    tlab_free(new_slot);
                    break;
                }

                FrontierTask child;
                child.gen_id = task.gen_id;
                child.parent_slot = task.alloc_slot;
                child.alloc_slot = new_slot;
                child.input_token = top_k[k];
                child.parent_tree_idx = new_tree_idx;
                child.branch_id = new_tree_idx; 
                child.fusion_depth = task.fusion_depth - 1;
                child.tree_idx = task.tree_idx;
                child.is_golden = false;

                local_queues[w_idx].push_bottom(child);
            }

            if (golden_survived) {
                if (child_golden.fusion_depth > 0) {
                    task = child_golden;
                    fusion_continue = true;
                } else {
                    local_queues[w_idx].push_bottom(child_golden); 
                }
            }
        }
    }
}

void AIEngineFront::RegisterVerifier(void (*cb)(void*, void*, void*), void* wrapper_ctx, void* user_data)
{
    this->verifier_callback = cb;
    this->wrapper_context = wrapper_ctx;
    this->verifier_user_data = user_data;
}

bool AIEngineFront::ResetState() noexcept
{
    if (speculative_enabled || !reference_k_cache || !reference_v_cache ||
        !reference_residual || !reference_logits) return false;
#if defined(MICIUS_REFERENCE_GEMV_PREPARED_F64)
    if (!reference_gemv_input) return false;
#endif

    pipeline_initialized = false;
    awaiting_fallback_token = false;
    current_cpu_tree = 0;
    current_gpu_tree = 0;
    pipeline_roots[0] = 0xFFFFFFFF;
    pipeline_roots[1] = 0xFFFFFFFF;
    pipeline_base_slots[0] = 0xFFFFFFFF;
    pipeline_base_slots[1] = 0xFFFFFFFF;
    last_accepted_token = 0;
    expected_input_token = 0;
    history_len = 0;
    token_queue_head = 0;
    token_queue_tail = 0;
    seed_claimed.store(1, std::memory_order_release);

    std::fill_n(accepted_tokens_queue, DraftProposal::MAX_K_DRAFTS + 2, uint32_t{0});
#if defined(MICIUS_REFERENCE_GEMV_PREPARED_F64)
    std::fill_n(reference_gemv_input.get(),
                std::max(hidden_dim, config.intermediate_dim), 0.0);
#endif
    reference_seq_len = 0;
    reference_has_logits = false;
    std::fill_n(reference_k_cache.get(), reference_kv_elements, 0.0f);
    std::fill_n(reference_v_cache.get(), reference_kv_elements, 0.0f);
    std::fill_n(reference_residual.get(), hidden_dim, 0.0f);
    std::fill_n(reference_normed.get(), hidden_dim, 0.0f);
    std::fill_n(reference_qkv.get(), reference_qkv_dim, 0.0f);
    std::fill_n(reference_attention.get(), hidden_dim, 0.0f);
    std::fill_n(reference_projection.get(), hidden_dim, 0.0f);
    std::fill_n(reference_gate.get(), config.intermediate_dim, 0.0f);
    std::fill_n(reference_up.get(), config.intermediate_dim, 0.0f);
    std::fill_n(reference_logits.get(), config.vocab_size, 0.0f);
    std::fill_n(reference_scores.get(), config.max_seq_len, 0.0f);
#if defined(MICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE)
    if (reference_query_f64) {
        std::fill_n(reference_query_f64.get(), config.head_dim, 0.0);
    }
#endif
    return true;
}

ReferenceStepResult AIEngineFront::RunReferenceToken(
    uint32_t token,
    uint32_t* out_token) noexcept
{
    MICIUS_D1_SCOPE(micius::d1::Part::Forward);
    if (speculative_enabled || !model_bound ||
        model_layers.size() != config.num_layers ||
        !token_embd_weight || !norm_weight_final ||
        (!i25_panels && (!engine_quant_weights ||
        !HasReferenceScales(engine_scales, engine_compact_scales)))) {
        return ReferenceStepResult::MODEL_NOT_BOUND;
    }
    if (token >= config.vocab_size) return ReferenceStepResult::INVALID_MODEL_DATA;
    if (reference_seq_len >= config.max_seq_len) {
        return ReferenceStepResult::SEQUENCE_LIMIT_REACHED;
    }
    if (!reference_k_cache || !reference_v_cache || !reference_residual ||
        !reference_normed || !reference_qkv || !reference_attention ||
        !reference_projection || !reference_gate || !reference_up ||
        !reference_logits || !reference_scores) {
        return ReferenceStepResult::INVALID_MODEL_DATA;
    }
#if defined(MICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE)
    if (!reference_query_f64) return ReferenceStepResult::INVALID_MODEL_DATA;
#endif

#if defined(MICIUS_REFERENCE_GEMV_PREPARED_F64)
    if (!reference_gemv_input) return ReferenceStepResult::INVALID_MODEL_DATA;
#endif

    // Preconditions that do not execute the model preserve the last successful
    // logits. Once computation begins, partial results must never be exposed.
    reference_has_logits = false;

    // Invocation-local views cannot survive this synchronous call. A future
    // row executor must join/drain all consumers before returning or reset.
#if MICIUS_I4_MODE
    uint32_t i4_layer=0;
    auto i4_op=micius::i4::Op::QKV;
#endif
#if defined(MICIUS_I2_DIAGNOSTICS)
    micius::i2::Session i2(token, reference_seq_len, MICIUS_I1_MODE);
    if (!i2.good()) return ReferenceStepResult::INVALID_MODEL_DATA;
#endif
#if defined(MICIUS_I3_INTERVENTION)
    micius::i3::Session i3(token, reference_seq_len, config.num_layers);
    if (!i3.good()) return ReferenceStepResult::INVALID_MODEL_DATA;
#endif
#if MICIUS_I1_MODE
    if (!i1_input || !i1_scales) return ReferenceStepResult::INVALID_MODEL_DATA;
#if defined(MICIUS_I25_PANEL_GEMV)
    micius::i23::Expanded expanded;
    const micius::i23::Matrix* panel=nullptr;
#endif
    const auto prepare_i1 = [&](const float* x,uint32_t n,micius::internal::PreparedW8A8Input& view) {
        const size_t cap=std::max(hidden_dim,config.intermediate_dim);
        auto status=micius::internal::PrepareW8A8Input({x,n},{i1_input.get(),cap},
            {i1_scales.get(),(cap+31U)/32U},view);
#if defined(MICIUS_I25_PANEL_GEMV)
        if(status==Q8_32_GEMVStatus::OK && i25_panels)
            status=micius::i23::Expand(view,i25_panels->scratch,expanded);
#endif
        return status;
    };
    const auto gemv = [&](const int8_t* w,const QuantDescriptor* s,const float*,
        const float* x,float* y,uint32_t cols,uint32_t rows,bool validated,
        const micius::internal::PreparedW8A8Input* supplied=nullptr,
        const micius::internal::PreparedF32GEMVWeights* =nullptr) {
#if defined(MICIUS_I2_DIAGNOSTICS)
        const auto snapshot = i2.capture(x, cols, rows);
        if (!i2.good()) return ReferenceMathStatus::INVALID_MODEL;
#endif
        micius::internal::PreparedW8A8Input local;
        auto status=Q8_32_GEMVStatus::OK;
#if MICIUS_I4_MODE
        if (micius::i4::Select(micius::i4::CompiledPolicy(),i4_layer,i4_op)) {
            if (!micius::i4::RecordSelected(i4_layer,i4_op)) return ReferenceMathStatus::INVALID_MODEL;
            status=micius::i4::F64(w,s,x,y,cols,rows,validated);
        } else
#endif
#if defined(MICIUS_I3_INTERVENTION)
        if (!i3.begin(x,cols,rows)) return ReferenceMathStatus::INVALID_MODEL;
        if (i3.selected()) {
            status=micius::i3::F64(w,s,x,y,cols,rows,validated);
        } else
#endif
        {
        if (!supplied) { status=prepare_i1(x,cols,local);supplied=&local; }
        if (status==Q8_32_GEMVStatus::OK && supplied->size()!=cols)
            status=Q8_32_GEMVStatus::INVALID_ARGUMENT;
        if (status==Q8_32_GEMVStatus::OK) {
#if defined(MICIUS_I25_PANEL_GEMV)
            if(i25_panels) {
                if(!panel || panel->cols()!=cols || panel->rows()!=rows)
                    return ReferenceMathStatus::INVALID_MODEL;
                status=micius::i23::Consume(*panel,expanded,{y,rows});
            } else
#endif
            status=micius::internal::GEMV_W8A8Rows(w,s,*supplied,y,rows,0,rows,validated,MICIUS_I1_MODE==1);
        }
        }
#if defined(MICIUS_I3_INTERVENTION)
        if (status==Q8_32_GEMVStatus::OK && !i3.end(x,y)) return ReferenceMathStatus::INVALID_MODEL;
#endif
#if defined(MICIUS_I2_DIAGNOSTICS)
        if (status==Q8_32_GEMVStatus::OK && !i2.finish(snapshot,w,s,x,y,rows,supplied))
            return ReferenceMathStatus::INVALID_MODEL;
#endif
        return status==Q8_32_GEMVStatus::OK ? ReferenceMathStatus::OK :
            status==Q8_32_GEMVStatus::NUMERIC_ERROR ? ReferenceMathStatus::NUMERIC_ERROR : ReferenceMathStatus::INVALID_MODEL;
    };
#else
    const auto gemv = [&](const int8_t* w, const QuantDescriptor* scales,
                          const float* compact, const float* x, float* y,
                          uint32_t cols, uint32_t rows, bool validated,
                          const micius::internal::PreparedF64GEMVInput* prepared = nullptr,
                          const micius::internal::PreparedF32GEMVWeights* prepared_weights = nullptr) {
#if defined(MICIUS_I2_DIAGNOSTICS)
        const auto snapshot = i2.capture(x, cols, rows);
        if (!i2.good()) return ReferenceMathStatus::INVALID_MODEL;
#endif
        const auto status = ReferenceGEMV(w, scales, compact, x, y, cols, rows, validated,
                             reference_gemv_input.get(),
                             std::max(hidden_dim, config.intermediate_dim), prepared, prepared_weights);
#if defined(MICIUS_I2_DIAGNOSTICS)
        if (status==ReferenceMathStatus::OK && !i2.finish(snapshot,w,scales,x,y,rows))
            return ReferenceMathStatus::INVALID_MODEL;
#endif
        return status;
    };
#endif

    const uint32_t position = reference_seq_len;
    const uint32_t q_dim = hidden_dim;
    const uint32_t kv_dim = reference_kv_dim;
    const double attention_score_denominator =
        std::sqrt(static_cast<double>(config.head_dim));

    const float* embedding = token_embd_weight + static_cast<size_t>(token) * hidden_dim;
    for (uint32_t i = 0; i < hidden_dim; ++i) {
        if (!std::isfinite(embedding[i])) return ReferenceStepResult::NUMERIC_ERROR;
        reference_residual[i] = embedding[i];
    }
#if defined(MICIUS_I2_DIAGNOSTICS)
    if (!i2.boundary(-1,"embedding",reference_residual.get(),hidden_dim)) return ReferenceStepResult::INVALID_MODEL_DATA;
#endif
#if defined(MICIUS_I3_INTERVENTION)
    if (!i3.boundary(-1,"embedding",reference_residual.get(),hidden_dim)) return ReferenceStepResult::INVALID_MODEL_DATA;
#endif

    for (uint32_t layer_index = 0; layer_index < config.num_layers; ++layer_index) {
        const MiciusLayerWeights& layer = model_layers[layer_index];
        if ((!i25_panels && (!layer.qkv_packed_weights ||
            !HasReferenceScales(
                layer.qkv_packed_scales, layer.qkv_packed_compact_scales) ||
            !layer.o_weight ||
            !HasReferenceScales(layer.o_scales, layer.o_compact_scales) ||
            !layer.gate_weight ||
            !HasReferenceScales(layer.gate_scales, layer.gate_compact_scales) ||
            !layer.up_weight ||
            !HasReferenceScales(layer.up_scales, layer.up_compact_scales) ||
            !layer.down_weight ||
            !HasReferenceScales(layer.down_scales, layer.down_compact_scales))) ||
            !layer.rms_attn_weight || !layer.rms_ffn_weight) {
            return ReferenceStepResult::INVALID_MODEL_DATA;
        }

        ReferenceMathStatus math_status = ReferenceRMSNorm(
            reference_normed.get(),
            reference_residual.get(),
            layer.rms_attn_weight,
            hidden_dim,
            config.rms_norm_epsilon);
        if (math_status != ReferenceMathStatus::OK) return ToStepResult(math_status);

        MICIUS_D1_TAG(QKV);
#if MICIUS_I4_MODE
        i4_layer=layer_index;i4_op=micius::i4::Op::QKV;
#endif
#if defined(MICIUS_I3_INTERVENTION)
        i3.set(layer_index,micius::i3::Op::QKV);
#endif
#if defined(MICIUS_I2_DIAGNOSTICS)
        i2.set(static_cast<int>(layer_index), "qkv");
#endif
#if defined(MICIUS_I25_PANEL_GEMV)
        panel=i25_panels ? &i25_panels->matrices[size_t(layer_index)*5U+0U] : nullptr;
#endif
        math_status = gemv(
            layer.qkv_packed_weights,
            layer.qkv_packed_scales,
            layer.qkv_packed_compact_scales,
            reference_normed.get(),
            reference_qkv.get(),
            hidden_dim,
            reference_qkv_dim,
            reference_scales_prevalidated
#if MICIUS_PW2_MODE == 1
            , nullptr, reference_prepared_weights.empty() ? nullptr : &reference_prepared_weights[layer_index]
#endif
            );
        if (math_status != ReferenceMathStatus::OK) return ToStepResult(math_status);

        float* query = reference_qkv.get();
        float* key = reference_qkv.get() + q_dim;
        float* value = key + kv_dim;
        if (layer.q_bias) {
            for (uint32_t j = 0; j < q_dim; ++j) query[j] += layer.q_bias[j];
            for (uint32_t j = 0; j < kv_dim; ++j) {
                key[j] += layer.k_bias[j];
                value[j] += layer.v_bias[j];
                if (!std::isfinite(value[j])) return ReferenceStepResult::NUMERIC_ERROR;
            }
        }
        math_status = ReferenceApplyRoPE(query, rope_cache, position, config.num_heads, layer.split_half_rope);
        if (math_status != ReferenceMathStatus::OK) return ToStepResult(math_status);
        math_status = ReferenceApplyRoPE(key, rope_cache, position, config.num_kv_heads, layer.split_half_rope);
        if (math_status != ReferenceMathStatus::OK) return ToStepResult(math_status);
#if defined(MICIUS_I2_DIAGNOSTICS)
        if (!i2.boundary(static_cast<int>(layer_index),"qkv.rope",reference_qkv.get(),reference_qkv_dim))
            return ReferenceStepResult::INVALID_MODEL_DATA;
#endif

        const size_t current_kv_offset =
            (static_cast<size_t>(layer_index) * config.max_seq_len + position) * kv_dim;
        std::copy_n(key, kv_dim, reference_k_cache.get() + current_kv_offset);
        std::copy_n(value, kv_dim, reference_v_cache.get() + current_kv_offset);

        const uint32_t group_size = config.num_heads / config.num_kv_heads;
#if defined(MICIUS_I19_SCORE_GROUP3)
        bool i19_ready=false;double i19_maxima[3];
        const bool i19_eligible=group_size==3 && config.head_dim==64 &&
            config.max_seq_len<=8192 && position+1U>=128;
#endif
        for (uint32_t query_head = 0; query_head < config.num_heads; ++query_head) {
            const uint32_t kv_head = query_head / group_size;
            const float* query_values = query + static_cast<size_t>(query_head) * config.head_dim;
            double maximum_score = -std::numeric_limits<double>::infinity();

#if defined(MICIUS_I11_SCORE64)
            const size_t i11_key_offset =
                static_cast<size_t>(layer_index) * config.max_seq_len * kv_dim +
                static_cast<size_t>(kv_head) * config.head_dim;
#if defined(MICIUS_I19_SCORE_GROUP3)
            if(i19_eligible && query_head%3U==0) {
                i19_ready=TryPrepareAttentionScoresI19(query_values,
                    reference_k_cache.get()+i11_key_offset,kv_dim,
                    reference_scores.get()+config.max_seq_len,config.max_seq_len,
                    i19_maxima,position+1U,config.head_dim,group_size,attention_score_denominator);
            }
            I11ScoreStatus i11_score_status;
            if(i19_eligible && i19_ready) {
                std::copy_n(reference_scores.get()+size_t(1U+query_head%3U)*config.max_seq_len,
                    position+1U,reference_scores.get());
                maximum_score=i19_maxima[query_head%3U];
                i11_score_status=I11ScoreStatus::OK;
            } else i11_score_status=ComputeAttentionScoresI17(
#elif defined(MICIUS_I17_SCORE_REUSE)
            const auto i11_score_status = ComputeAttentionScoresI17(
#else
            const auto i11_score_status = ComputeAttentionScoresI11(
#endif
                query_values, reference_k_cache.get() + i11_key_offset, kv_dim,
                reference_scores.get(), position + 1U, config.head_dim,
                attention_score_denominator, &maximum_score);
            if (i11_score_status == I11ScoreStatus::NUMERIC_ERROR)
                return ReferenceStepResult::NUMERIC_ERROR;
            if (i11_score_status == I11ScoreStatus::UNSUPPORTED)
#endif
            {
#if defined(MICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD)
            uint32_t quad_time = 0U;
            if (position >= kAttentionScoreTimeQuadWidth - 1U) {
                const size_t first_key_offset =
                    static_cast<size_t>(layer_index) * config.max_seq_len * kv_dim +
                    static_cast<size_t>(kv_head) * config.head_dim;
                for (;
                     quad_time <= position - (kAttentionScoreTimeQuadWidth - 1U);
                     quad_time += kAttentionScoreTimeQuadWidth) {
                    const float* key0 =
                        reference_k_cache.get() + first_key_offset +
                        static_cast<size_t>(quad_time) * kv_dim;
                    const float* key1 = key0 + kv_dim;
                    const float* key2 = key1 + kv_dim;
                    const float* key3 = key2 + kv_dim;
                    double dots[kAttentionScoreTimeQuadWidth];
                    micius::internal::AttentionScoreDotTimeQuadF32(
                        query_values,
                        key0,
                        key1,
                        key2,
                        key3,
                        config.head_dim,
                        dots);

                    for (uint32_t lane = 0U;
                         lane < kAttentionScoreTimeQuadWidth;
                         ++lane) {
                        const double score =
                            dots[lane] / attention_score_denominator;
                        if (!std::isfinite(score)) {
                            return ReferenceStepResult::NUMERIC_ERROR;
                        }
                        reference_scores[quad_time + lane] =
                            static_cast<float>(score);
                        maximum_score = std::max(maximum_score, score);
                    }
                }
            }
            for (; quad_time <= position; ++quad_time) {
                const size_t kv_offset =
                    (static_cast<size_t>(layer_index) * config.max_seq_len +
                     quad_time) *
                        kv_dim +
                    static_cast<size_t>(kv_head) * config.head_dim;
                const float* key_values = reference_k_cache.get() + kv_offset;
                double dot = 0.0;
                for (uint32_t d = 0; d < config.head_dim; ++d) {
                    dot += static_cast<double>(query_values[d]) * key_values[d];
                }
                const double score = dot / attention_score_denominator;
                if (!std::isfinite(score)) {
                    return ReferenceStepResult::NUMERIC_ERROR;
                }
                reference_scores[quad_time] = static_cast<float>(score);
                maximum_score = std::max(maximum_score, score);
            }
#elif defined(MICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE)
            if (position >= kAttentionQueryF64CacheMinimumRows - 1U) {
                micius::internal::PrepareAttentionQueryF64(
                    query_values,
                    reference_query_f64.get(),
                    config.head_dim);
                for (uint32_t time = 0; time <= position; ++time) {
                    const size_t kv_offset =
                        (static_cast<size_t>(layer_index) * config.max_seq_len + time) *
                            kv_dim +
                        static_cast<size_t>(kv_head) * config.head_dim;
                    const float* key_values = reference_k_cache.get() + kv_offset;
                    const double dot =
                        micius::internal::AttentionScoreDotPreparedQueryF64(
                            reference_query_f64.get(),
                            key_values,
                            config.head_dim);
                    const double score = dot / attention_score_denominator;
                    if (!std::isfinite(score)) {
                        return ReferenceStepResult::NUMERIC_ERROR;
                    }
                    reference_scores[time] = static_cast<float>(score);
                    maximum_score = std::max(maximum_score, score);
                }
            } else
#endif
#if !defined(MICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD)
            {
                for (uint32_t time = 0; time <= position; ++time) {
                    const size_t kv_offset =
                        (static_cast<size_t>(layer_index) * config.max_seq_len + time) * kv_dim +
                        static_cast<size_t>(kv_head) * config.head_dim;
                    const float* key_values = reference_k_cache.get() + kv_offset;
                    double dot = 0.0;
                    for (uint32_t d = 0; d < config.head_dim; ++d) {
                        dot += static_cast<double>(query_values[d]) * key_values[d];
                    }
                    const double score = dot / attention_score_denominator;
                    if (!std::isfinite(score)) return ReferenceStepResult::NUMERIC_ERROR;
                    reference_scores[time] = static_cast<float>(score);
                    maximum_score = std::max(maximum_score, score);
                }
            }
#endif

            } // existing Score route when I11 is disabled or unsupported

            double probability_sum = 0.0;
            for (uint32_t time = 0; time <= position; ++time) {
                const double probability =
                    std::exp(static_cast<double>(reference_scores[time]) - maximum_score);
                if (!std::isfinite(probability)) return ReferenceStepResult::NUMERIC_ERROR;
                reference_scores[time] = static_cast<float>(probability);
                probability_sum += probability;
            }
            if (!std::isfinite(probability_sum) || probability_sum <= 0.0) {
                return ReferenceStepResult::NUMERIC_ERROR;
            }

            float* attention_head =
                reference_attention.get() + static_cast<size_t>(query_head) * config.head_dim;
#if defined(MICIUS_REFERENCE_ATTENTION_VALUE_AVX2)
            const size_t first_value_offset =
                static_cast<size_t>(layer_index) * config.max_seq_len * kv_dim +
                static_cast<size_t>(kv_head) * config.head_dim;
#if defined(MICIUS_I15_VALUE_TILE32)
            switch (AccumulateAttentionValuesI15(
#elif defined(MICIUS_I10_VALUE64)
            switch (AccumulateAttentionValuesI10(
#else
            switch (AccumulateAttentionValuesF32_AVX2(
#endif
                reference_scores.get(),
                probability_sum,
                reference_v_cache.get() + first_value_offset,
                kv_dim,
                attention_head,
                position + 1U,
                config.head_dim)) {
                case AttentionValueStatus::OK:
                    break;
                case AttentionValueStatus::INVALID_ARGUMENT:
                    return ReferenceStepResult::INVALID_MODEL_DATA;
                case AttentionValueStatus::NUMERIC_ERROR:
                    return ReferenceStepResult::NUMERIC_ERROR;
            }
#else
            std::fill_n(attention_head, config.head_dim, 0.0f);
            for (uint32_t time = 0; time <= position; ++time) {
                const size_t kv_offset =
                    (static_cast<size_t>(layer_index) * config.max_seq_len + time) * kv_dim +
                    static_cast<size_t>(kv_head) * config.head_dim;
                const float* value_values = reference_v_cache.get() + kv_offset;
                const double probability = reference_scores[time] / probability_sum;
                for (uint32_t d = 0; d < config.head_dim; ++d) {
                    const double accumulated =
                        static_cast<double>(attention_head[d]) + probability * value_values[d];
                    if (!std::isfinite(accumulated) ||
                        accumulated > std::numeric_limits<float>::max() ||
                        accumulated < -std::numeric_limits<float>::max()) {
                        return ReferenceStepResult::NUMERIC_ERROR;
                    }
                    attention_head[d] = static_cast<float>(accumulated);
                }
            }
#endif
        }

        MICIUS_D1_TAG(O);
#if MICIUS_I4_MODE
        i4_op=micius::i4::Op::O;
#endif
#if defined(MICIUS_I3_INTERVENTION)
        i3.set(layer_index,micius::i3::Op::O);
#endif
#if defined(MICIUS_I2_DIAGNOSTICS)
        i2.set(static_cast<int>(layer_index), "o");
#endif
#if defined(MICIUS_I25_PANEL_GEMV)
        panel=i25_panels ? &i25_panels->matrices[size_t(layer_index)*5U+1U] : nullptr;
#endif
        math_status = gemv(
            layer.o_weight,
            layer.o_scales,
            layer.o_compact_scales,
            reference_attention.get(),
            reference_projection.get(),
            hidden_dim,
            hidden_dim,
            reference_scales_prevalidated
#if MICIUS_PW2_MODE == 2
            , nullptr, reference_prepared_weights.empty() ? nullptr : &reference_prepared_weights[layer_index]
#endif
            );
        if (math_status != ReferenceMathStatus::OK) return ToStepResult(math_status);
        for (uint32_t i = 0; i < hidden_dim; ++i) {
            const double value_sum =
                static_cast<double>(reference_residual[i]) + reference_projection[i];
            if (!std::isfinite(value_sum) ||
                value_sum > std::numeric_limits<float>::max() ||
                value_sum < -std::numeric_limits<float>::max()) {
                return ReferenceStepResult::NUMERIC_ERROR;
            }
            reference_residual[i] = static_cast<float>(value_sum);
        }

        math_status = ReferenceRMSNorm(
            reference_normed.get(),
            reference_residual.get(),
            layer.rms_ffn_weight,
            hidden_dim,
            config.rms_norm_epsilon);
        if (math_status != ReferenceMathStatus::OK) return ToStepResult(math_status);
        {
#if MICIUS_I1_MODE
#if defined(MICIUS_I3_INTERVENTION)
        if (!i3.boundary(static_cast<int>(layer_index),"attention_residual",reference_residual.get(),hidden_dim) ||
            !i3.start_ffn(layer_index,reference_normed.get(),hidden_dim)) return ReferenceStepResult::INVALID_MODEL_DATA;
#endif
        micius::internal::PreparedW8A8Input ffn_prepared;
        const auto i1_status=prepare_i1(reference_normed.get(),hidden_dim,ffn_prepared);
        if (i1_status!=Q8_32_GEMVStatus::OK)
            return i1_status==Q8_32_GEMVStatus::NUMERIC_ERROR ? ReferenceStepResult::NUMERIC_ERROR : ReferenceStepResult::INVALID_MODEL_DATA;
#elif defined(MICIUS_REFERENCE_GEMV_PREPARED_F64)
        micius::internal::PreparedF64GEMVInput ffn_prepared;
        const auto preparation_status = [&]() {
            MICIUS_D1_SCOPE(micius::d1::Part::PrepFFN);
            return micius::internal::PrepareF64GEMVInput(
                {reference_normed.get(), hidden_dim},
                {reference_gemv_input.get(), std::max(hidden_dim, config.intermediate_dim)},
                ffn_prepared);
        }();
        if (preparation_status != Q8_32_GEMVStatus::OK)
            return preparation_status == Q8_32_GEMVStatus::NUMERIC_ERROR
                ? ReferenceStepResult::NUMERIC_ERROR : ReferenceStepResult::INVALID_MODEL_DATA;
#endif
        MICIUS_D1_TAG(Gate);
#if MICIUS_I4_MODE
        i4_op=micius::i4::Op::Gate;
#endif
#if defined(MICIUS_I3_INTERVENTION)
        i3.set(layer_index,micius::i3::Op::Gate);
#endif
#if defined(MICIUS_I2_DIAGNOSTICS)
        if (!i2.boundary(static_cast<int>(layer_index),"attention_residual",reference_residual.get(),hidden_dim))
            return ReferenceStepResult::INVALID_MODEL_DATA;
        i2.set(static_cast<int>(layer_index), "gate");
#endif
#if defined(MICIUS_I25_PANEL_GEMV)
        panel=i25_panels ? &i25_panels->matrices[size_t(layer_index)*5U+2U] : nullptr;
#endif
        math_status = gemv(
            layer.gate_weight,
            layer.gate_scales,
            layer.gate_compact_scales,
            reference_normed.get(),
            reference_gate.get(),
            hidden_dim,
            config.intermediate_dim,
            reference_scales_prevalidated
#if defined(MICIUS_REFERENCE_GEMV_PREPARED_F64) || MICIUS_I1_MODE
            , &ffn_prepared
#endif
            );
        if (math_status != ReferenceMathStatus::OK) return ToStepResult(math_status);
        MICIUS_D1_TAG(Up);
#if MICIUS_I4_MODE
        i4_op=micius::i4::Op::Up;
#endif
#if defined(MICIUS_I3_INTERVENTION)
        i3.set(layer_index,micius::i3::Op::Up);
#endif
#if defined(MICIUS_I2_DIAGNOSTICS)
        i2.set(static_cast<int>(layer_index), "up");
#endif
#if defined(MICIUS_I25_PANEL_GEMV)
        panel=i25_panels ? &i25_panels->matrices[size_t(layer_index)*5U+3U] : nullptr;
#endif
        math_status = gemv(
            layer.up_weight,
            layer.up_scales,
            layer.up_compact_scales,
            reference_normed.get(),
            reference_up.get(),
            hidden_dim,
            config.intermediate_dim,
            reference_scales_prevalidated
#if defined(MICIUS_REFERENCE_GEMV_PREPARED_F64) || MICIUS_I1_MODE
            , &ffn_prepared
#endif
            );
        if (math_status != ReferenceMathStatus::OK) return ToStepResult(math_status);

#if defined(MICIUS_I3_INTERVENTION)
        if (!i3.end_ffn(reference_normed.get(),hidden_dim)) return ReferenceStepResult::INVALID_MODEL_DATA;
#endif
        } // both FFN consumers finished; the prepared view expires here

        for (uint32_t i = 0; i < config.intermediate_dim; ++i) {
            const double gate_value = reference_gate[i];
            const double activated =
                gate_value / (1.0 + std::exp(-gate_value)) * reference_up[i];
            if (!std::isfinite(activated) ||
                activated > std::numeric_limits<float>::max() ||
                activated < -std::numeric_limits<float>::max()) {
                return ReferenceStepResult::NUMERIC_ERROR;
            }
            reference_gate[i] = static_cast<float>(activated);
        }

        MICIUS_D1_TAG(Down);
#if MICIUS_I4_MODE
        i4_op=micius::i4::Op::Down;
#endif
#if defined(MICIUS_I3_INTERVENTION)
        i3.set(layer_index,micius::i3::Op::Down);
#endif
#if defined(MICIUS_I2_DIAGNOSTICS)
        i2.set(static_cast<int>(layer_index), "down");
#endif
#if defined(MICIUS_I25_PANEL_GEMV)
        panel=i25_panels ? &i25_panels->matrices[size_t(layer_index)*5U+4U] : nullptr;
#endif
        math_status = gemv(
            layer.down_weight,
            layer.down_scales,
            layer.down_compact_scales,
            reference_gate.get(),
            reference_projection.get(),
            config.intermediate_dim,
            hidden_dim,
            reference_scales_prevalidated);
        if (math_status != ReferenceMathStatus::OK) return ToStepResult(math_status);
        for (uint32_t i = 0; i < hidden_dim; ++i) {
            const double value_sum =
                static_cast<double>(reference_residual[i]) + reference_projection[i];
            if (!std::isfinite(value_sum) ||
                value_sum > std::numeric_limits<float>::max() ||
                value_sum < -std::numeric_limits<float>::max()) {
                return ReferenceStepResult::NUMERIC_ERROR;
            }
            reference_residual[i] = static_cast<float>(value_sum);
        }
#if defined(MICIUS_I2_DIAGNOSTICS)
        if (!i2.boundary(static_cast<int>(layer_index),"layer_output",reference_residual.get(),hidden_dim))
            return ReferenceStepResult::INVALID_MODEL_DATA;
#endif
#if defined(MICIUS_I3_INTERVENTION)
        if (!i3.boundary(static_cast<int>(layer_index),"layer_output",reference_residual.get(),hidden_dim))
            return ReferenceStepResult::INVALID_MODEL_DATA;
#endif
    }

    ReferenceMathStatus math_status = ReferenceRMSNorm(
        reference_normed.get(),
        reference_residual.get(),
        norm_weight_final,
        hidden_dim,
        config.rms_norm_epsilon);
    if (math_status != ReferenceMathStatus::OK) return ToStepResult(math_status);
    MICIUS_D1_TAG(LM);
#if MICIUS_I4_MODE
    i4_layer=config.num_layers;i4_op=micius::i4::Op::LM;
#endif
#if defined(MICIUS_I3_INTERVENTION)
    i3.set(config.num_layers,micius::i3::Op::LM);
#endif
#if defined(MICIUS_I2_DIAGNOSTICS)
    i2.set(static_cast<int>(config.num_layers), "lm");
#endif
#if defined(MICIUS_I25_PANEL_GEMV)
    panel=i25_panels ? &i25_panels->matrices.back() : nullptr;
#endif
    math_status = gemv(
        engine_quant_weights,
        engine_scales,
        engine_compact_scales,
        reference_normed.get(),
        reference_logits.get(),
        hidden_dim,
        config.vocab_size,
        reference_scales_prevalidated);
    if (math_status != ReferenceMathStatus::OK) return ToStepResult(math_status);

    uint32_t best_token = 0;
    float best_logit = -std::numeric_limits<float>::infinity();
    for (uint32_t candidate = 0; candidate < config.vocab_size; ++candidate) {
        const float logit = reference_logits[candidate];
        if (!std::isfinite(logit)) return ReferenceStepResult::NUMERIC_ERROR;
        if (logit > best_logit) {
            best_logit = logit;
            best_token = candidate;
        }
    }

#if defined(MICIUS_I2_DIAGNOSTICS)
    if (!i2.complete()) return ReferenceStepResult::INVALID_MODEL_DATA;
#endif
#if defined(MICIUS_I3_INTERVENTION)
    if (!i3.complete()) return ReferenceStepResult::INVALID_MODEL_DATA;
#endif
    ++reference_seq_len;
    reference_has_logits = true;
    if (out_token) *out_token = best_token;
    return ReferenceStepResult::SUCCESS;
}

ReferenceStepResult AIEngineFront::PrefillReferenceToken(uint32_t token) noexcept
{
    return RunReferenceToken(token, nullptr);
}

ReferenceStepResult AIEngineFront::DecodeReferenceToken(
    uint32_t token,
    uint32_t* out_token) noexcept
{
    if (!out_token) return ReferenceStepResult::INVALID_MODEL_DATA;
    const ReferenceStepResult result = RunReferenceToken(token, out_token);
    if (result == ReferenceStepResult::SUCCESS) {
        total_tokens_generated.fetch_add(1, std::memory_order_relaxed);
    }
    return result;
}

bool AIEngineFront::CopyLastReferenceLogits(float* out_logits, size_t count) const noexcept
{
    if (!reference_has_logits || !reference_logits || !out_logits || count < config.vocab_size) {
        return false;
    }
    std::copy_n(reference_logits.get(), config.vocab_size, out_logits);
    return true;
}

bool AIEngineFront::CopyReferenceKVSlot(
    uint32_t layer,
    uint32_t position,
    float* out_key,
    float* out_value,
    size_t element_count) const noexcept
{
    if (!reference_k_cache || !reference_v_cache || !out_key || !out_value ||
        layer >= config.num_layers || position >= reference_seq_len ||
        element_count < reference_kv_dim) {
        return false;
    }
    const size_t offset =
        (static_cast<size_t>(layer) * config.max_seq_len + position) * reference_kv_dim;
    std::copy_n(reference_k_cache.get() + offset, reference_kv_dim, out_key);
    std::copy_n(reference_v_cache.get() + offset, reference_kv_dim, out_value);
    return true;
}

void AIEngineFront::InitPipeline(float* root_hidden_state, uint32_t initial_token)
{
    //使用新的拓扑方案绑核
    HardwareTopology::SetThreadAffinity(topology_plan.hq_core_id);
    
    pipeline_initialized = true;
    awaiting_fallback_token = false;
    current_cpu_tree = 0;
    last_accepted_token = initial_token;
    expected_input_token = initial_token;
    token_queue_head = 0;
    token_queue_tail = 0;

    history_len = 0; //管线重建时复位长程记忆
    
    pipeline_roots[0] = kv_cache->AllocateSlot();
    pipeline_base_slots[0] = kv_cache->AllocateSlot();

    if (pipeline_roots[0] == 0xFFFFFFFF || pipeline_base_slots[0] == 0xFFFFFFFF) {
        pipeline_initialized = false;
        return;
    }

    kv_cache->WriteTreeSlotFP32(pipeline_roots[0], root_hidden_state);
    shared_proposal[current_cpu_tree]->Reset();
    
    //发射初始根任务
    FrontierTask seed;
    //接入调度器的Epoch纪元
    seed.gen_id = (uint32_t)scheduler.GetBarrier().GetCurrentEpoch();
    seed.parent_slot = pipeline_roots[0];
    seed.alloc_slot = pipeline_base_slots[0];
    seed.input_token = last_accepted_token;
    seed.parent_tree_idx = -1;
    seed.branch_id = 0;
    seed.fusion_depth = 4;
    seed.tree_idx = static_cast<uint8_t>(current_cpu_tree);
    seed.is_golden = true;
    
    global_seed_task = seed;
    seed_claimed.store(0, std::memory_order_release);
}

PumpResult AIEngineFront::PumpPipeline(uint32_t external_input_token, const float* external_hidden_state, uint32_t* out_token)
{
    if (!pipeline_initialized) return PumpResult::OOM_ERROR;

    //外部输入校验与管线硬重置
    if (external_input_token != expected_input_token || awaiting_fallback_token) {
        
        token_queue_head = 0;
        token_queue_tail = 0;
        awaiting_fallback_token = false;
        last_accepted_token = external_input_token; 

        //推进全局Epoch不等待Worker，拿到旧Epoch的任务会在下一次循环丢弃
        scheduler.GetBarrier().AdvanceEpoch();

        pipeline_base_slots[current_cpu_tree] = kv_cache->AllocateSlot();
        if (pipeline_base_slots[current_cpu_tree] == 0xFFFFFFFF) return PumpResult::OOM_ERROR;

        if (external_hidden_state) {
            kv_cache->WriteTreeSlotFP32(pipeline_roots[current_cpu_tree], external_hidden_state);
        }

        shared_proposal[current_cpu_tree]->Reset();

        FrontierTask next_seed;
        next_seed.gen_id = (uint32_t)scheduler.GetBarrier().GetCurrentEpoch(); //绑定新纪元
        next_seed.parent_slot = pipeline_roots[current_cpu_tree];
        next_seed.alloc_slot = pipeline_base_slots[current_cpu_tree];
        next_seed.input_token = last_accepted_token;
        next_seed.parent_tree_idx = -1;
        next_seed.branch_id = 0;
        next_seed.fusion_depth = 4;
        next_seed.tree_idx = static_cast<uint8_t>(current_cpu_tree);
        next_seed.is_golden = true;

        global_seed_task = next_seed;
        seed_claimed.store(0, std::memory_order_release);
    }

    //缓冲输出
    if (token_queue_head < token_queue_tail) {
        *out_token = accepted_tokens_queue[token_queue_head++];
        expected_input_token = *out_token; //同步握手信号
        total_tokens_generated.fetch_add(1, std::memory_order_relaxed);
        return PumpResult::YIELD_TOKEN;
    }

    //草稿推测树验证等待
    int spin_1 = 0;
    while(shared_proposal[current_cpu_tree]->commit_counter.load(std::memory_order_acquire) < DraftProposal::MAX_K_DRAFTS) {
        if (shared_proposal[current_cpu_tree]->state_flags.load(std::memory_order_acquire) == 3) break; // OOM退出
        int pause_count = 1 << std::min(spin_1 / 10, 8); 
        for(int p = 0; p < pause_count; ++p) _mm_pause();
        spin_1++;
    }

    //封存草稿树，准备移交验证
    shared_proposal[current_cpu_tree]->state_flags.store(2, std::memory_order_release);
    current_gpu_tree = current_cpu_tree;
    uint32_t next_tree = 1 - current_cpu_tree; //翻转Ping-Pong Buffer
    
    shared_proposal[next_tree]->Reset();
    
    //这个时候启动推测流水线
    uint32_t total_drafts = shared_proposal[current_gpu_tree]->commit_counter.load(std::memory_order_acquire);
    int32_t golden_idx = shared_proposal[current_gpu_tree]->golden_leaf_idx.load(std::memory_order_relaxed);
    uint32_t next_token = 0;

    if (total_drafts == 0) {
        pipeline_roots[next_tree] = pipeline_base_slots[current_gpu_tree];
        next_token = last_accepted_token;
    } else {
        if (golden_idx < 0 || golden_idx >= (int32_t)total_drafts) {
            golden_idx = total_drafts - 1; 
        }
        pipeline_roots[next_tree] = shared_proposal[current_gpu_tree]->GetSlot(golden_idx);
        next_token = shared_proposal[current_gpu_tree]->GetToken(golden_idx);
    }

    pipeline_base_slots[next_tree] = kv_cache->AllocateSlot();
    if (pipeline_base_slots[next_tree] == 0xFFFFFFFF) return PumpResult::OOM_ERROR;
    
    //为推理过程的next_tree推进新的Epoch
    scheduler.GetBarrier().AdvanceEpoch();

    FrontierTask next_seed;
    next_seed.gen_id = (uint32_t)scheduler.GetBarrier().GetCurrentEpoch();
    next_seed.parent_slot = pipeline_roots[next_tree];
    next_seed.alloc_slot = pipeline_base_slots[next_tree];
    next_seed.input_token = next_token;
    next_seed.parent_tree_idx = -1;
    next_seed.branch_id = 0;
    next_seed.fusion_depth = 4;
    next_seed.tree_idx = static_cast<uint8_t>(next_tree);
    next_seed.is_golden = true;
    
    global_seed_task = next_seed;
    seed_claimed.store(0, std::memory_order_release);

    //异步提交至验证
    if (verifier_callback) {
        verifier_callback(this->wrapper_context, (void*)shared_proposal[current_gpu_tree], verifier_user_data);
    } else {
        // Dummy Fallback (For CPU-only testing)
        shared_proposal[current_gpu_tree]->accepted_tree_node_idx = (total_drafts > 0) ? golden_idx : -1;
        shared_proposal[current_gpu_tree]->state_flags.store(0, std::memory_order_release);
    }

    //等待外壳模型验证完成
    int spin_2 = 0;
    while(shared_proposal[current_gpu_tree]->state_flags.load(std::memory_order_acquire) != 0) {
        int pause_count = 1 << std::min(spin_2 / 5, 8); 
        for(int p = 0; p < pause_count; ++p) _mm_pause();
        spin_2++;
    }

    //审计结果解析与状态对齐
    int32_t accepted_idx = shared_proposal[current_gpu_tree]->accepted_tree_node_idx;
    uint32_t num_gpu_drafts = shared_proposal[current_gpu_tree]->commit_counter.load(std::memory_order_relaxed);
    
    token_queue_head = 0;
    token_queue_tail = 0;
    uint32_t path_buffer[DraftProposal::MAX_K_DRAFTS];
    uint32_t slot_buffer[DraftProposal::MAX_K_DRAFTS];
    uint32_t path_len = 0;

    //分支A
    //完整命中审计结果
    if (accepted_idx == golden_idx && accepted_idx >= 0) {
        int32_t curr = accepted_idx;
        while(curr != -1) {
            path_buffer[path_len] = shared_proposal[current_gpu_tree]->GetToken(curr);
            slot_buffer[path_len] = shared_proposal[current_gpu_tree]->GetSlot(curr);
            path_len++;
            curr = shared_proposal[current_gpu_tree]->GetParent(curr);
        }

        if (history_len < config.max_seq_len) accepted_slots_history[history_len++] = pipeline_base_slots[current_gpu_tree];

        for (int i = path_len - 1; i >= 0; --i) {
            accepted_tokens_queue[token_queue_tail++] = path_buffer[i];
            if (history_len < config.max_seq_len) accepted_slots_history[history_len++] = slot_buffer[i];
        }
        
        //释放未接受的冗余槽位
        uint32_t rej_slots[DraftProposal::MAX_K_DRAFTS + 2];
        uint32_t rc = 0;
        rej_slots[rc++] = pipeline_roots[current_gpu_tree];
        for(uint32_t i = 0; i < num_gpu_drafts; ++i) {
            if (i != (uint32_t)accepted_idx) rej_slots[rc++] = shared_proposal[current_gpu_tree]->GetSlot(i);
        }
        if (rc > 0) kv_cache->BulkFreeSlots(rej_slots, rc);
        
        current_cpu_tree = next_tree;

    } 
    //分支B
    //完整或部分未命中审计结果
    else {
        //之前需要阻塞直到所有Worker清理窃取的错误任务
        //如今只推进一次Epoch，Worker的废任务会在几微秒内彻底失去合法性
        scheduler.GetBarrier().AdvanceEpoch();
        
        //清理丢弃的next_tree占用的插槽
        uint32_t abort_slots[DraftProposal::MAX_K_DRAFTS + 1];
        uint32_t ac = 0;
        abort_slots[ac++] = pipeline_base_slots[next_tree];
        
        uint32_t num_next = shared_proposal[next_tree]->commit_counter.load(std::memory_order_relaxed);
        for(uint32_t i = 0; i < num_next; ++i) abort_slots[ac++] = shared_proposal[next_tree]->GetSlot(i);
        kv_cache->BulkFreeSlots(abort_slots, ac);

        uint32_t rej_slots[DraftProposal::MAX_K_DRAFTS + 2];
        uint32_t rc = 0;
        
        //部分命中的情况
        if (accepted_idx >= 0) {
            int32_t curr = accepted_idx;
            while(curr != -1) {
                path_buffer[path_len] = shared_proposal[current_gpu_tree]->GetToken(curr);
                slot_buffer[path_len] = shared_proposal[current_gpu_tree]->GetSlot(curr);
                path_len++;
                curr = shared_proposal[current_gpu_tree]->GetParent(curr);
            }
            
            if (history_len < config.max_seq_len) accepted_slots_history[history_len++] = pipeline_base_slots[current_gpu_tree];

            for (int i = path_len - 1; i >= 0; --i) {
                accepted_tokens_queue[token_queue_tail++] = path_buffer[i];
                if (history_len < config.max_seq_len) accepted_slots_history[history_len++] = slot_buffer[i];
            }
            
            //对齐新的管线根节点
            pipeline_roots[next_tree] = shared_proposal[current_gpu_tree]->GetSlot(accepted_idx);
            last_accepted_token = shared_proposal[current_gpu_tree]->GetToken(accepted_idx);

            rej_slots[rc++] = pipeline_roots[current_gpu_tree];
            for(uint32_t i = 0; i < num_gpu_drafts; ++i) {
                if (i != (uint32_t)accepted_idx) rej_slots[rc++] = shared_proposal[current_gpu_tree]->GetSlot(i);
            }
        } else {
            //完全未命中的根节点保持不变
            pipeline_roots[next_tree] = pipeline_roots[current_gpu_tree]; 
            rej_slots[rc++] = pipeline_base_slots[current_gpu_tree];
            for(uint32_t i = 0; i < num_gpu_drafts; ++i) rej_slots[rc++] = shared_proposal[current_gpu_tree]->GetSlot(i);
        }

        if (rc > 0) kv_cache->BulkFreeSlots(rej_slots, rc);
        shared_proposal[next_tree]->Reset();

        //根据命中结果重新发射种子
        if (accepted_idx >= 0) {
            pipeline_base_slots[next_tree] = kv_cache->AllocateSlot();
            if (pipeline_base_slots[next_tree] == 0xFFFFFFFF) return PumpResult::OOM_ERROR;
            
            //补发Seed绑定当前新Epoch
            next_seed.gen_id = (uint32_t)scheduler.GetBarrier().GetCurrentEpoch();
            next_seed.parent_slot = pipeline_roots[next_tree];
            next_seed.alloc_slot = pipeline_base_slots[next_tree];
            next_seed.input_token = last_accepted_token;
            next_seed.parent_tree_idx = -1;
            next_seed.tree_idx = static_cast<uint8_t>(next_tree);
            next_seed.is_golden = true;
            
            global_seed_task = next_seed;
            seed_claimed.store(0, std::memory_order_release);
            current_cpu_tree = next_tree;
        } else {
            //如果连根Token都未命中，挂起管线等待下次传入Fallback Token校正
            awaiting_fallback_token = true;
            current_cpu_tree = next_tree;
            tree_reject_count.fetch_add(1, std::memory_order_relaxed);
            return PumpResult::TREE_REJECTED;
        }
    }

    //尾部清算，尝试立刻弹出结果
    if (token_queue_head < token_queue_tail) {
        *out_token = accepted_tokens_queue[token_queue_head++];
        expected_input_token = *out_token; //更新握手预期
        total_tokens_generated.fetch_add(1, std::memory_order_relaxed);
        return PumpResult::YIELD_TOKEN;
    }

    tree_reject_count.fetch_add(1, std::memory_order_relaxed);
    return PumpResult::TREE_REJECTED;
}

micius::i25::Stats AIEngineFront::I25Stats() const noexcept {
    if(!i25_panels)return {};
    auto s=i25_panels->stats;s.legacy_scale_buffers=owned_scale_buffers.size();
    s.legacy_consumer_pointers=bool(engine_quant_weights)+bool(engine_scales)+bool(engine_compact_scales);
    for(const auto& l:model_layers)s.legacy_consumer_pointers+=
        bool(l.qkv_packed_weights)+bool(l.qkv_packed_scales)+bool(l.qkv_packed_compact_scales)+
        bool(l.o_weight)+bool(l.o_scales)+bool(l.o_compact_scales)+
        bool(l.gate_weight)+bool(l.gate_scales)+bool(l.gate_compact_scales)+
        bool(l.up_weight)+bool(l.up_scales)+bool(l.up_compact_scales)+
        bool(l.down_weight)+bool(l.down_scales)+bool(l.down_compact_scales);
    return s;
}
std::span<const micius::i25::RetiredWeight> AIEngineFront::I25Retired() const noexcept {
    return i25_panels ? std::span<const micius::i25::RetiredWeight>(i25_panels->retired) : std::span<const micius::i25::RetiredWeight>{};
}

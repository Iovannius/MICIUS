#pragma once

#include "AIPrimitives.h"
#include "AIHetMemPool.h"
#include "AIDataFront.h"
#include "AIComputKrnl.h"
#include "PreparedGEMVWeights.h"
#include "AI/I25Binding.h"
#include "AI/FileTensorBinding.h"
namespace micius::i25 { struct Panels; }

#include "SchedulerCore.h"
#include "HardwareTopology.h"

#include "../micius_runtime.h"

#include <cstdint>
#include <vector>
#include <thread>
#include <atomic>
#include <cstring>
#include <memory>

struct alignas(64) WorkerContext {
    
    AlignedArray<int8_t> workspace_i8;
    AlignedArray<float> logits;

    //TreeAttention_GQA的计数器，大小为max_seq_len
    AlignedArray<float> attn_score_workspace;

    //QKV投影的工作数组，容量须容纳Q + K + V的维度总和
    AlignedArray<float> qkv_workspace;

    //FFN工作数组 (SwiGLU)
    AlignedArray<float> ffn_workspace;
    
    //树状注意力机制的指针追踪数组
    AlignedArray<const float*> k_ptrs;
    AlignedArray<const float*> v_ptrs;
    
    uint32_t tlab_slots[16];
    uint32_t tlab_count = 0;
    uint32_t prng_state;

    std::atomic<uint32_t> tasks_processed{0};
    std::atomic<uint32_t> tasks_stolen{0};

    //维度等项分配的构造函数
    WorkerContext(uint32_t hidden_dim, uint32_t vocab_size, uint32_t max_seq_len, uint32_t prng_seed) 
        : prng_state(prng_seed)
    {
        if (prng_state == 0) prng_state = 1;

        workspace_i8 = MakeAlignedArray<int8_t>(hidden_dim * 4);
        ffn_workspace = MakeAlignedArray<float>(hidden_dim * 8);
        logits = MakeAlignedArray<float>(vocab_size);
        attn_score_workspace = MakeAlignedArray<float>(max_seq_len);
        qkv_workspace = MakeAlignedArray<float>(hidden_dim * 4);
        k_ptrs = MakeAlignedArray<const float*>(max_seq_len);
        v_ptrs = MakeAlignedArray<const float*>(max_seq_len);
        
        std::memset(workspace_i8.get(), 0, hidden_dim * 4 * sizeof(int8_t));
        std::memset(logits.get(), 0, vocab_size * sizeof(float));
    }
};

//L1优化的Chase-Lev无锁双端队列
//必须2的幂次容量
template<typename T, size_t Cap> class ChaseLevDeque
{
    static_assert(Cap > 0 && (Cap & (Cap - 1)) == 0, "deque capacity must be a power of two");
    static_assert(Cap <= static_cast<size_t>(INT64_MAX), "deque capacity exceeds index range");
    alignas(64) std::atomic<int64_t> top{0};
    alignas(64) std::atomic<int64_t> bottom{0};
    alignas(64) T data[Cap];
public:
    inline bool push_bottom(const T& item)
    {
        int64_t b=bottom.load(std::memory_order_relaxed), t=top.load(std::memory_order_acquire);
        if (b - t >= static_cast<int64_t>(Cap)) return false;
        data[b&(Cap-1)]=item;
        std::atomic_thread_fence(std::memory_order_release);
        bottom.store(b+1, std::memory_order_relaxed);
        return true;
    }
    inline bool pop_bottom(T& item)
    {
        int64_t b=bottom.load(std::memory_order_relaxed)-1;
        bottom.store(b, std::memory_order_relaxed);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        int64_t t=top.load(std::memory_order_relaxed);
        bool ok=false;
        if(t<=b) {
            item=data[b&(Cap-1)];
            ok=true;
            if(t==b) {
                if(!top.compare_exchange_strong(t, t+1, std::memory_order_seq_cst, std::memory_order_relaxed)) ok=false;
                bottom.store(b+1, std::memory_order_relaxed);
            }
        } else {
            bottom.store(b+1, std::memory_order_relaxed);
        }
        
        return ok;
    }
    inline bool steal_top(T& item)
    {
        int64_t t=top.load(std::memory_order_acquire);
        std::atomic_thread_fence(std::memory_order_seq_cst);
        int64_t b=bottom.load(std::memory_order_acquire);
        if(t<b) {
            item=data[t&(Cap-1)];
            if(top.compare_exchange_strong(t, t+1, std::memory_order_seq_cst, std::memory_order_relaxed)) return true;
        }
        
            return false;
    }
    inline void clear()
    {
        top.store(0, std::memory_order_relaxed);
        bottom.store(0, std::memory_order_relaxed);
    }
};

//L3优化
//引入微内核融合深度fusion_depth
//单次抢占后在寄存器内连续计算Token
//双提案区索引与真路径判定
struct FrontierTask {
    uint32_t gen_id, parent_slot, alloc_slot, input_token;
    int32_t parent_tree_idx;
    uint32_t branch_id;
    uint32_t fusion_depth;
    uint8_t tree_idx;
    bool is_golden;
};

enum class PumpResult
{
    YIELD_TOKEN,
    TREE_REJECTED,
    OOM_ERROR
};

enum class ReferenceStepResult
{
    SUCCESS,
    MODEL_NOT_BOUND,
    SEQUENCE_LIMIT_REACHED,
    INVALID_MODEL_DATA,
    NUMERIC_ERROR
};

struct MiciusLayerWeights {
    bool split_half_rope = false;
    const float* q_bias = nullptr;
    const float* k_bias = nullptr;
    const float* v_bias = nullptr;
    int8_t* qkv_packed_weights = nullptr;
    QuantDescriptor* qkv_packed_scales = nullptr;
    float* qkv_packed_compact_scales = nullptr;

    const int8_t* o_weight = nullptr;
    const QuantDescriptor* o_scales = nullptr;
    const float* o_compact_scales = nullptr;
    const int8_t* gate_weight = nullptr;
    const QuantDescriptor* gate_scales = nullptr;
    const float* gate_compact_scales = nullptr;
    const int8_t* up_weight = nullptr;
    const QuantDescriptor* up_scales = nullptr;
    const float* up_compact_scales = nullptr;
    const int8_t* down_weight = nullptr;
    const QuantDescriptor* down_scales = nullptr;
    const float* down_compact_scales = nullptr;

    const float* rms_attn_weight = nullptr;
    const float* rms_ffn_weight = nullptr;
};

class AIEngineFront
{

private:

    struct OwnedScaleBuffer {
        std::unique_ptr<QuantDescriptor[]> data;
        std::unique_ptr<float[]> compact_data;
        size_t count = 0;
    };

    std::unique_ptr<micius::i25::Panels> i25_panels;
    MiciusConfig config;
    std::vector<MiciusLayerWeights> model_layers;
    std::vector<OwnedScaleBuffer> owned_scale_buffers;
    const float* norm_weight_final = nullptr;
    const float* token_embd_weight = nullptr;
    bool model_bound = false;
    bool speculative_enabled = false;
    bool reference_scales_prevalidated = false;

    int8_t* safe_dummy_weights = nullptr;
    QuantDescriptor* safe_dummy_scales = nullptr;
    size_t safe_dummy_scale_count = 0;
    
    float* safe_dummy_fp32 = nullptr; //用于FP32浮点张量（如RMSNorm）的安全指针

    void* wrapper_context = nullptr; //用于保存外部的MiciusContext指针

    /*---v0.5 Runtime隔离边界---*/
    const int8_t* engine_quant_weights = nullptr;
    const QuantDescriptor* engine_scales = nullptr;
    const float* engine_compact_scales = nullptr;

    /*---v0.5 状态机持久上下文---*/
    bool pipeline_initialized = false;
    bool awaiting_fallback_token = false;
    uint32_t current_cpu_tree = 0;
    uint32_t current_gpu_tree = 0;
    uint32_t pipeline_roots[2]{0xFFFFFFFF, 0xFFFFFFFF};
    uint32_t pipeline_base_slots[2]{0xFFFFFFFF, 0xFFFFFFFF};
    uint32_t last_accepted_token = 0;

    //用于与外壳的通讯，判断外部是否在注入Prompt
    uint32_t expected_input_token = 0;

    //上下文记忆槽
    uint32_t* accepted_slots_history = nullptr;
    uint32_t history_len = 0;

    /*---v0.5 Token泵队列---*/
    //用于缓存单次预言验证后推测树被连续接受的多个Token
    uint32_t accepted_tokens_queue[DraftProposal::MAX_K_DRAFTS + 2];
    uint32_t token_queue_head = 0;
    uint32_t token_queue_tail = 0;

    //外部预言回调注入
    void* verifier_user_data = nullptr;
    void (*verifier_callback)(void* ctx, void* tree_handle, void* user_data) = nullptr;

    HeteroMemPool* mem_pool = nullptr;
    KVCacheRingBuffer* kv_cache = nullptr;
    AiDataFront* data_front = nullptr;
    DraftProposal* shared_proposal[2]{nullptr, nullptr}; //双提案区隔离
    uint32_t hidden_dim = 0;
    
    RoPEFreqCache rope_cache{}; //RoPE频率表预计算缓存
    //供所有核只读

    // Single-thread authoritative path. K/V are indexed as
    // [layer][position][kv_head][head_dim]; residual and scratch never alias KV.
    AlignedArray<float> reference_k_cache;
    AlignedArray<float> reference_v_cache;
    AlignedArray<float> reference_residual;
    AlignedArray<float> reference_normed;
    AlignedArray<float> reference_qkv;
    AlignedArray<float> reference_attention;
    AlignedArray<float> reference_projection;
    AlignedArray<float> reference_gate;
    AlignedArray<float> reference_up;
    AlignedArray<float> reference_logits;
    AlignedArray<float> reference_scores;
    // One per-context activation snapshot, allocated only by the prepared-F64
    // experiment. Member layout stays the same across internal feature flags.
    AlignedArray<double> reference_gemv_input;
    // I1 activation snapshot only; no expanded weight shadow.
    AlignedArray<int8_t> i1_input;
    AlignedArray<double> i1_scales;
    uint32_t reference_seq_len = 0;
    uint32_t reference_kv_dim = 0;
    uint32_t reference_qkv_dim = 0;
    size_t reference_kv_elements = 0;
    bool reference_has_logits = false;

    // Only the shared checked tensor binder can enter this boundary. It takes
    // ownership of the private scale representation created after full public
    // descriptor validation, so no public C++ caller can assert the invariant.
    bool BindValidatedTensors(
        const MiciusTensorEntry* entries,
        uint32_t count,
        std::vector<OwnedScaleBuffer>&& scale_buffers, bool split_half_rope);

    friend bool micius::internal::BindTypedTensors(
        MiciusContext* context,
        const MiciusTensorEntryV1* entries,
        uint32_t count,
        size_t entry_size,
        uint32_t abi_version, micius::internal::ProjectionSemantics);

    ReferenceStepResult RunReferenceToken(uint32_t token, uint32_t* out_token) noexcept;
    
    /*---v0.5.2 管线核心---*/
    SpeculativeScheduler scheduler;
    ThreadTopologyPlan   topology_plan; // 替代原来散落的 hq_core_id 等
    
    std::vector<int> worker_cores;
    std::vector<std::thread> worker_threads;
    std::vector<WorkerContext*> worker_ctxs;

    alignas(64) std::atomic<bool> global_terminate{false};

    alignas(64) std::atomic<int> seed_claimed{1};
    alignas(64) FrontierTask global_seed_task{};
    
    std::vector<ChaseLevDeque<FrontierTask, 1024>> local_queues;
    std::atomic<uint8_t> branch_tombstones[DraftProposal::MAX_K_DRAFTS]{};
    
    void WorkerRoutine(int worker_idx);

    inline bool ComputeStateTransition(WorkerContext* ctx, const FrontierTask& task, const float* parent_state, float* new_state);
    
public:

    micius::i25::Stats I25Stats() const noexcept;
    std::span<const micius::i25::RetiredWeight> I25Retired() const noexcept;
    int recommended_shadow_core = -1; //暴露的影子线程绑核建议

    //宏观的负载探针
    alignas(64) std::atomic<uint32_t> total_tokens_generated{0};
    alignas(64) std::atomic<uint32_t> tree_reject_count{0};

    AIEngineFront(const MiciusConfig& cfg);
    ~AIEngineFront();

    inline int GetRecommendedShadowCore() const
    {
        return topology_plan.recommended_shadow_core;
    }

    //注册外部预言回调
    void RegisterVerifier(void (*cb)(void*, void*, void*), void* wrapper_ctx, void* user_data);

    // Reference mode has no live workers, so its session state can be reset in place.
    // Speculative reset remains rejected until the worker quiescence protocol is rebuilt.
    bool ResetState() noexcept;

    size_t ReferenceStateBytes() const;
    std::vector<uint8_t> ExportReferenceState() const;
    void ImportReferenceState(std::span<const uint8_t> bytes, uint32_t expected_length);

    ReferenceStepResult PrefillReferenceToken(uint32_t token) noexcept;
    ReferenceStepResult DecodeReferenceToken(uint32_t token, uint32_t* out_token) noexcept;
    bool CopyLastReferenceLogits(float* out_logits, size_t count) const noexcept;
    bool CopyReferenceKVSlot(
        uint32_t layer,
        uint32_t position,
        float* out_key,
        float* out_value,
        size_t element_count) const noexcept;
    inline uint32_t GetReferenceSequenceLength() const noexcept { return reference_seq_len; }
    inline uint32_t GetUsedKVSlots() const noexcept
    {
        return speculative_enabled && kv_cache
            ? kv_cache->GetUsedSlotsCount()
            : reference_seq_len;
    }

    //废弃RunTreeInferencePipeline，替换为以下方法
    void InitPipeline(float* root_hidden_state, uint32_t initial_token);
    
    PumpResult PumpPipeline(uint32_t external_input_token, const float* external_hidden_state, uint32_t* out_token);
    
    AIEngineFront(const AIEngineFront&) = delete;
    AIEngineFront& operator=(const AIEngineFront&) = delete;

    inline DraftProposal* GetSharedProposal(int idx) const
    {
        return shared_proposal[idx];
    }

    //暴露内存池给预言线程执行后台GC
    inline KVCacheRingBuffer* GetKVCache() const
    {
        return kv_cache;
    }

    //解除Private访问限制Getters
    inline size_t GetWorkerCount() const { return worker_cores.size(); }
    inline WorkerContext* GetWorkerContext(size_t idx) const { return worker_ctxs[idx]; }
    inline int GetWorkerCoreId(size_t idx) const { return worker_cores[idx]; }
    inline bool IsModelBound() const { return model_bound; }
    inline bool IsSpeculativeEnabled() const { return speculative_enabled; }
    micius::internal::PreparedWeightStats GetPreparedWeightStats() const noexcept
    { return reference_weight_stats; }

private:
    // Appended to preserve every existing member offset in option-off builds.
    AlignedArray<double> reference_query_f64;

    // PW2: one immutable representation per selected matrix in this binding.
    // Each engine owns its own binding; no claim of cross-Context model sharing.
    std::vector<micius::internal::PreparedF32GEMVWeights> reference_prepared_weights;
    micius::internal::PreparedWeightStats reference_weight_stats;

};

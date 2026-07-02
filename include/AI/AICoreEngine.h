#pragma once

#include "AIPrimitives.h"
#include "AIHetMemPool.h"
#include "AIDataFront.h"

#include <cstdint>
#include <vector>
#include <thread>
#include <atomic>

struct alignas(64) WorkerContext {
    
    int8_t* workspace_i8;
    alignas(64) float* logits;

    uint32_t tlab_slots[16];
    uint32_t tlab_count;
    uint32_t prng_state;

    //微观的遥测探针
    uint32_t tasks_processed = 0;
    uint32_t tasks_stolen = 0;
};

//L1优化的Chase-Lev无锁双端队列
//必须2的幂次容量
template<typename T, size_t Cap> class ChaseLevDeque
{
    alignas(64) std::atomic<int64_t> top{0};
    alignas(64) std::atomic<int64_t> bottom{0};
    alignas(64) T data[Cap];
public:
    inline bool push_bottom(const T& item)
    {
        int64_t b=bottom.load(std::memory_order_relaxed), t=top.load(std::memory_order_acquire);
        if(b-t>=Cap) return false;
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

class AIEngineFront
{

private:

    void* wrapper_context; //用于保存外部的 MiciusContext 指针

    /*---v0.5 Runtime隔离边界---*/
    const int8_t* engine_quant_weights;
    const QuantDescriptor* engine_scales;

    /*---v0.5 状态机持久上下文---*/
    bool pipeline_initialized;
    bool awaiting_fallback_token;
    uint32_t current_cpu_tree;
    uint32_t current_gpu_tree;
    uint32_t pipeline_roots[2];
    uint32_t pipeline_base_slots[2];
    uint32_t last_accepted_token;

    /*---v0.5 Token泵队列---*/
    //用于缓存单次预言验证后推测树被连续接受的多个Token
    uint32_t accepted_tokens_queue[DraftProposal::MAX_K_DRAFTS + 2];
    uint32_t token_queue_head;
    uint32_t token_queue_tail;

    /*---外部预言回调注入---*/
    void* verifier_user_data;
    void (*verifier_callback)(void* ctx, void* tree_handle, void* user_data);

    HeteroMemPool* mem_pool;
    KVCacheRingBuffer* kv_cache;
    AiDataFront* data_front;
    DraftProposal* shared_proposal[2]; //双提案区隔离
    uint32_t hidden_dim;
    
    int hq_core_id;
    std::vector<int> worker_cores;
    std::vector<std::thread> worker_threads;
    std::vector<WorkerContext*> worker_ctxs;

    alignas(64) std::atomic<uint32_t> global_generation{0};
    alignas(64) std::atomic<bool> global_terminate{false};
    
    //取消worker_gen数组，使用单一的高速全局屏障以降低O(N)级总线阻塞
    alignas(64) std::atomic<uint32_t> workers_finished{0};

    alignas(64) std::atomic<int> seed_claimed{1};
    alignas(64) FrontierTask global_seed_task;
    
    std::vector<ChaseLevDeque<FrontierTask, 1024>> local_queues;
    std::atomic<uint8_t> branch_tombstones[DraftProposal::MAX_K_DRAFTS];
    
    std::vector<int> ProbePhysicalCores(); void WorkerRoutine(int worker_idx);

    inline void ComputeStateTransition(uint32_t token, const float* parent_state, float* new_state, uint32_t hidden_dim);
    
public:

    int recommended_shadow_core; //暴露的影子线程绑核建议

    //宏观的负载探针
    alignas(64) std::atomic<uint32_t> total_tokens_generated{0};
    alignas(64) std::atomic<uint32_t> tree_reject_count{0};

    AIEngineFront(size_t pool_size, uint32_t max_seq_len, uint32_t hidden_dim);
    ~AIEngineFront();

    //绑定外部模型权重
    void LoadWeights(const int8_t* q_weights, const QuantDescriptor* scales);
    
    //注册外部预言回调
    void RegisterVerifier(void (*cb)(void*, void*, void*), void* wrapper_ctx, void* user_data);

    //废弃RunTreeInferencePipeline，替换为以下方法
    void InitPipeline(float* root_hidden_state, uint32_t initial_token);
    
    PumpResult PumpSingleToken(uint32_t external_input_token, const float* external_hidden_state, uint32_t* out_token); //核心状态机驱动器
    //每次调用泵出一个Token，返回true表示出现新Token，返回false表示内部异常或OOM
    
    AIEngineFront(const AIEngineFront&) = delete;
    AIEngineFront& operator=(const AIEngineFront&) = delete;

    void RunTreeInferencePipeline(const int8_t* quant_weights, const QuantDescriptor* weight_scales, float* root_hidden_state, uint32_t target_tokens);
    void SetThreadAffinity(int core_id);
    
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

};
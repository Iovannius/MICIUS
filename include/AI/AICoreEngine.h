#pragma once

#include "AIPrimitives.h"
#include "AIHetMemPool.h"
#include "AIDataFront.h"

#include <cstdint>
#include <vector>
#include <thread>
#include <atomic>

struct alignas(64) WorkerContext {
    int16_t* workspace_i16;
    alignas(64) float* logits;
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

class AIEngineFront
{

private:

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
    
    struct alignas(64) PaddedWorkerGen {
        std::atomic<uint32_t> val;
        // 构造函数初始化，避免残留脏数据
        PaddedWorkerGen()
        {
            val.store(0, std::memory_order_relaxed);
        }
    };
    
    PaddedWorkerGen worker_gen[16];

    alignas(64) std::atomic<int> seed_claimed{1};
    alignas(64) FrontierTask global_seed_task;
    
    std::vector<ChaseLevDeque<FrontierTask, 1024>> local_queues;
    std::atomic<uint8_t> branch_tombstones[DraftProposal::MAX_K_DRAFTS];
    
    std::vector<int> ProbePhysicalCores(); void WorkerRoutine(int worker_idx);

    inline void ComputeStateTransition(uint32_t token, const float* parent_state, float* new_state, uint32_t hidden_dim);
    
public:

    AIEngineFront(size_t pool_size, uint32_t max_seq_len, uint32_t hidden_dim);
    ~AIEngineFront();
    
    AIEngineFront(const AIEngineFront&) = delete;
    AIEngineFront& operator=(const AIEngineFront&) = delete;

    void RunTreeInferencePipeline(const int8_t* quant_weights, const QuantDescriptor* weight_scales, float* root_hidden_state, uint32_t target_tokens);
    void SetThreadAffinity(int core_id);
    
    inline DraftProposal* GetSharedProposal(int idx) const
    {
        return shared_proposal[idx];
    }

};
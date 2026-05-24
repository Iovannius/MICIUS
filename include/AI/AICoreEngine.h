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

//v0.3分布式双端工作窃取队列
template<typename T, size_t Cap> class WorkStealQueue
{
    alignas(64) T data[Cap];
    uint32_t head=0, tail=0;
    NanoSpinLock lock;

public:
    inline bool push_bottom(const T& item)
    {
        lock.lock();
        bool ok=(tail-head<Cap);
        if(ok) data[tail++&(Cap-1)]=item;
        lock.unlock();
        return ok;
    }
    inline bool pop_bottom(T& item)
    {
        lock.lock();
        bool ok=(head<tail);
        if(ok) item=data[--tail&(Cap-1)];
        lock.unlock();
        return ok;
    }
    inline bool steal_top(T& item)
    {
        lock.lock();
        bool ok=(head<tail);
        if(ok) item=data[head++&(Cap-1)];
        lock.unlock();
        return ok;
    }
    inline void clear()
    {
        lock.lock();
        head=tail=0;
        lock.unlock();
    }
    inline bool empty() const
    {
        return head==tail;
    }
};

struct FrontierTask {
    uint32_t gen_id, parent_slot, alloc_slot, input_token;
    int32_t parent_tree_idx;
    uint32_t branch_id;
};

class AIEngineFront
{
private:
    HeteroMemPool* mem_pool;        
    KVCacheRingBuffer* kv_cache;    
    AiDataFront* data_front;        
    DraftProposal* shared_proposal; 
    
    uint32_t hidden_dim;
    int hq_core_id;                 
    std::vector<int> worker_cores;  
    
    std::vector<std::thread> worker_threads;
    std::vector<WorkerContext*> worker_ctxs; 

    std::atomic<uint32_t> global_generation{0}; 
    std::atomic<bool> global_terminate{false};
    
    std::vector<WorkStealQueue<FrontierTask, 1024>> local_queues;
    std::atomic<uint8_t> branch_tombstones[DraftProposal::MAX_K_DRAFTS];

    std::vector<int> ProbePhysicalCores();
    void WorkerRoutine(int worker_idx);
    
    inline void ComputeStateTransition(uint32_t token, const float* parent_state, float* new_state, uint32_t hidden_dim);
    inline void ExtractTop3_AVX2(float* logits, uint32_t vocab_size, uint32_t out_tokens[3]);

public:
    AIEngineFront(size_t pool_size, uint32_t max_seq_len, uint32_t hidden_dim);
    ~AIEngineFront();
    AIEngineFront(const AIEngineFront&) = delete;
    AIEngineFront& operator=(const AIEngineFront&) = delete;

    void RunTreeInferencePipeline(const int8_t* quant_weights, const QuantDescriptor* weight_scales, float* root_hidden_state, uint32_t target_tokens);
    void SetThreadAffinity(int core_id);
    inline DraftProposal* GetSharedProposal() const { return shared_proposal; }
};
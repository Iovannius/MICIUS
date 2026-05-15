#pragma once

#include "AIPrimitives.h"
#include "AIHetMemPool.h"
#include "AIDataFront.h"

#include <cstdint>
#include <vector>
#include <thread>
#include <atomic>

//-------------------------------------------------------------------------
//运算任务描述块
//-------------------------------------------------------------------------
//对齐64字节防止伪共享
struct alignas(64) WorkerTaskBox {
    uint32_t parent_slot;          //依赖的父节点KV槽位
    uint32_t allocated_slot;       //分配给当前推测分支的独占KV槽位
    uint32_t input_token;          //输入的假想Token
    int32_t  tree_node_idx;        //在DraftProposal里的拓扑索引
    
    uint32_t predicted_token;      //算子输出的最高概率Token
    
    //0=空闲, 1=工作(HQ->Worker), 2=完成(Worker->HQ)
    std::atomic<uint32_t> state_flag; 
};

//-------------------------------------------------------------------------
//异构调度主控
//树状拓扑管理，控制分支扇出、并发流调度与iGPU验证回滚
//-------------------------------------------------------------------------
class AIEngineFront
{
private:
    HeteroMemPool* mem_pool;        
    KVCacheRingBuffer* kv_cache;    
    AiDataFront* data_front;        
    DraftProposal* shared_proposal; 
    
    int hq_core_id;                 //总控物理核 (Core 0)
    std::vector<int> worker_cores;  //运算物理核 (Core 1, 2, 3...)
    
    std::vector<std::thread> worker_threads;
    std::vector<WorkerTaskBox*> task_boxes; //每个Worker的专属TaskBox
    
    std::atomic<bool> global_terminate;

    std::vector<int> ProbePhysicalCores();
    inline double GetTime() const;

    //工作核心执行的无锁热路径
    void WorkerRoutine(int worker_idx);
    
    //基于Token的隐状态转移 (Transformer Layer/RNN Cell)
    inline void ComputeStateTransition(uint32_t token, const float* parent_state, float* new_state, uint32_t hidden_dim);

public:
    AIEngineFront(size_t pool_size, uint32_t max_seq_len, uint32_t hidden_dim);
    ~AIEngineFront();

    AIEngineFront(const AIEngineFront&) = delete;
    AIEngineFront& operator=(const AIEngineFront&) = delete;

    //V2主频管线
    //接收量化权重，执行树状推测与回滚
    void RunTreeInferencePipeline(
        const int8_t* quant_weights, 
        const QuantDescriptor* weight_scales,
        float* root_hidden_state, 
        uint32_t target_tokens);

    void SetThreadAffinity(int core_id);

    inline DraftProposal* GetSharedProposal() const
    {
        return shared_proposal;
    }
};
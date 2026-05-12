#pragma once

#include "AIPrimitives.h"
#include "AIHetMemPool.h"
#include "AIDataFront.h"
#include <cstdint>
#include <vector>

/**
 *@brief 异构调度主控
 **实现硬件拓扑探测、线程绑核、跨总线握手时序控制及流水线重叠。
 *负责在同构核心架构或Intel P核驱动投机模型，并与 iGPU 共享物理内存进行验证。
 */
class AIEngineFront
{
private:
    HeteroMemPool* mem_pool;        //异构无拷贝内存池
    KVCacheRingBuffer* kv_cache;    //投机采样的KV Cache环形缓冲区
    AiDataFront* data_front;        //数据流整流前端（负责L1D热复用与碎片整理）
    DraftProposal* shared_proposal; //跨总线共享的投机提案区（对齐缓存行）
    
    int primary_physical_core;      //锁定的主物理核ID

    std::vector<int> ProbePhysicalCores();
    
    /**
     *@brief 获取高精度时间戳
     */
    inline double GetTime() const;

public:
    /**
     *@param pool_size 零拷贝区域大小
     *@param max_seq_len KV Cache最大序列长度
     *@param hidden_dim 模型隐藏层维度
     */
    AIEngineFront(size_t pool_size, uint32_t max_seq_len, uint32_t hidden_dim);
    
    ~AIEngineFront();

    //禁用拷贝构造与赋值
    AIEngineFront(const AIEngineFront&) = delete;
    AIEngineFront& operator=(const AIEngineFront&) = delete;

    /**
     *@brief 执行投机推理管线
     **包含CPU端投机Token生成、填充共享内存、触发iGPU验证以及根据验证结果进行KV Cache回滚。
     **@param draft_weights 投机模型权重（Host侧）
     *@param current_hidden_state 当前层的隐藏状态
     *@param target_tokens 目标生成Token数量
     */
    void RunInferencePipeline(const float* draft_weights, 
                              float* current_hidden_state, 
                              uint32_t target_tokens);

    /**
     *@brief 锁定执行线程到指定的物理核心
     */
    void SetThreadAffinity(int core_id);

    /**
     *@brief 获取当前共享提案区的状态
     */
    uint32_t GetDispatcherState() const {
        return shared_proposal->state_flags.load(std::memory_order_acquire);
    }

    /**
     *@brief 获取共享提案区指针
     */
    inline DraftProposal* GetSharedProposal() const {
        return shared_proposal;
    }
};
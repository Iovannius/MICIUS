#pragma once

#include "AIPrimitives.h"
#include "AIHetMemPool.h"
#include <cstdint>
#include <cstddef>

//-------------------------------------------------------------------------
//异构数据整流前端
//实现L1D缓存热复用、KV Cache碎片整理和跨总线数据对齐的专门实现等
//-------------------------------------------------------------------------

class AiDataFront 
{

private:
    KVCacheRingBuffer* kv_cache;
    DraftProposal* shared_proposal;

public:
    //注入KV环形缓存池与零拷贝共享提案区的构造函数
    AiDataFront(KVCacheRingBuffer* kv, DraftProposal* proposal);

    //L1D热复用整流
    //衍生自ControlFlow.cpp中的local_micro_compact
    //引入StreamingLLM或H2O算法淘汰低注意力分数的Token时KV Cache会出现内存空洞
    //该函数负责将存活的Hidden States高速紧凑化，已处理越界和非对齐问题
    void LocalMicroCompactKV(float* hidden_states, uint32_t* valid_masks, size_t current_seq_len, size_t hidden_dim);

    //总线层内存搬运
    //衍生自ControlFlow.cpp中的global_cluster_gc
    //当iGPU拒绝了极大跨度的草稿Token（例如树状预测失败）时触发
    //类似磁盘逻辑，对连续失效的簇进行大块覆写
    void GlobalBusRectify(float* kv_pool_base, size_t dest_chunk_idx, size_t src_chunk_idx, size_t chunks_to_move);

    //无拷贝协议装填
    //保持inline以确保内层热循环的极速吞吐
    inline void PrepareDraftForiGPU(uint32_t token_id, float confidence_logit)
    {
        // 仅当置信度高于阈值时才进行装填
        if (confidence_logit > -10.0f) { 
            shared_proposal->PushToken(token_id, confidence_logit);
        }
    }
};
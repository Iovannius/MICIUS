#pragma once

#include "AIPrimitives.h"
#include "AIHetMemPool.h"
#include <cstdint>
#include <cstddef>

//-------------------------------------------------------------------------
//v0.4.1异构数据流前端
//适配工作阵列，支持INT8量化描述符与权重数据同步整流及树状拓扑探针协议
//-------------------------------------------------------------------------

class AiDataFront 
{
private:
    KVCacheRingBuffer* kv_cache;
    DraftProposal* shared_proposal;

public:
    //KV环形缓存池与零拷贝共享提案区的构造函数
    AiDataFront(KVCacheRingBuffer* kv, DraftProposal* proposal);

    //L1D热复用整流 (FP32)
    /*启用__restrict*/
    //解除4K指针别名惩罚，允许乱序执行引擎不掉速重排
    void LocalMicroCompactKV_FP32(float* __restrict hidden_states, 
                                  uint32_t* __restrict valid_masks, 
                                  size_t current_seq_len, 
                                  size_t hidden_dim);

    //L1D热复用整流 (INT8)
    void LocalMicroCompactKV_INT8(int8_t* __restrict quant_states, 
                                  QuantDescriptor* __restrict scales, 
                                  uint32_t* __restrict valid_masks, 
                                  size_t current_seq_len, 
                                  size_t hidden_dim);

    //总线层内存搬运
    void GlobalBusRectify(void* kv_pool_base, 
                          size_t dest_chunk_idx, 
                          size_t src_chunk_idx, 
                          size_t chunks_to_move);

    //V0.4.1无拷贝协议装填，适配树状拓扑
    inline void PrepareTreeDraftForiGPU(uint32_t token_id, 
                                        int32_t parent_idx, 
                                        float confidence_logit)
    {
        /*当且仅当置信度高于阈值时进行装填，减少无效总线握手*/
        if (confidence_logit > -10.0f) { 
            shared_proposal->SpawnNodeConcurrent(token_id, parent_idx, 0xFFFFFFFF, false);
        }
    }
};
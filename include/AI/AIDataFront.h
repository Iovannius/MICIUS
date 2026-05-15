#pragma once

#include "AIPrimitives.h"
#include "AIHetMemPool.h"
#include <cstdint>
#include <cstddef>

//-------------------------------------------------------------------------
//V2异构数据流前端
//适配4核工作阵列，支持INT8量化描述符与权重数据同步整流及树状拓扑探针协议
//-------------------------------------------------------------------------

class AiDataFront 
{
private:
    KVCacheRingBuffer* kv_cache;
    DraftProposal* shared_proposal;

public:
    //注入KV环形缓存池与零拷贝共享提案区的构造函数
    AiDataFront(KVCacheRingBuffer* kv, DraftProposal* proposal);

    //L1D热复用整流 (FP32)
    //用于处理尚未量化的中间层激活值或Logits缓存空洞
    void LocalMicroCompactKV_FP32(float* hidden_states, 
                                  uint32_t* valid_masks, 
                                  size_t current_seq_len, 
                                  size_t hidden_dim);

    //L1D热复用整流 (INT8)
    //在L1内发生内存空洞时，必须将存活INT8权重与其对应的反量化参数在同一次总线/L1D事务中打包搬运
    
    //假设Block Size为64字节
    void LocalMicroCompactKV_INT8(int8_t* quant_states, 
                                  QuantDescriptor* scales, 
                                  uint32_t* valid_masks, 
                                  size_t current_seq_len, 
                                  size_t hidden_dim);

    //总线层内存搬运
    //树状预测因极大跨度发生预测失败（比如全部分支被否决）时触发，进行粗粒度大块覆写
    void GlobalBusRectify(void* kv_pool_base, 
                          size_t dest_chunk_idx, 
                          size_t src_chunk_idx, 
                          size_t chunks_to_move);

    //V2无拷贝协议装填，适配树状拓扑
    //引入parent_idx，允许工作核心构建并发探索树
    //保持inline以保证内层热循环的吞吐
    inline void PrepareTreeDraftForiGPU(uint32_t token_id, 
                                        int32_t parent_idx, 
                                        float confidence_logit)
    {
        //当且仅当置信度高于阈值时进行装填，减少无效总线握手
        if (confidence_logit > -10.0f) { 
            shared_proposal->PushTreeToken(token_id, parent_idx);
        }
    }
};
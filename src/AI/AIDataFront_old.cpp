#include "AIPrimitives.h"
#include "AIHetMemPool.h"
#include <immintrin.h>
#include <cstring>

//-------------------------------------------------------------------------
//异构数据流前端 (Data Flow & Rectification Frontend)
//L1D缓存热复用、KV Cache碎片整理、跨总线数据对齐的专门实现
//-------------------------------------------------------------------------

class AiDataFront 
{
private:
    KVCacheRingBuffer* kv_cache;
    DraftProposal* shared_proposal;

public:
    AiDataFront(KVCacheRingBuffer* kv, DraftProposal* proposal)
        : kv_cache(kv), shared_proposal(proposal) {}

    /*
    //L1D热复用整流，衍生自ControlFlow.cpp运算整流器中的local_micro_compact
    //引入StreamingLLM或H2O算法淘汰低注意力分数的Token时KV Cache会出现内存空洞
    //该函数负责将存活的Hidden States高速紧凑化
    void LocalMicroCompactKV(float* hidden_states, uint32_t* valid_masks, size_t current_seq_len, size_t hidden_dim)
    {
        size_t left = 0;
        size_t right = current_seq_len - 1;

        //判定实现为256位寄存器直接吞吐测试
        //此处简化为uint32测试，实际可根据ChunkMask16布局使用_mm256_testz_si256
        auto is_token_dead = [&](size_t idx) -> bool {
            return valid_masks[idx] == 0x00000000;
        };

        //左侧寻找空洞，右侧寻找存活Token
        while (left < right) {
            
            while (left < right && !is_token_dead(left)) left++;
            while (left < right && is_token_dead(right)) right--;

            if (left < right) {
                //64字节对齐下的编译器激进优化（几乎全是VMOVAPS）
                float* dest = hidden_states + left * hidden_dim;
                float* src  = hidden_states + right * hidden_dim;

                //假设hidden_dim为128
                //对应16个YMM寄存器，分两次循环展开
                for (size_t offset = 0; offset < hidden_dim; offset += 8) {
                    __m256 val = _mm256_load_ps(src + offset);
                    _mm256_store_ps(dest + offset, val);
                }

                //继承掩码并抹除原死区痕迹
                valid_masks[left] = valid_masks[right];
                valid_masks[right] = 0x00000000;

                right--;
            }
        }
    }

    */

    void LocalMicroCompactKV(float* hidden_states, uint32_t* valid_masks, size_t current_seq_len, size_t hidden_dim)
    {
        size_t left = 0;
        size_t right = current_seq_len - 1;

        auto is_token_dead = [&](size_t idx) -> bool {
            return valid_masks[idx] == 0x00000000;
        };

        //预计算向量化边界，防止hidden_dim不是8的倍数导致越界
        size_t vec_end = (hidden_dim / 8) * 8; 

        while (left < right) {
            while (left < right && !is_token_dead(left)) left++;
            while (left < right && is_token_dead(right)) right--;

            if (left < right) {
                float* dest = hidden_states + left * hidden_dim;
                float* src  = hidden_states + right * hidden_dim;

                size_t offset = 0;
                //妥协与安全使用loadu/storeu (VMOVUPS)
                //即使dest和src在运行时不是32字节对齐，也不会触发#GP
                for (; offset < vec_end; offset += 8) {
                    __m256 val = _mm256_loadu_ps(src + offset);
                    _mm256_storeu_ps(dest + offset, val);
                }

                //标量尾部处理
                //专门清理remaining的维度（例如hidden_dim=100时的最后4个float）
                for (; offset < hidden_dim; ++offset) {
                    dest[offset] = src[offset];
                }

                valid_masks[left] = valid_masks[right];
                valid_masks[right] = 0x00000000;

                right--;
            }
        }
    }

    //总线级内存搬运，衍生自ControlFlow.cpp运算整流器中的global_cluster_gc
    //当iGPU拒绝了极大跨度的草稿Token（例如树状预测失败）时触发总线级搬运
    //类似磁盘逻辑，对连续失效的簇进行覆写
    void GlobalBusRectify(float* kv_pool_base, size_t dest_chunk_idx, size_t src_chunk_idx, size_t chunks_to_move)
    {

        size_t bytes_to_move = chunks_to_move * sizeof(TensorMicroChunk);

        //强拷阶段要求地址绝对对齐
        //此处假设每Chunk共16个float
        float* dest_ptr = kv_pool_base + dest_chunk_idx * 16; 
        float* src_ptr  = kv_pool_base + src_chunk_idx * 16;

        //4KB簇对齐强拷，绕过处理器缓存直接走内存控制器（可选非临时存储指令_mm256_stream_ps）
        std::memcpy(dest_ptr, src_ptr, bytes_to_move);
    }

    //无拷贝协议装填
    inline void PrepareDraftForiGPU(uint32_t token_id, float confidence_logit)
    {
        //仅当置信度高于阈值时才进行装填
        if (confidence_logit > -10.0f) { 
            shared_proposal->PushToken(token_id, confidence_logit);
        }
    }
};
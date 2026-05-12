#include "AIDataFront.h"

#include <immintrin.h>
#include <cstring>

//-------------------------------------------------------------------------
//异构数据流前端
//L1D缓存热复用、KV Cache碎片整理、跨总线数据对齐的专门实现
//-------------------------------------------------------------------------

AiDataFront::AiDataFront(KVCacheRingBuffer* kv, DraftProposal* proposal)
    : kv_cache(kv), shared_proposal(proposal) {}

//-------------------------------------------------------------------------
//L1D热复用整流 (安全版)
//-------------------------------------------------------------------------
void AiDataFront::LocalMicroCompactKV(float* hidden_states, uint32_t* valid_masks, size_t current_seq_len, size_t hidden_dim)
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
            for (; offset < vec_end; offset += 8) {
                __m256 val = _mm256_loadu_ps(src + offset);
                _mm256_storeu_ps(dest + offset, val);
            }

            //标量尾部处理
            for (; offset < hidden_dim; ++offset) {
                dest[offset] = src[offset];
            }

            valid_masks[left] = valid_masks[right];
            valid_masks[right] = 0x00000000;

            right--;
        }
    }
}

//-------------------------------------------------------------------------
//总线级内存搬运
//-------------------------------------------------------------------------
void AiDataFront::GlobalBusRectify(float* kv_pool_base, size_t dest_chunk_idx, size_t src_chunk_idx, size_t chunks_to_move)
{
    size_t bytes_to_move = chunks_to_move * sizeof(TensorMicroChunk);

    float* dest_ptr = kv_pool_base + dest_chunk_idx * 16; 
    float* src_ptr  = kv_pool_base + src_chunk_idx * 16;

    std::memcpy(dest_ptr, src_ptr, bytes_to_move);
}
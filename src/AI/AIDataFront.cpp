#include "AIDataFront.h"

#include <immintrin.h>
#include <cstring>

//-------------------------------------------------------------------------
//V2异构运算整流前端实现
//AVX2软解与混合精度场景优化
//-------------------------------------------------------------------------

AiDataFront::AiDataFront(KVCacheRingBuffer* kv, DraftProposal* proposal)
    : kv_cache(kv), shared_proposal(proposal) {}

//L1D热复用整流 (FP32)
void AiDataFront::LocalMicroCompactKV_FP32(float* hidden_states, uint32_t* valid_masks, size_t current_seq_len, size_t hidden_dim)
{
    size_t left = 0;
    size_t right = current_seq_len - 1;

    auto is_token_dead = [&](size_t idx) -> bool {
        return valid_masks[idx] == 0x00000000;
    };

    size_t vec_end = (hidden_dim / 8) * 8; 

    while (left < right) {
        while (left < right && !is_token_dead(left)) left++;
        while (left < right && is_token_dead(right)) right--;

        if (left < right) {
            float* dest = hidden_states + left * hidden_dim;
            const float* src  = hidden_states + right * hidden_dim;

            size_t offset = 0;
            //AVX2浮点搬运
            //256-bit=32B=8*FP32
            for (; offset < vec_end; offset += 8) {
                __m256 val = _mm256_loadu_ps(src + offset);
                _mm256_storeu_ps(dest + offset, val);
            }

            for (; offset < hidden_dim; ++offset) {
                dest[offset] = src[offset];
            }

            valid_masks[left] = valid_masks[right];
            valid_masks[right] = 0x00000000;
            right--;
        }
    }
}

//L1D热复用整流 (INT8)
void AiDataFront::LocalMicroCompactKV_INT8(int8_t* quant_states, QuantDescriptor* scales, uint32_t* valid_masks, size_t current_seq_len, size_t hidden_dim)
{
    size_t left = 0;
    size_t right = current_seq_len - 1;

    auto is_token_dead = [&](size_t idx) -> bool {
        return valid_masks[idx] == 0x00000000;
    };

    //一次处理32个INT8权重
    size_t vec_end_q = (hidden_dim / 32) * 32; 
    
    //假定Block Size为64B，计算每个Token对应的描述符量
    size_t num_blocks = (hidden_dim + 63) / 64; 

    while (left < right) {
        while (left < right && !is_token_dead(left)) left++;
        while (left < right && is_token_dead(right)) right--;

        if (left < right) {
            //搬运底层INT8权重
            int8_t* dest_q = quant_states + left * hidden_dim;
            const int8_t* src_q  = quant_states + right * hidden_dim;
            size_t offset = 0;
            
            for (; offset < vec_end_q; offset += 32) {
                __m256i val = _mm256_loadu_si256((const __m256i*)(src_q + offset));
                _mm256_storeu_si256((__m256i*)(dest_q + offset), val);
            }
            //INT8标量尾部
            for (; offset < hidden_dim; ++offset) {
                dest_q[offset] = src_q[offset];
            }

            //同步搬运绑定的QuantDescriptor
            QuantDescriptor* dest_s = scales + left * num_blocks;
            const QuantDescriptor* src_s = scales + right * num_blocks;
            
            size_t desc_bytes = num_blocks * sizeof(QuantDescriptor);
            size_t vec_end_s = (desc_bytes / 32) * 32;
            size_t b_offset = 0;

            //如果有2个及以上的QuantDescriptor则AVX2合并解决
            for (; b_offset < vec_end_s; b_offset += 32) {
                __m256i val = _mm256_loadu_si256((const __m256i*)((const uint8_t*)src_s + b_offset));
                _mm256_storeu_si256((__m256i*)((uint8_t*)dest_s + b_offset), val);
            }
            //QuantDescriptor标量尾部，以字节处理避免越界
            for (; b_offset < desc_bytes; ++b_offset) {
                ((uint8_t*)dest_s)[b_offset] = ((const uint8_t*)src_s)[b_offset];
            }

            valid_masks[left] = valid_masks[right];
            valid_masks[right] = 0x00000000;
            right--;
        }
    }
}

//总线级内存搬运
//兼容无类型指针
void AiDataFront::GlobalBusRectify(void* kv_pool_base, size_t dest_chunk_idx, size_t src_chunk_idx, size_t chunks_to_move)
{
    //无论是INT8还是FP32的Chunk都是绝对的64字节对齐
    size_t bytes_to_move = chunks_to_move * 64; 

    uint8_t* base_ptr = static_cast<uint8_t*>(kv_pool_base);
    uint8_t* dest_ptr = base_ptr + (dest_chunk_idx * 64); 
    const uint8_t* src_ptr  = base_ptr + (src_chunk_idx * 64);

    std::memcpy(dest_ptr, src_ptr, bytes_to_move);
}
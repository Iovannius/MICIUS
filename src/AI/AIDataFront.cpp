#include "AIDataFront.h"

#include <immintrin.h>
#include <cstring>

//-------------------------------------------------------------------------
//v0.4.1异构运算整流前端实现
//融合分支预测友好型检索与OoO物理寄存器饱和展开 (4-Way Unrolling)
//-------------------------------------------------------------------------

AiDataFront::AiDataFront(KVCacheRingBuffer* kv, DraftProposal* proposal)
    : kv_cache(kv), shared_proposal(proposal) {}

//L1D热复用整流 (FP32)
void AiDataFront::LocalMicroCompactKV_FP32(float* __restrict hidden_states, uint32_t* __restrict valid_masks, size_t current_seq_len, size_t hidden_dim)
{
    size_t left = 0;
    size_t right = current_seq_len - 1;

    //预计算循环展开边界
    size_t vec_end_unroll = (hidden_dim / 32) * 32; //4*8个float=32个元素
    size_t vec_end = (hidden_dim / 8) * 8;          //单路AVX2，8个float

    while (left < right) {
        //信任现代CPU的分支预测器处理这种短序列
        while (left < right && valid_masks[left] != 0x00000000) left++;
        while (left < right && valid_masks[right] == 0x00000000) right--;

        if (left < right) {
            float* __restrict dest = hidden_states + left * hidden_dim;
            const float* __restrict src  = hidden_states + right * hidden_dim;

            size_t offset = 0;
            
            //4路展开
            for (; offset < vec_end_unroll; offset += 32) {
                __m256 val0 = _mm256_loadu_ps(src + offset + 0);
                __m256 val1 = _mm256_loadu_ps(src + offset + 8);
                __m256 val2 = _mm256_loadu_ps(src + offset + 16);
                __m256 val3 = _mm256_loadu_ps(src + offset + 24);
                
                _mm256_storeu_ps(dest + offset + 0, val0);
                _mm256_storeu_ps(dest + offset + 8, val1);
                _mm256_storeu_ps(dest + offset + 16, val2);
                _mm256_storeu_ps(dest + offset + 24, val3);
            }

            //处理剩余的完整AVX2块
            for (; offset < vec_end; offset += 8) {
                _mm256_storeu_ps(dest + offset, _mm256_loadu_ps(src + offset));
            }

            //处理标量尾部
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
void AiDataFront::LocalMicroCompactKV_INT8(int8_t* __restrict quant_states, QuantDescriptor* __restrict scales, uint32_t* __restrict valid_masks, size_t current_seq_len, size_t hidden_dim)
{
    size_t left = 0;
    size_t right = current_seq_len - 1;

    //INT8展开边界计算
    //4*32B(YMM)=128字节
    size_t vec_end_q_unroll = (hidden_dim / 128) * 128; 
    size_t vec_end_q = (hidden_dim / 32) * 32; 
    
    //量化描述符边界计算
    size_t num_blocks = (hidden_dim + 63) / 64; 
    size_t desc_bytes = num_blocks * sizeof(QuantDescriptor);
    size_t vec_end_s_unroll = (desc_bytes / 128) * 128;
    size_t vec_end_s = (desc_bytes / 32) * 32;

    while (left < right) {
        //依然采用纯粹的标量检索
        while (left < right && valid_masks[left] != 0x00000000) left++;
        while (left < right && valid_masks[right] == 0x00000000) right--;

        if (left < right) {
            //1. 搬运底层INT8权重
            int8_t* __restrict dest_q = quant_states + left * hidden_dim;
            const int8_t* __restrict src_q  = quant_states + right * hidden_dim;
            size_t offset = 0;
            
            //4路并发重整（INT8权重）
            for (; offset < vec_end_q_unroll; offset += 128) {
                __m256i v0 = _mm256_loadu_si256((const __m256i*)(src_q + offset + 0));
                __m256i v1 = _mm256_loadu_si256((const __m256i*)(src_q + offset + 32));
                __m256i v2 = _mm256_loadu_si256((const __m256i*)(src_q + offset + 64));
                __m256i v3 = _mm256_loadu_si256((const __m256i*)(src_q + offset + 96));
                
                _mm256_storeu_si256((__m256i*)(dest_q + offset + 0), v0);
                _mm256_storeu_si256((__m256i*)(dest_q + offset + 32), v1);
                _mm256_storeu_si256((__m256i*)(dest_q + offset + 64), v2);
                _mm256_storeu_si256((__m256i*)(dest_q + offset + 96), v3);
            }

            for (; offset < vec_end_q; offset += 32) {
                _mm256_storeu_si256((__m256i*)(dest_q + offset), _mm256_loadu_si256((const __m256i*)(src_q + offset)));
            }
            
            for (; offset < hidden_dim; ++offset) {
                dest_q[offset] = src_q[offset];
            }

            //2. 搬运对应的QuantDescriptor
            QuantDescriptor* __restrict dest_s = scales + left * num_blocks;
            const QuantDescriptor* __restrict src_s = scales + right * num_blocks;
            size_t b_offset = 0;

            //4路展开并发（量化描述符）
            for (; b_offset < vec_end_s_unroll; b_offset += 128) {
                __m256i v0 = _mm256_loadu_si256((const __m256i*)((const uint8_t*)src_s + b_offset + 0));
                __m256i v1 = _mm256_loadu_si256((const __m256i*)((const uint8_t*)src_s + b_offset + 32));
                __m256i v2 = _mm256_loadu_si256((const __m256i*)((const uint8_t*)src_s + b_offset + 64));
                __m256i v3 = _mm256_loadu_si256((const __m256i*)((const uint8_t*)src_s + b_offset + 96));
                
                _mm256_storeu_si256((__m256i*)((uint8_t*)dest_s + b_offset + 0), v0);
                _mm256_storeu_si256((__m256i*)((uint8_t*)dest_s + b_offset + 32), v1);
                _mm256_storeu_si256((__m256i*)((uint8_t*)dest_s + b_offset + 64), v2);
                _mm256_storeu_si256((__m256i*)((uint8_t*)dest_s + b_offset + 96), v3);
            }

            for (; b_offset < vec_end_s; b_offset += 32) {
                __m256i val = _mm256_loadu_si256((const __m256i*)((const uint8_t*)src_s + b_offset));
                _mm256_storeu_si256((__m256i*)((uint8_t*)dest_s + b_offset), val);
            }
            
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
void AiDataFront::GlobalBusRectify(void* kv_pool_base, size_t dest_chunk_idx, size_t src_chunk_idx, size_t chunks_to_move)
{
    //无论是INT8还是FP32的Chunk都是绝对的64字节对齐
    size_t bytes_to_move = chunks_to_move * 64; 
    uint8_t* dest_ptr = static_cast<uint8_t*>(kv_pool_base) + (dest_chunk_idx * 64); 
    const uint8_t* src_ptr  = static_cast<uint8_t*>(kv_pool_base) + (src_chunk_idx * 64);

    std::memcpy(dest_ptr, src_ptr, bytes_to_move);
}
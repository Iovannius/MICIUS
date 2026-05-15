#include "AIPrimitives.h"
#include "AIComputKrnl.h"

#include <immintrin.h>
#include <cmath>
#include <algorithm>
#include <vector>
#include <cstdint>

//-------------------------------------------------------------------------
//V2核心运算引擎
//-------------------------------------------------------------------------

/**
 *@brief AVX2寄存器内8个32位整数的水平求和
 */
inline int32_t _mm256_reduce_add_epi32(__m256i v)
{
    __m128i vlow = _mm256_castsi256_si128(v);
    __m128i vhigh = _mm256_extracti128_si256(v, 1);
    __m128i vsum = _mm_add_epi32(vlow, vhigh);
    
    vsum = _mm_add_epi32(vsum, _mm_srli_si128(vsum, 8));
    vsum = _mm_add_epi32(vsum, _mm_srli_si128(vsum, 4));
    
    return _mm_cvtsi128_si32(vsum);
}

/**
 *@brief V2版Small-GEMV
 *INT16路线，兼容有符号激活值与权重同时规避AVX2通道乱序
 */
extern "C" void DraftGEMV_INT8_AVX2(
    const int8_t* __restrict quant_weights, 
    const QuantDescriptor* __restrict scales,
    const float* __restrict input, 
    float* __restrict output, 
    uint32_t hidden_dim, 
    uint32_t vocab_size)
{
    //使用thread_local复用内存，避免热路径malloc以适配超大量级的hidden_dim
    static thread_local std::vector<int16_t> dyn_quant_input;
    if (dyn_quant_input.size() < hidden_dim) {
        dyn_quant_input.resize(hidden_dim);
    }

    //Per-Tensor安全动态量化
    //FP32->线性连续INT16，绝对值最大值映射到127，确保后续乘法不溢出INT32
    float max_abs_val = 1e-9f;
    for (uint32_t k = 0; k < hidden_dim; ++k) {
        max_abs_val = std::max(max_abs_val, std::abs(input[k]));
    }
    
    float input_scale = max_abs_val / 127.0f;
    float inv_input_scale = 127.0f / max_abs_val;

    uint32_t k = 0;
    for (; k + 7 < hidden_dim; k += 8) {
        __m256 f = _mm256_loadu_ps(input + k);
        __m256i i32 = _mm256_cvtps_epi32(_mm256_mul_ps(f, _mm256_set1_ps(inv_input_scale)));
        
        //提取高低位的128-bit，使用128-bit的packs
        //保持内存线性顺序
        __m128i lo = _mm256_castsi256_si128(i32);
        __m128i hi = _mm256_extracti128_si256(i32, 1);
        __m128i i16 = _mm_packs_epi32(lo, hi);
        
        _mm_storeu_si128((__m128i*)(dyn_quant_input.data() + k), i16);
    }
    for (; k < hidden_dim; ++k) {
        dyn_quant_input[k] = static_cast<int16_t>(input[k] * inv_input_scale);
    }

    uint32_t num_blocks = (hidden_dim + 63) / 64;

    //Block-Wise的GEMV展开
    //(INT16 Et INT8_extended)
    uint32_t m = 0;

    for (; m + 3 < vocab_size; m += 4) {
        float row_sum0 = 0.0f, row_sum1 = 0.0f, row_sum2 = 0.0f, row_sum3 = 0.0f;

        for (uint32_t b = 0; b < num_blocks; ++b) {
            uint32_t k_start = b * 64;
            uint32_t k_end = std::min(k_start + 64, hidden_dim);

            __m256i acc0 = _mm256_setzero_si256();
            __m256i acc1 = _mm256_setzero_si256();
            __m256i acc2 = _mm256_setzero_si256();
            __m256i acc3 = _mm256_setzero_si256();

            const int8_t* w0 = quant_weights + (m + 0) * hidden_dim;
            const int8_t* w1 = quant_weights + (m + 1) * hidden_dim;
            const int8_t* w2 = quant_weights + (m + 2) * hidden_dim;
            const int8_t* w3 = quant_weights + (m + 3) * hidden_dim;

            uint32_t kk = k_start;
            //每次取16个参数
            //16x INT16输入及16x INT8权重
            for (; kk + 15 < k_end; kk += 16) {
                __m256i x16 = _mm256_loadu_si256((const __m256i*)(dyn_quant_input.data() + kk));

                //载入INT8权重并符号扩展为INT16
                __m256i w16_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(w0 + kk)));
                __m256i w16_1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(w1 + kk)));
                __m256i w16_2 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(w2 + kk)));
                __m256i w16_3 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(w3 + kk)));

                acc0 = _mm256_add_epi32(acc0, _mm256_madd_epi16(x16, w16_0));
                acc1 = _mm256_add_epi32(acc1, _mm256_madd_epi16(x16, w16_1));
                acc2 = _mm256_add_epi32(acc2, _mm256_madd_epi16(x16, w16_2));
                acc3 = _mm256_add_epi32(acc3, _mm256_madd_epi16(x16, w16_3));
            }

            int32_t b_sum0 = _mm256_reduce_add_epi32(acc0);
            int32_t b_sum1 = _mm256_reduce_add_epi32(acc1);
            int32_t b_sum2 = _mm256_reduce_add_epi32(acc2);
            int32_t b_sum3 = _mm256_reduce_add_epi32(acc3);

            for (; kk < k_end; ++kk) {
                int16_t x_val = dyn_quant_input[kk];
                b_sum0 += x_val * static_cast<int16_t>(w0[kk]);
                b_sum1 += x_val * static_cast<int16_t>(w1[kk]);
                b_sum2 += x_val * static_cast<int16_t>(w2[kk]);
                b_sum3 += x_val * static_cast<int16_t>(w3[kk]);
            }

            //Block级别反量化
            //Descriptor数组结构为vocab_size*num_blocks
            float c_scale0 = input_scale * scales[(m + 0) * num_blocks + b].scale;
            float c_scale1 = input_scale * scales[(m + 1) * num_blocks + b].scale;
            float c_scale2 = input_scale * scales[(m + 2) * num_blocks + b].scale;
            float c_scale3 = input_scale * scales[(m + 3) * num_blocks + b].scale;

            row_sum0 += b_sum0 * c_scale0;
            row_sum1 += b_sum1 * c_scale1;
            row_sum2 += b_sum2 * c_scale2;
            row_sum3 += b_sum3 * c_scale3;
        }

        output[m + 0] = row_sum0;
        output[m + 1] = row_sum1;
        output[m + 2] = row_sum2;
        output[m + 3] = row_sum3;
    }

    //处理M维度标量尾部，如Vocab不能被4整除
    for (; m < vocab_size; ++m) {
        float row_sum = 0.0f;
        const int8_t* w = quant_weights + m * hidden_dim;

        for (uint32_t b = 0; b < num_blocks; ++b) {
            uint32_t k_start = b * 64;
            uint32_t k_end = std::min(k_start + 64, hidden_dim);
            int32_t b_sum = 0;
            
            for (uint32_t kk = k_start; kk < k_end; ++kk) {
                b_sum += dyn_quant_input[kk] * static_cast<int16_t>(w[kk]);
            }
            row_sum += b_sum * (input_scale * scales[m * num_blocks + b].scale);
        }
        output[m] = row_sum;
    }
}

//无分支ArgMax（掩码混合版本）
extern "C" uint32_t BranchlessArgMax_AVX2(const float* logits, uint32_t vocab_size) 
{
    __m256 v_max_val = _mm256_set1_ps(-INFINITY);
    __m256i v_max_idx = _mm256_setzero_si256();
    
    __m256i v_curr_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7); 
    __m256i v_step = _mm256_set1_epi32(8);

    uint32_t i = 0;
    for (; i + 7 < vocab_size; i += 8) {
        __m256 v_val = _mm256_loadu_ps(logits + i);
        
        //生成掩码cmp_mask并利用
        __m256 cmp_mask = _mm256_cmp_ps(v_val, v_max_val, _CMP_GT_OQ);
        
        v_max_val = _mm256_blendv_ps(v_max_val, v_val, cmp_mask);
        v_max_idx = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(v_max_idx), 
            _mm256_castsi256_ps(v_curr_idx), 
            cmp_mask
        ));
        
        v_curr_idx = _mm256_add_epi32(v_curr_idx, v_step);
    }

    alignas(32) float max_vals[8];
    alignas(32) uint32_t max_idxs[8];
    _mm256_store_ps(max_vals, v_max_val);
    _mm256_store_si256((__m256i*)max_idxs, v_max_idx);

    float global_max = max_vals[0];
    uint32_t global_idx = max_idxs[0];
    for (int k = 1; k < 8; ++k) {
        if (max_vals[k] > global_max) {
            global_max = max_vals[k];
            global_idx = max_idxs[k];
        }
    }

    for (; i < vocab_size; ++i) {
        if (logits[i] > global_max) {
            global_max = logits[i];
            global_idx = i;
        }
    }

    return global_idx;
}
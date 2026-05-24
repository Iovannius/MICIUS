#include "AIPrimitives.h"
#include "AIComputKrnl.h"

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdint>

//-------------------------------------------------------------------------
//V3核心运算引擎
//彻底清除流水线气泡、冷启动页表中断及计算域切换惩罚
//-------------------------------------------------------------------------

/**
 *@brief YMM寄存器内8个float的水平求和
 *用于行级延迟规约
 */
inline float _mm256_reduce_add_ps(__m256 v)
{
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 vsum = _mm_add_ps(vlow, vhigh);
    
    __m128 vshuf1 = _mm_movehl_ps(vsum, vsum);
    vsum = _mm_add_ps(vsum, vshuf1);
    __m128 vshuf2 = _mm_shuffle_ps(vsum, vsum, _MM_SHUFFLE(1, 1, 1, 1));
    vsum = _mm_add_ps(vsum, vshuf2);
    
    return _mm_cvtss_f32(vsum);
}

/**
 *@brief V3版Small-GEMV
 *INT16路线及延迟规约
 */
extern "C" void DraftGEMV_INT8_AVX2(
    const int8_t* __restrict quant_weights, 
    const QuantDescriptor* __restrict scales,
    const float* __restrict input, 
    float* __restrict output, 
    int16_t* __restrict workspace_i16,
    uint32_t hidden_dim, 
    uint32_t vocab_size)
{

    //Per-Tensor安全动态量化，改进消除了内存分配开销
    //直接使用外部传入的已锁页物理内存workspace_i16
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
        
        __m128i lo = _mm256_castsi256_si128(i32);
        __m128i hi = _mm256_extracti128_si256(i32, 1);
        __m128i i16 = _mm_packs_epi32(lo, hi);
        
        _mm_storeu_si128((__m128i*)(workspace_i16 + k), i16);
    }
    for (; k < hidden_dim; ++k) {
        workspace_i16[k] = static_cast<int16_t>(input[k] * inv_input_scale);
    }

    uint32_t num_blocks = (hidden_dim + 63) / 64;

    //Block-Wise GEMV展开，改进消除了流水线气泡
    uint32_t m = 0;

    for (; m + 3 < vocab_size; m += 4) {
        //将FP32的累加维持在寄存器向量级别，推迟到最后做水平求和
        __m256 row_v_acc0 = _mm256_setzero_ps();
        __m256 row_v_acc1 = _mm256_setzero_ps();
        __m256 row_v_acc2 = _mm256_setzero_ps();
        __m256 row_v_acc3 = _mm256_setzero_ps();

        float row_s_acc0 = 0.0f, row_s_acc1 = 0.0f, row_s_acc2 = 0.0f, row_s_acc3 = 0.0f;

        for (uint32_t b = 0; b < num_blocks; ++b) {
            uint32_t k_start = b * 64;
            uint32_t k_end = std::min(k_start + 64, hidden_dim);

            __m256i b_acc0 = _mm256_setzero_si256();
            __m256i b_acc1 = _mm256_setzero_si256();
            __m256i b_acc2 = _mm256_setzero_si256();
            __m256i b_acc3 = _mm256_setzero_si256();

            const int8_t* w0 = quant_weights + (m + 0) * hidden_dim;
            const int8_t* w1 = quant_weights + (m + 1) * hidden_dim;
            const int8_t* w2 = quant_weights + (m + 2) * hidden_dim;
            const int8_t* w3 = quant_weights + (m + 3) * hidden_dim;

            uint32_t kk = k_start;
            
            //内层热循环
            for (; kk + 15 < k_end; kk += 16) {
                __m256i x16 = _mm256_loadu_si256((const __m256i*)(workspace_i16 + kk));

                __m256i w16_0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(w0 + kk)));
                __m256i w16_1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(w1 + kk)));
                __m256i w16_2 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(w2 + kk)));
                __m256i w16_3 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(w3 + kk)));

                b_acc0 = _mm256_add_epi32(b_acc0, _mm256_madd_epi16(x16, w16_0));
                b_acc1 = _mm256_add_epi32(b_acc1, _mm256_madd_epi16(x16, w16_1));
                b_acc2 = _mm256_add_epi32(b_acc2, _mm256_madd_epi16(x16, w16_2));
                b_acc3 = _mm256_add_epi32(b_acc3, _mm256_madd_epi16(x16, w16_3));
            }

            int32_t b_tail0 = 0, b_tail1 = 0, b_tail2 = 0, b_tail3 = 0;
            for (; kk < k_end; ++kk) {
                int16_t x_val = workspace_i16[kk];
                b_tail0 += x_val * static_cast<int16_t>(w0[kk]);
                b_tail1 += x_val * static_cast<int16_t>(w1[kk]);
                b_tail2 += x_val * static_cast<int16_t>(w2[kk]);
                b_tail3 += x_val * static_cast<int16_t>(w3[kk]);
            }

            __m256 f_b_acc0 = _mm256_cvtepi32_ps(b_acc0);
            __m256 f_b_acc1 = _mm256_cvtepi32_ps(b_acc1);
            __m256 f_b_acc2 = _mm256_cvtepi32_ps(b_acc2);
            __m256 f_b_acc3 = _mm256_cvtepi32_ps(b_acc3);

            float c_scale0 = input_scale * scales[(m + 0) * num_blocks + b].scale;
            float c_scale1 = input_scale * scales[(m + 1) * num_blocks + b].scale;
            float c_scale2 = input_scale * scales[(m + 2) * num_blocks + b].scale;
            float c_scale3 = input_scale * scales[(m + 3) * num_blocks + b].scale;

            __m256 v_scale0 = _mm256_set1_ps(c_scale0);
            __m256 v_scale1 = _mm256_set1_ps(c_scale1);
            __m256 v_scale2 = _mm256_set1_ps(c_scale2);
            __m256 v_scale3 = _mm256_set1_ps(c_scale3);

            row_v_acc0 = _mm256_fmadd_ps(f_b_acc0, v_scale0, row_v_acc0);
            row_v_acc1 = _mm256_fmadd_ps(f_b_acc1, v_scale1, row_v_acc1);
            row_v_acc2 = _mm256_fmadd_ps(f_b_acc2, v_scale2, row_v_acc2);
            row_v_acc3 = _mm256_fmadd_ps(f_b_acc3, v_scale3, row_v_acc3);

            row_s_acc0 += b_tail0 * c_scale0;
            row_s_acc1 += b_tail1 * c_scale1;
            row_s_acc2 += b_tail2 * c_scale2;
            row_s_acc3 += b_tail3 * c_scale3;
        }

        //行级唯一的一次水平求和
        output[m + 0] = _mm256_reduce_add_ps(row_v_acc0) + row_s_acc0;
        output[m + 1] = _mm256_reduce_add_ps(row_v_acc1) + row_s_acc1;
        output[m + 2] = _mm256_reduce_add_ps(row_v_acc2) + row_s_acc2;
        output[m + 3] = _mm256_reduce_add_ps(row_v_acc3) + row_s_acc3;
    }

    //M维度标量尾部
    for (; m < vocab_size; ++m) {
        float row_sum = 0.0f;
        const int8_t* w = quant_weights + m * hidden_dim;

        for (uint32_t b = 0; b < num_blocks; ++b) {
            uint32_t k_start = b * 64;
            uint32_t k_end = std::min(k_start + 64, hidden_dim);
            int32_t b_sum = 0;
            
            for (uint32_t kk = k_start; kk < k_end; ++kk) {
                b_sum += workspace_i16[kk] * static_cast<int16_t>(w[kk]);
            }
            row_sum += b_sum * (input_scale * scales[m * num_blocks + b].scale);
        }
        output[m] = row_sum;
    }
}

extern "C" uint32_t BranchlessArgMax_AVX2(const float* logits, uint32_t vocab_size) 
{
    __m256 v_max_val = _mm256_set1_ps(-FLT_MAX);
    __m256i v_max_idx = _mm256_setzero_si256();
    
    __m256i v_curr_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7); 
    __m256i v_step = _mm256_set1_epi32(8);

    uint32_t i = 0;
    for (; i + 7 < vocab_size; i += 8) {
        __m256 v_val = _mm256_loadu_ps(logits + i);
        
        //生成浮点比较掩码
        __m256 cmp_mask = _mm256_cmp_ps(v_val, v_max_val, _CMP_GT_OQ);
        
        /*浮点域*/
        //使用_mm256_blendv_ps更新最大值
        v_max_val = _mm256_blendv_ps(v_max_val, v_val, cmp_mask);
        
        /*整型域*/
        //使用_mm256_blendv_epi8规避时钟惩罚
        //cmp_mask中的全1字节完美契合了epi8的字节级混算逻辑
        v_max_idx = _mm256_blendv_epi8(
            v_max_idx, 
            v_curr_idx, 
            _mm256_castps_si256(cmp_mask)
        );
        
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

//v0.3更新
void BranchlessTop3_AVX2(const float* __restrict logits, uint32_t vocab_size, uint32_t out_tokens[3]) 
{
    __m256 v_max1 = _mm256_set1_ps(-FLT_MAX);
    __m256 v_max2 = _mm256_set1_ps(-FLT_MAX);
    __m256 v_max3 = _mm256_set1_ps(-FLT_MAX);

    __m256i v_idx1 = _mm256_setzero_si256();
    __m256i v_idx2 = _mm256_setzero_si256();
    __m256i v_idx3 = _mm256_setzero_si256();
    
    __m256i v_curr_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7); 
    __m256i v_step = _mm256_set1_epi32(8);

    uint32_t i = 0;
    for (; i + 7 < vocab_size; i += 8) {
        __m256 v_val = _mm256_loadu_ps(logits + i);
        
        //3级比较掩码
        __m256 m1 = _mm256_cmp_ps(v_val, v_max1, _CMP_GT_OQ);
        __m256 m2 = _mm256_cmp_ps(v_val, v_max2, _CMP_GT_OQ);
        __m256 m3 = _mm256_cmp_ps(v_val, v_max3, _CMP_GT_OQ);

        //---------------------------------------------------------
        //浮点域级联下推
        //比第一大，则第三=第二，第二=第一，第一=新值
        //---------------------------------------------------------
        //此处根据blendv的掩码特性可以用极为高效的方式逆向组装
        //更新Top-3，比Top-2大 (m2)被Top-2挤占；否则看是否比Top-3大 (m3)
        __m256 new_max3 = _mm256_blendv_ps(v_max3, v_val, m3);
        new_max3 = _mm256_blendv_ps(new_max3, v_max2, m2);

        //更新Top-2，比Top-1大 (m1)被Top-1挤占；否则看是否比Top-2大 (m2)
        __m256 new_max2 = _mm256_blendv_ps(v_max2, v_val, m2);
        new_max2 = _mm256_blendv_ps(new_max2, v_max1, m1);

        //更新Top-1，这次没有额外的挤占逻辑
        __m256 new_max1 = _mm256_blendv_ps(v_max1, v_val, m1);

        v_max3 = new_max3; v_max2 = new_max2; v_max1 = new_max1;

        //---------------------------------------------------------
        //整型域索引
        //与浮点域完全相同的级联下推逻辑
        //---------------------------------------------------------
        __m256i mi1 = _mm256_castps_si256(m1);
        __m256i mi2 = _mm256_castps_si256(m2);
        __m256i mi3 = _mm256_castps_si256(m3);

        __m256i new_idx3 = _mm256_blendv_epi8(v_idx3, v_curr_idx, mi3);
        new_idx3 = _mm256_blendv_epi8(new_idx3, v_idx2, mi2);

        __m256i new_idx2 = _mm256_blendv_epi8(v_idx2, v_curr_idx, mi2);
        new_idx2 = _mm256_blendv_epi8(new_idx2, v_idx1, mi1);

        __m256i new_idx1 = _mm256_blendv_epi8(v_idx1, v_curr_idx, mi1);

        v_idx3 = new_idx3; v_idx2 = new_idx2; v_idx1 = new_idx1;
        v_curr_idx = _mm256_add_epi32(v_curr_idx, v_step);
    }

    //最后的标量排序
    alignas(32) float f_m1[8], f_m2[8], f_m3[8];
    alignas(32) uint32_t i_m1[8], i_m2[8], i_m3[8];
    
    _mm256_store_ps(f_m1, v_max1);
    _mm256_store_si256((__m256i*)i_m1, v_idx1);
    _mm256_store_ps(f_m2, v_max2);
    _mm256_store_si256((__m256i*)i_m2, v_idx2);
    _mm256_store_ps(f_m3, v_max3);
    _mm256_store_si256((__m256i*)i_m3, v_idx3);

    float final_m[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
    uint32_t final_i[3] = {0, 0, 0};

    auto push_val = [&](float v, uint32_t idx) {
        if(v > final_m[2]) {
            if(v > final_m[1]) {
                if(v > final_m[0]) {
                    final_m[2]=final_m[1]; final_i[2]=final_i[1];
                    final_m[1]=final_m[0]; final_i[1]=final_i[0];
                    final_m[0]=v; final_i[0]=idx;
                } else {
                    final_m[2]=final_m[1]; final_i[2]=final_i[1];
                    final_m[1]=v; final_i[1]=idx;
                }
            } else {
                final_m[2]=v; final_i[2]=idx;
            }
        }
    };

    for (int k = 0; k < 8; ++k) {
        push_val(f_m1[k], i_m1[k]);
        push_val(f_m2[k], i_m2[k]);
        push_val(f_m3[k], i_m3[k]);
    }
    
    //处理未对齐的尾部词表
    for (; i < vocab_size; ++i) {
        push_val(logits[i], i);
    }

    out_tokens[0] = final_i[0];
    out_tokens[1] = final_i[1];
    out_tokens[2] = final_i[2];
}
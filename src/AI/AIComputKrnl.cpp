#include "AIPrimitives.h"
#include "AIComputKrnl.h"

#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <cfloat>
#include <cstdint>

//-------------------------------------------------------------------------
//v0.5核心运算引擎
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
 *@brief v0.5版Small-GEMV
 *INT16路线及延迟规约
 */
extern "C" void DraftGEMV_INT8_AVX2(
    const int8_t* __restrict quant_weights,
    const QuantDescriptor* __restrict scales,
    const float* __restrict input,
    float* __restrict output,
    int8_t* __restrict workspace_i8,
    uint32_t hidden_dim,
    uint32_t vocab_size)
{
    
    float max_abs_val = 1e-9f;

    for (uint32_t k = 0; k < hidden_dim; ++k) {
        max_abs_val = std::max(max_abs_val, std::abs(input[k]));
    }
    
    float input_scale = max_abs_val / 127.0f;
    float inv_input_scale = 127.0f / max_abs_val;

    uint32_t k = 0;

    //纯洁的128-bit包络，规避跨Lane的数据错位
    for (; k + 15 < hidden_dim; k += 16) {

        //外部AIEmuMain 已经保证了input和workspace是alignas(64)分配
        //且这里的k是16的倍数，地址绝对满足32字节对齐，直接使用_mm256_load_ps消除隐式的对齐检查开销
        __m256 f0 = _mm256_load_ps(input + k);
        __m256 f1 = _mm256_load_ps(input + k + 8);
        
        __m256i i32_0 = _mm256_cvtps_epi32(_mm256_mul_ps(f0, _mm256_set1_ps(inv_input_scale)));
        __m256i i32_1 = _mm256_cvtps_epi32(_mm256_mul_ps(f1, _mm256_set1_ps(inv_input_scale)));
        
        __m128i lo0 = _mm256_castsi256_si128(i32_0);
        __m128i hi0 = _mm256_extracti128_si256(i32_0, 1);
        __m128i lo1 = _mm256_castsi256_si128(i32_1);
        __m128i hi1 = _mm256_extracti128_si256(i32_1, 1);

        __m128i i16_0 = _mm_packs_epi32(lo0, hi0);
        __m128i i16_1 = _mm_packs_epi32(lo1, hi1);

        __m128i i8 = _mm_packs_epi16(i16_0, i16_1);
        
        //k必然是16的倍数，因而其绝对满足_mm_store_si128指令要求的16字节对齐
        _mm_store_si128((__m128i*)(workspace_i8 + k), i8);
    }
    
    for (; k < hidden_dim; ++k) {
        workspace_i8[k] = static_cast<int8_t>(input[k] * inv_input_scale);
    }

    uint32_t num_blocks = (hidden_dim + 63) / 64;
    uint32_t m = 0;
    
    __m256i ones = _mm256_set1_epi16(1); //常数1
    //用于相邻INT16归并求和

    for (; m + 3 < vocab_size; m += 4) {

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
            
            /*v0.4.1 -> V0.5重构*/
            //最大程度利用AVX2的执行端口，不使用掩码
            for (; kk + 31 < k_end; kk += 32) {

                //L1D主动预取
                //提前4个Cache Line (128 Bytes)将下一轮权重拉入L1掩盖DDR与Ring Bus延迟
                _mm_prefetch((const char*)(w0 + kk + 128), _MM_HINT_T0);
                _mm_prefetch((const char*)(w1 + kk + 128), _MM_HINT_T0);
                _mm_prefetch((const char*)(w2 + kk + 128), _MM_HINT_T0);
                _mm_prefetch((const char*)(w3 + kk + 128), _MM_HINT_T0);

                //阶段A 
                //workspace_i8与quant_weights均是alignas(64)分配且hidden_dim(768)与kk都是32的倍数
                //此处的内存边界绝对安全，LSU可以满负荷对齐加载，同时消除loadu带来的微指令撕裂隐患
                __m256i x_i8 = _mm256_load_si256((const __m256i*)(workspace_i8 + kk));
                __m256i w_i8_0 = _mm256_load_si256((const __m256i*)(w0 + kk));
                __m256i w_i8_1 = _mm256_load_si256((const __m256i*)(w1 + kk));
                __m256i w_i8_2 = _mm256_load_si256((const __m256i*)(w2 + kk));
                __m256i w_i8_3 = _mm256_load_si256((const __m256i*)(w3 + kk));

                //阶段B
                //ALU绝对值与符号并发转移
                __m256i abs_x = _mm256_abs_epi8(x_i8); 
                __m256i signed_w_0 = _mm256_sign_epi8(w_i8_0, x_i8);
                __m256i signed_w_1 = _mm256_sign_epi8(w_i8_1, x_i8);
                __m256i signed_w_2 = _mm256_sign_epi8(w_i8_2, x_i8);
                __m256i signed_w_3 = _mm256_sign_epi8(w_i8_3, x_i8);

                //阶段C
                //并发FMA发射塞满的多路执行端口
                __m256i dot16_0 = _mm256_maddubs_epi16(abs_x, signed_w_0);
                __m256i dot16_1 = _mm256_maddubs_epi16(abs_x, signed_w_1);
                __m256i dot16_2 = _mm256_maddubs_epi16(abs_x, signed_w_2);
                __m256i dot16_3 = _mm256_maddubs_epi16(abs_x, signed_w_3);

                //阶段D
                //归约累加
                b_acc0 = _mm256_add_epi32(b_acc0, _mm256_madd_epi16(dot16_0, ones));
                b_acc1 = _mm256_add_epi32(b_acc1, _mm256_madd_epi16(dot16_1, ones));
                b_acc2 = _mm256_add_epi32(b_acc2, _mm256_madd_epi16(dot16_2, ones));
                b_acc3 = _mm256_add_epi32(b_acc3, _mm256_madd_epi16(dot16_3, ones));
            }

            int32_t b_tail0 = 0, b_tail1 = 0, b_tail2 = 0, b_tail3 = 0;

            for (; kk < k_end; ++kk) {
                
                int8_t x_val = workspace_i8[kk];
                
                b_tail0 += x_val * w0[kk];
                b_tail1 += x_val * w1[kk];
                b_tail2 += x_val * w2[kk];
                b_tail3 += x_val * w3[kk];
            }

            __m256 f_b_acc0 = _mm256_cvtepi32_ps(b_acc0);
            __m256 f_b_acc1 = _mm256_cvtepi32_ps(b_acc1);
            __m256 f_b_acc2 = _mm256_cvtepi32_ps(b_acc2);
            __m256 f_b_acc3 = _mm256_cvtepi32_ps(b_acc3);

            float c_scale0 = input_scale * scales[(m + 0) * num_blocks + b].scale;
            float c_scale1 = input_scale * scales[(m + 1) * num_blocks + b].scale;
            float c_scale2 = input_scale * scales[(m + 2) * num_blocks + b].scale;
            float c_scale3 = input_scale * scales[(m + 3) * num_blocks + b].scale;

            row_v_acc0 = _mm256_fmadd_ps(f_b_acc0, _mm256_set1_ps(c_scale0), row_v_acc0);
            row_v_acc1 = _mm256_fmadd_ps(f_b_acc1, _mm256_set1_ps(c_scale1), row_v_acc1);
            row_v_acc2 = _mm256_fmadd_ps(f_b_acc2, _mm256_set1_ps(c_scale2), row_v_acc2);
            row_v_acc3 = _mm256_fmadd_ps(f_b_acc3, _mm256_set1_ps(c_scale3), row_v_acc3);

            row_s_acc0 += b_tail0 * c_scale0;
            row_s_acc1 += b_tail1 * c_scale1;
            row_s_acc2 += b_tail2 * c_scale2;
            row_s_acc3 += b_tail3 * c_scale3;
        }

        output[m + 0] = _mm256_reduce_add_ps(row_v_acc0) + row_s_acc0;
        output[m + 1] = _mm256_reduce_add_ps(row_v_acc1) + row_s_acc1;
        output[m + 2] = _mm256_reduce_add_ps(row_v_acc2) + row_s_acc2;
        output[m + 3] = _mm256_reduce_add_ps(row_v_acc3) + row_s_acc3;
    }

    //M维度尾部
    for (; m < vocab_size; ++m) {
        float row_sum = 0.0f;
        
        const int8_t* w = quant_weights + m * hidden_dim;
        for (uint32_t b = 0; b < num_blocks; ++b) {
            
            uint32_t k_start = b * 64;
            uint32_t k_end = std::min(k_start + 64, hidden_dim);
            int32_t b_sum = 0;
            for (uint32_t kk = k_start; kk < k_end; ++kk) {
                b_sum += workspace_i8[kk] * w[kk];
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
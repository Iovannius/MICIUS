#include "AIPrimitives.h"
#include "AIComputKrnlInternal.h"

#include <immintrin.h>
#include <algorithm>
#include <cstddef>
#include <cmath>
#include <cfloat>
#include <cstdint>
#include <cstring>
#include <limits>

//-------------------------------------------------------------------------
//v0.5.2核心运算引擎
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
                //此处的内存边界绝对安全，LSU可以满负荷对齐加载，同时消除loadu带来的微指令级隐患
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

            row_s_acc0 += static_cast<float>(b_tail0) * c_scale0;
            row_s_acc1 += static_cast<float>(b_tail1) * c_scale1;
            row_s_acc2 += static_cast<float>(b_tail2) * c_scale2;
            row_s_acc3 += static_cast<float>(b_tail3) * c_scale3;
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
            
            row_sum += static_cast<float>(b_sum) *
                       (input_scale * scales[m * num_blocks + b].scale);
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

extern "C" bool InitRoPECache(RoPEFreqCache* cache, uint32_t seq_len, uint32_t head_dim, float base) 
{
    if (!cache || seq_len == 0 || head_dim == 0 || (head_dim & 1U) != 0U ||
        !std::isfinite(base) || base <= 0.0f) {
        return false;
    }

    cache->cos_table = nullptr;
    cache->sin_table = nullptr;
    cache->max_seq_len = seq_len;
    cache->head_dim = head_dim;

    if (static_cast<size_t>(seq_len) >
        std::numeric_limits<size_t>::max() / static_cast<size_t>(head_dim)) {
        return false;
    }
    const size_t table_elements = static_cast<size_t>(seq_len) * head_dim;
    if (table_elements > std::numeric_limits<size_t>::max() / sizeof(float)) return false;
    const size_t table_bytes = table_elements * sizeof(float);
#ifdef _WIN32
    cache->cos_table = (float*)_aligned_malloc(table_bytes, 64);
    cache->sin_table = (float*)_aligned_malloc(table_bytes, 64);
#else
    if (posix_memalign(reinterpret_cast<void**>(&cache->cos_table), 64, table_bytes) != 0) {
        cache->cos_table = nullptr;
    }
    if (posix_memalign(reinterpret_cast<void**>(&cache->sin_table), 64, table_bytes) != 0) {
        cache->sin_table = nullptr;
    }
#endif

    if (!cache->cos_table || !cache->sin_table) {
#ifdef _WIN32
        if (cache->cos_table) _aligned_free(cache->cos_table);
        if (cache->sin_table) _aligned_free(cache->sin_table);
#else
        if (cache->cos_table) free(cache->cos_table);
        if (cache->sin_table) free(cache->sin_table);
#endif
        cache->cos_table = nullptr;
        cache->sin_table = nullptr;
        cache->max_seq_len = 0;
        cache->head_dim = 0;
        return false;
    }

    //预计算三角函数表
    for (uint32_t pos = 0; pos < seq_len; ++pos) {
        for (uint32_t i = 0; i < head_dim; i += 2) {
            //传统RoPE频率计算公式
            const float theta = static_cast<float>(pos) /
                std::pow(base, static_cast<float>(i) / static_cast<float>(head_dim));
            float cos_val = std::cos(theta);
            float sin_val = std::sin(theta);

            //将c0, c0, c1, c1连续排列避免额外的重组
            uint32_t idx_even = pos * head_dim + i;
            uint32_t idx_odd  = pos * head_dim + i + 1;
            
            cache->cos_table[idx_even] = cos_val;
            cache->cos_table[idx_odd]  = cos_val;
            cache->sin_table[idx_even] = sin_val;
            cache->sin_table[idx_odd]  = sin_val;
        }
    }
    return true;
}

extern "C" void DestroyRoPECache(RoPEFreqCache* cache) 
{
    if (!cache) return;
#ifdef _WIN32
    if (cache->cos_table) _aligned_free(cache->cos_table);
    if (cache->sin_table) _aligned_free(cache->sin_table);
#else
    if (cache->cos_table) free(cache->cos_table);
    if (cache->sin_table) free(cache->sin_table);
#endif
    cache->cos_table = nullptr;
    cache->sin_table = nullptr;
    cache->max_seq_len = 0;
    cache->head_dim = 0;
}

extern "C" void ApplyRoPE_AVX2(float* __restrict q_or_k, const RoPEFreqCache* __restrict cache, uint32_t pos, uint32_t num_heads) 
{
    if (!q_or_k || !cache || !cache->cos_table || !cache->sin_table ||
        cache->max_seq_len == 0 || cache->head_dim == 0 ||
        (cache->head_dim & 1U) != 0U || pos >= cache->max_seq_len) {
        return;
    }
    
    const float* __restrict cos_row = cache->cos_table + pos * cache->head_dim;
    const float* __restrict sin_row = cache->sin_table + pos * cache->head_dim;
    uint32_t head_dim = cache->head_dim;

    //掩码符号: [-0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f, -0.0f, +0.0f]
    /*这其实是实现-x_1和+x_0的关键*/
    __m256i v_neg_mask_i = _mm256_setr_epi32(0x80000000, 0, 0x80000000, 0, 0x80000000, 0, 0x80000000, 0);
    __m256 v_neg_mask = _mm256_castsi256_ps(v_neg_mask_i);

    for (uint32_t h = 0; h < num_heads; ++h) {
        float* head_ptr = q_or_k + h * head_dim;
        
        uint32_t i = 0;
        for (; i + 7 < head_dim; i += 8) {
            __m256 v_x = _mm256_loadu_ps(head_ptr + i);
            __m256 v_cos = _mm256_load_ps(cos_row + i); //保证对齐
            __m256 v_sin = _mm256_load_ps(sin_row + i); //同上

            //相邻元素交叉 [x0, x1, x2, x3...] -> [x1, x0, x3, x2...]
            __m256 v_swap = _mm256_permute_ps(v_x, 0xB1); 
            
            //经过异或取奇位 -> [-x1, x0, -x3, x2...]
            __m256 v_swap_neg = _mm256_xor_ps(v_swap, v_neg_mask);

            //乘加融合运算的 (v_x * v_cos) + (v_swap_neg * v_sin)
            //实际上复合了RoPE的旋转逻辑
            __m256 v_res = _mm256_fmadd_ps(v_swap_neg, v_sin, _mm256_mul_ps(v_x, v_cos));

            _mm256_storeu_ps(head_ptr + i, v_res);
        }
        for (; i + 1 < head_dim; i += 2) {
            const float x0 = head_ptr[i];
            const float x1 = head_ptr[i + 1];
            const float c = cos_row[i];
            const float s = sin_row[i];
            head_ptr[i] = x0 * c - x1 * s;
            head_ptr[i + 1] = x1 * c + x0 * s;
        }
    }
}

extern "C" void RMSNorm_AVX2(float* out, const float* in, const float* weight, uint32_t dim, float eps)
{
    if (!out || !in || !weight || dim == 0) return;
    __m256 v_sum = _mm256_setzero_ps();
    uint32_t i = 0;
    
    //平方和规约
    for (; i + 7 < dim; i += 8) {
        __m256 v_x = _mm256_loadu_ps(in + i);
        v_sum = _mm256_fmadd_ps(v_x, v_x, v_sum);
    }
    float sum = _mm256_reduce_add_ps(v_sum);
    for (; i < dim; ++i) sum += in[i] * in[i];

    float inv_rms = 1.0f / std::sqrt(sum / static_cast<float>(dim) + eps);
    __m256 v_inv_rms = _mm256_set1_ps(inv_rms);

    //缩放与输出
    for (i = 0; i + 7 < dim; i += 8) {
        __m256 v_x = _mm256_loadu_ps(in + i);
        __m256 v_w = _mm256_loadu_ps(weight + i);
        __m256 v_out = _mm256_mul_ps(_mm256_mul_ps(v_x, v_inv_rms), v_w);
        _mm256_storeu_ps(out + i, v_out);
    }
    for (; i < dim; ++i) out[i] = in[i] * inv_rms * weight[i];
}

extern "C" void ForwardQKV_Proj_RoPE_AVX2(
    const int8_t* __restrict qkv_weights,
    const QuantDescriptor* __restrict qkv_scales,
    const float* __restrict hidden_state,
    float* __restrict qkv_out,
    int8_t* __restrict workspace_i8,
    uint32_t hidden_dim,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t pos,
    const RoPEFreqCache* __restrict rope_cache) 
{
    uint32_t q_dim = num_heads * head_dim;
    uint32_t kv_dim = num_kv_heads * head_dim;
    uint32_t total_qkv_dim = q_dim + 2 * kv_dim;

    //一次输入量化
    DraftGEMV_INT8_AVX2(qkv_weights, qkv_scales, hidden_state, qkv_out, workspace_i8, hidden_dim, total_qkv_dim);

    //切片指针映射，在内存空间拼接
    float* q_ptr = qkv_out;
    float* k_ptr = qkv_out + q_dim;
    //float* v_ptr = qkv_out + q_dim + kv_dim; //V不参与RoPE

    //旋转
    ApplyRoPE_AVX2(q_ptr, rope_cache, pos, num_heads);
    ApplyRoPE_AVX2(k_ptr, rope_cache, pos, num_kv_heads);
}

extern "C" void TreeAttention_GQA_AVX2(
    const float* q_vec,
    const float* const* k_ptrs,
    const float* const* v_ptrs,
    float* out_vec,
    float* score_workspace,
    uint32_t seq_len,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim)
{
    if (!q_vec || !k_ptrs || !v_ptrs || !out_vec || !score_workspace ||
        seq_len == 0 || num_heads == 0 || num_kv_heads == 0 || head_dim == 0 ||
        num_heads % num_kv_heads != 0) {
        return;
    }
    float inv_sqrt_d = 1.0f / std::sqrt((float)head_dim);
    uint32_t kv_group_ratio = num_heads / num_kv_heads; //GQA分组比率，如Llama-3通常为4 (32/8)

    for (uint32_t h = 0; h < num_heads; ++h) {
        const float* q_h = q_vec + h * head_dim;
        uint32_t kv_h = h / kv_group_ratio; //映射至共享K/V头

        //阶段A
        //离散内存追踪点积 Q * K^T
        float max_score = -FLT_MAX;
        for (uint32_t i = 0; i < seq_len; ++i) {
            const float* k_i = k_ptrs[i] + kv_h * head_dim; //提取不连续的指针
            
            __m256 v_dot = _mm256_setzero_ps();
            uint32_t d = 0;
            for (; d + 7 < head_dim; d += 8) {
                v_dot = _mm256_fmadd_ps(_mm256_loadu_ps(q_h + d), _mm256_loadu_ps(k_i + d), v_dot);
            }
            float dot = _mm256_reduce_add_ps(v_dot);
            for (; d < head_dim; ++d) dot += q_h[d] * k_i[d];
            
            float score = dot * inv_sqrt_d;
            score_workspace[i] = score;
            if (score > max_score) max_score = score;
        }

        //阶段B
        //获得稳定的Softmax
        float sum_exp = 0.0f;
        for (uint32_t i = 0; i < seq_len; ++i) {
            float e = std::exp(score_workspace[i] - max_score);
            score_workspace[i] = e;
            sum_exp += e;
        }
        float inv_sum = 1.0f / sum_exp;
        
        //阶段C
        //离散内存追踪 Softmax * V
        float* out_h = out_vec + h * head_dim;
        for (uint32_t d = 0; d < head_dim; ++d) out_h[d] = 0.0f;

        for (uint32_t i = 0; i < seq_len; ++i) {
            float prob = score_workspace[i] * inv_sum;
            __m256 v_prob = _mm256_set1_ps(prob);
            const float* v_i = v_ptrs[i] + kv_h * head_dim;
            
            uint32_t d = 0;
            for (; d + 7 < head_dim; d += 8) {
                __m256 v_v = _mm256_loadu_ps(v_i + d);
                __m256 v_out = _mm256_loadu_ps(out_h + d);
                v_out = _mm256_fmadd_ps(v_v, v_prob, v_out);
                _mm256_storeu_ps(out_h + d, v_out);
            }
            for (; d < head_dim; ++d) out_h[d] += v_i[d] * prob;
        }
    }
}

//FastMath多项式近似 exp(-x)
inline __m256 _mm256_fast_exp_ps(__m256 x)
{
    //限制x防止上下溢出
    x = _mm256_max_ps(x, _mm256_set1_ps(-88.3762626647950f));
    x = _mm256_min_ps(x, _mm256_set1_ps(88.3762626647950f));

    //以2为底
    //y = x * log2(e)
    __m256 y = _mm256_mul_ps(x, _mm256_set1_ps(1.4426950408889634f));
    
    //n = round(y)
    __m256i n = _mm256_cvtps_epi32(y);
    
    //取小数部分
    //f = y - n
    __m256 f = _mm256_sub_ps(y, _mm256_cvtepi32_ps(n));

    //泰勒多项式逼近2^f
    __m256 p = _mm256_set1_ps(0.00133389f);
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.00961812f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.05550410f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.24022650f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(0.69314718f));
    p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(1.0f));

    //计算2^n
    __m256i pow2n = _mm256_slli_epi32(_mm256_add_epi32(n, _mm256_set1_epi32(127)), 23);

    //返回2^n * 2^f
    return _mm256_mul_ps(_mm256_castsi256_ps(pow2n), p);
}

extern "C" void SiLU_Mul_AVX2(float* __restrict gate_out, const float* __restrict up_vec, uint32_t dim)
{
    uint32_t i = 0;
    
    __m256 v_one = _mm256_set1_ps(1.0f);
    __m256 v_two = _mm256_set1_ps(2.0f); //取值2.0作为Newton-Raphson迭代的常数
    
    //通过异或符号位(-0.0f = 0x80000000)实现逻辑取反
    __m256i v_neg_mask = _mm256_set1_epi32(0x80000000); 

    //2-Way循环展开
    //每次迭代吞吐16个float，通过V0和V1两条完全独立的数据流
    //乱序执行引擎会动态调度至两条不同的超标量发射管线并发执行
    for (; i + 15 < dim; i += 16) {
        __m256 v_gate0 = _mm256_loadu_ps(gate_out + i);
        __m256 v_gate1 = _mm256_loadu_ps(gate_out + i + 8);
        
        __m256 v_up0 = _mm256_loadu_ps(up_vec + i);
        __m256 v_up1 = _mm256_loadu_ps(up_vec + i + 8);

        __m256 v_neg_gate0 = _mm256_xor_ps(v_gate0, _mm256_castsi256_ps(v_neg_mask));
        __m256 v_neg_gate1 = _mm256_xor_ps(v_gate1, _mm256_castsi256_ps(v_neg_mask));

        //多项式逼近
        __m256 v_exp0 = _mm256_fast_exp_ps(v_neg_gate0);
        __m256 v_exp1 = _mm256_fast_exp_ps(v_neg_gate1);

        __m256 v_denom0 = _mm256_add_ps(v_one, v_exp0);
        __m256 v_denom1 = _mm256_add_ps(v_one, v_exp1);

        //使用硬件查表快速倒数估计与Newton-Raphson迭代精修替代AVX2硬件除法指令
        //后者(_mm256_div_ps)并不是一个完全流水线化的向量指令，造成的流水线气泡会阻塞发射端口造成严重气泡
        //y_new = y_old * (2.0 - x * y_old)
        
        //初始倒数估计（~11位精度）
        __m256 v_rcp0 = _mm256_rcp_ps(v_denom0);
        __m256 v_rcp1 = _mm256_rcp_ps(v_denom1);
        
        //使用负乘加融合指令计算2.0 - x * y_old
        //在这里_mm256_fnmadd_ps(a, b, c)计算的是-(a * b) + c
        __m256 v_nr_step0 = _mm256_fnmadd_ps(v_denom0, v_rcp0, v_two);
        __m256 v_nr_step1 = _mm256_fnmadd_ps(v_denom1, v_rcp1, v_two);

        //得出高精度倒数（~23位精度）
        __m256 v_inv0 = _mm256_mul_ps(v_rcp0, v_nr_step0);
        __m256 v_inv1 = _mm256_mul_ps(v_rcp1, v_nr_step1);

        //gate * (1 / (1 + exp(-gate)))
        __m256 v_silu0 = _mm256_mul_ps(v_gate0, v_inv0);
        __m256 v_silu1 = _mm256_mul_ps(v_gate1, v_inv1);

        //乘法融合Up Projection
        __m256 v_res0 = _mm256_mul_ps(v_silu0, v_up0);
        __m256 v_res1 = _mm256_mul_ps(v_silu1, v_up1);

        //写回释放缓存
        _mm256_storeu_ps(gate_out + i, v_res0);
        _mm256_storeu_ps(gate_out + i + 8, v_res1);
    }

    //处理未被16整除的FFN尾部维度
    for (; i < dim; ++i) {
        float x = gate_out[i];
        float silu_x = x / (1.0f + std::exp(-x));
        gate_out[i] = silu_x * up_vec[i];
    }
}

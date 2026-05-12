#include "AIPrimitives.h"
#include <immintrin.h>
#include <cmath>

//-------------------------------------------------------------------------
//核心数学引擎
//-------------------------------------------------------------------------

//AVX2寄存器内8个float的水平求和（水平求和）
inline float _mm256_reduce_add_ps(__m256 v)
{
    __m128 vlow = _mm256_castps256_ps128(v);
    __m128 vhigh = _mm256_extractf128_ps(v, 1);
    __m128 vsum = _mm_add_ps(vlow, vhigh);
    
    //128-bit内水平相加
    __m128 vshuf1 = _mm_movehl_ps(vsum, vsum);
    vsum = _mm_add_ps(vsum, vshuf1);
    __m128 vshuf2 = _mm_shuffle_ps(vsum, vsum, _MM_SHUFFLE(1, 1, 1, 1));
    vsum = _mm_add_ps(vsum, vshuf2);
    
    return _mm_cvtss_f32(vsum);
}

//算子1：Zen 3+ Small-GEMV，8路输出*4路输入展开
//尽可能压榨L1D带宽与FMA吞吐，实现零缓存溢出
void DraftGEMV_AVX2(const float* __restrict weight_matrix, 
                    const float* __restrict hidden_state, 
                    float* __restrict output, 
                    uint32_t in_dim, 
                    uint32_t out_dim) 
{
    uint32_t i = 0;
    
    //外层循环每次并行处理8个输出，锁定8个YMM寄存器作为8行权重的独立累加器
    for (; i + 7 < out_dim; i += 8) {
        __m256 acc0 = _mm256_setzero_ps();
        __m256 acc1 = _mm256_setzero_ps();
        __m256 acc2 = _mm256_setzero_ps();
        __m256 acc3 = _mm256_setzero_ps();
        __m256 acc4 = _mm256_setzero_ps();
        __m256 acc5 = _mm256_setzero_ps();
        __m256 acc6 = _mm256_setzero_ps();
        __m256 acc7 = _mm256_setzero_ps();

        const float* w0 = weight_matrix + (i + 0) * in_dim;
        const float* w1 = weight_matrix + (i + 1) * in_dim;
        const float* w2 = weight_matrix + (i + 2) * in_dim;
        const float* w3 = weight_matrix + (i + 3) * in_dim;
        const float* w4 = weight_matrix + (i + 4) * in_dim;
        const float* w5 = weight_matrix + (i + 5) * in_dim;
        const float* w6 = weight_matrix + (i + 6) * in_dim;
        const float* w7 = weight_matrix + (i + 7) * in_dim;

        uint32_t j = 0;
        
        //4路展开，每次吃掉32个Hidden State共128 Bytes，此处旨在提高Zen 3的双Load/FMA端口利用率
        //YMM8-YMM11作为输入缓存器 (VMOVAPS)
        for (; j + 31 < in_dim; j += 32) {

            __m256 x0 = _mm256_load_ps(hidden_state + j + 0);
            __m256 x1 = _mm256_load_ps(hidden_state + j + 8);
            __m256 x2 = _mm256_load_ps(hidden_state + j + 16);
            __m256 x3 = _mm256_load_ps(hidden_state + j + 24);

            //权重作为内存操作数直接控制FMA
            //每组8个输入，共4组
            acc0 = _mm256_fmadd_ps(x0, _mm256_load_ps(w0 + j + 0), acc0);
            acc1 = _mm256_fmadd_ps(x0, _mm256_load_ps(w1 + j + 0), acc1);
            acc2 = _mm256_fmadd_ps(x0, _mm256_load_ps(w2 + j + 0), acc2);
            acc3 = _mm256_fmadd_ps(x0, _mm256_load_ps(w3 + j + 0), acc3);
            acc4 = _mm256_fmadd_ps(x0, _mm256_load_ps(w4 + j + 0), acc4);
            acc5 = _mm256_fmadd_ps(x0, _mm256_load_ps(w5 + j + 0), acc5);
            acc6 = _mm256_fmadd_ps(x0, _mm256_load_ps(w6 + j + 0), acc6);
            acc7 = _mm256_fmadd_ps(x0, _mm256_load_ps(w7 + j + 0), acc7);

            acc0 = _mm256_fmadd_ps(x1, _mm256_load_ps(w0 + j + 8), acc0);
            acc1 = _mm256_fmadd_ps(x1, _mm256_load_ps(w1 + j + 8), acc1);
            acc2 = _mm256_fmadd_ps(x1, _mm256_load_ps(w2 + j + 8), acc2);
            acc3 = _mm256_fmadd_ps(x1, _mm256_load_ps(w3 + j + 8), acc3);
            acc4 = _mm256_fmadd_ps(x1, _mm256_load_ps(w4 + j + 8), acc4);
            acc5 = _mm256_fmadd_ps(x1, _mm256_load_ps(w5 + j + 8), acc5);
            acc6 = _mm256_fmadd_ps(x1, _mm256_load_ps(w6 + j + 8), acc6);
            acc7 = _mm256_fmadd_ps(x1, _mm256_load_ps(w7 + j + 8), acc7);

            acc0 = _mm256_fmadd_ps(x2, _mm256_load_ps(w0 + j + 16), acc0);
            acc1 = _mm256_fmadd_ps(x2, _mm256_load_ps(w1 + j + 16), acc1);
            acc2 = _mm256_fmadd_ps(x2, _mm256_load_ps(w2 + j + 16), acc2);
            acc3 = _mm256_fmadd_ps(x2, _mm256_load_ps(w3 + j + 16), acc3);
            acc4 = _mm256_fmadd_ps(x2, _mm256_load_ps(w4 + j + 16), acc4);
            acc5 = _mm256_fmadd_ps(x2, _mm256_load_ps(w5 + j + 16), acc5);
            acc6 = _mm256_fmadd_ps(x2, _mm256_load_ps(w6 + j + 16), acc6);
            acc7 = _mm256_fmadd_ps(x2, _mm256_load_ps(w7 + j + 16), acc7);

            acc0 = _mm256_fmadd_ps(x3, _mm256_load_ps(w0 + j + 24), acc0);
            acc1 = _mm256_fmadd_ps(x3, _mm256_load_ps(w1 + j + 24), acc1);
            acc2 = _mm256_fmadd_ps(x3, _mm256_load_ps(w2 + j + 24), acc2);
            acc3 = _mm256_fmadd_ps(x3, _mm256_load_ps(w3 + j + 24), acc3);
            acc4 = _mm256_fmadd_ps(x3, _mm256_load_ps(w4 + j + 24), acc4);
            acc5 = _mm256_fmadd_ps(x3, _mm256_load_ps(w5 + j + 24), acc5);
            acc6 = _mm256_fmadd_ps(x3, _mm256_load_ps(w6 + j + 24), acc6);
            acc7 = _mm256_fmadd_ps(x3, _mm256_load_ps(w7 + j + 24), acc7);
        }

        //补齐剩余的in_dim边缘数据
        //但通常是128/256对齐维度，此循环因而极少执行
        for (; j < in_dim; ++j) {
            __m256 v_x = _mm256_set1_ps(hidden_state[j]);
            acc0 = _mm256_fmadd_ps(v_x, _mm256_set1_ps(w0[j]), acc0);
            acc1 = _mm256_fmadd_ps(v_x, _mm256_set1_ps(w1[j]), acc1);
            acc2 = _mm256_fmadd_ps(v_x, _mm256_set1_ps(w2[j]), acc2);
            acc3 = _mm256_fmadd_ps(v_x, _mm256_set1_ps(w3[j]), acc3);
            acc4 = _mm256_fmadd_ps(v_x, _mm256_set1_ps(w4[j]), acc4);
            acc5 = _mm256_fmadd_ps(v_x, _mm256_set1_ps(w5[j]), acc5);
            acc6 = _mm256_fmadd_ps(v_x, _mm256_set1_ps(w6[j]), acc6);
            acc7 = _mm256_fmadd_ps(v_x, _mm256_set1_ps(w7[j]), acc7);
        }

        //离开热路径，水平求和并落盘
        output[i + 0] = _mm256_reduce_add_ps(acc0);
        output[i + 1] = _mm256_reduce_add_ps(acc1);
        output[i + 2] = _mm256_reduce_add_ps(acc2);
        output[i + 3] = _mm256_reduce_add_ps(acc3);
        output[i + 4] = _mm256_reduce_add_ps(acc4);
        output[i + 5] = _mm256_reduce_add_ps(acc5);
        output[i + 6] = _mm256_reduce_add_ps(acc6);
        output[i + 7] = _mm256_reduce_add_ps(acc7);
    }

    //处理out_dim不是8的倍数的边缘行
    for (; i < out_dim; ++i) {
        float sum = 0.0f;
        const float* w = weight_matrix + i * in_dim;
        for (uint32_t j = 0; j < in_dim; ++j) {
            sum += w[j] * hidden_state[j];
        }
        output[i] = sum;
    }
}

//算子2：无分支ArgMax
uint32_t BranchlessArgMax_AVX2(const float* logits, uint32_t vocab_size) {
    //初始化最大值与对应索引
    __m256 v_max_val = _mm256_set1_ps(-INFINITY);
    __m256i v_max_idx = _mm256_setzero_si256();
    
    __m256i v_curr_idx = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7); //当前索引追踪
    __m256i v_step = _mm256_set1_epi32(8);

    uint32_t i = 0;
    //并行吃下8个Logits
    for (; i + 7 < vocab_size; i += 8) {
        __m256 v_val = _mm256_loadu_ps(logits + i);
        
        //v_val > v_max_val ? 0xFFFFFFFF : 0x00000000
        __m256 cmp_mask = _mm256_cmp_ps(v_val, v_max_val, _CMP_GT_OQ);
        
        //最大值与索引的掩码替换
        v_max_val = _mm256_blendv_ps(v_max_val, v_val, cmp_mask);
        v_max_idx = _mm256_castps_si256(_mm256_blendv_ps(
            _mm256_castsi256_ps(v_max_idx), 
            _mm256_castsi256_ps(v_curr_idx), 
            cmp_mask
        ));
        
        //索引推进
        v_curr_idx = _mm256_add_epi32(v_curr_idx, v_step);
    }

    //将256位寄存器提取到栈对齐内存进行最终标量竞争
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

    //此处处理vocab_size无法被8整除的尾部
    for (; i < vocab_size; ++i) {
        if (logits[i] > global_max) {
            global_max = logits[i];
            global_idx = i;
        }
    }

    return global_idx;
}
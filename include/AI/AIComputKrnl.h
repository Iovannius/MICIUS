#pragma once

#include "AIPrimitives.h"
#include <cstddef>
#include <cstdint>

// Status returned by the checked Q8_32 x F32 GEMV backend. The optimized
// kernel keeps FP32 activations intact and consumes one QuantDescriptor for
// every 32 weights; it does not use the legacy activation requantization or
// Q8_64 scale contract.
enum class Q8_32_GEMVStatus : uint8_t {
    OK = 0,
    INVALID_ARGUMENT,
    INVALID_MODEL,
    NUMERIC_ERROR
};

Q8_32_GEMVStatus GEMV_Q8_32_F32_AVX2(
    const int8_t* quant_weights,
    const QuantDescriptor* scales,
    const float* input,
    float* output,
    uint32_t input_dim,
    uint32_t output_dim) noexcept;

// Status returned by the order-preserving attention-value accumulator.  The
// optimized path vectorizes across independent head dimensions while keeping
// the time dimension serial and rounding every time-step result back to FP32,
// matching the reference attention state transition.
enum class AttentionValueStatus : uint8_t {
    OK = 0,
    INVALID_ARGUMENT,
    NUMERIC_ERROR
};

AttentionValueStatus AccumulateAttentionValuesF32_AVX2(
    const float* unnormalized_probabilities,
    double probability_sum,
    const float* value_rows,
    size_t value_stride,
    float* output,
    uint32_t sequence_length,
    uint32_t head_dim) noexcept;

//-------------------------------------------------------------------------
//v0.5核心运算引擎
//专为移动端与边缘节点设计的AVX2算子库（无VNNI）
//-------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

/**
 *@brief v0.5版Small-GEMV
 *采用INT16扩展法，纯AVX2指令集实现的高性能安全乘加。
 **更新：去除了内部thread_local隐患，要求外部调用方传入预先在HeteroMemPool中锁定的大页物理内存作为workspace
 *
 *@param quant_weights 已量化的INT8权重矩阵，行优先 (size=vocab_size*hidden_dim)
 *@param scales        权重对应的量化描述符数组 (Block Size = 64)
 *@param input         尚未量化的FP32隐状态，激活值 (size=hidden_dim)
 *@param output        输出的FP32 Logits (size=vocab_size)
 *@param workspace_i8 外部传入的动态量化缓冲区 (size>=hidden_dim*sizeof(int16_t))
 *@param hidden_dim    K维度，输入特征
 *@param vocab_size    M维度，词表大小或输出特征
 */
void DraftGEMV_INT8_AVX2(
    const int8_t* __restrict quant_weights, 
    const QuantDescriptor* __restrict scales,
    const float* __restrict input, 
    float* __restrict output, 
    int8_t* __restrict workspace_i8,
    uint32_t hidden_dim, 
    uint32_t vocab_size
);

/**
 *@brief 无分支ArgMax算子
 *完全摒弃标量分支预测，利用AVX2纯整型掩码Blend指令规避跨域惩罚
 **@param logits     FP32 Logits数组首指针
 *@param vocab_size 词表大小
 *@return uint32_t  最大Logit对应的Token ID
 */
uint32_t BranchlessArgMax_AVX2(
    const float* logits, 
    uint32_t vocab_size
);

/**
 *@brief RoPE频率表预计算缓存池
 *加载模型时初始化将三角函数降维为L1D缓存读取
 */
typedef struct {
    float* cos_table;
    float* sin_table;
    uint32_t max_seq_len;
    uint32_t head_dim;
} RoPEFreqCache;

/**
 *@brief 初始化RoPE缓存表，其通常在InitPipeline或加载权重时调用一次
 *@param cache 外部传入的未初始化指针
 *@param seq_len 模型最大上下文长度
 *@param head_dim 注意力头维度
 *@param base RoPE频率基数，如Llama-3通常为500000.0f，Llama-2为10000.0f
 */
bool InitRoPECache(RoPEFreqCache* cache, uint32_t seq_len, uint32_t head_dim, float base);

/**
 *@brief 释放RoPE缓存表
 */
void DestroyRoPECache(RoPEFreqCache* cache);

/**
 *@brief RoPE核心算子，无分支单周期双路运算
 *利用_mm256_permute_ps和_mm256_xor_ps解码而不是数据拆包
 **@param q_or_k      需要旋转的Query或Key张量首地址
 * @param cache       预计算的频率表
 * @param pos         当前Token所在的序列绝对位置
 * @param num_heads   注意力头数
 */
void ApplyRoPE_AVX2(
    float* __restrict q_or_k, 
    const RoPEFreqCache* __restrict cache, 
    uint32_t pos, 
    uint32_t num_heads
);

/**
 *@brief RMSNorm归一化
 *必须在QKV投影前对隐状态进行处理
 */
void RMSNorm_AVX2(
    float* out, 
    const float* in, 
    const float* weight, 
    uint32_t dim, 
    float eps
);

/**
 *@brief QKV投影与RoPE宏算子
 *假设外部的Loader已将W_q, W_k, W_v拼接好了，共享一次输入量化开销并在计算完毕后原地对Q和K旋转。
 */
void ForwardQKV_Proj_RoPE_AVX2(
    const int8_t* __restrict qkv_weights,
    const QuantDescriptor* __restrict qkv_scales,
    const float* __restrict hidden_state,
    float* __restrict qkv_out, //连续输出[Q || K || V]
    int8_t* __restrict workspace_i8,
    uint32_t hidden_dim,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim,
    uint32_t pos,
    const RoPEFreqCache* __restrict rope_cache
);

/**
 *@brief GQA树状注意力算子
 *由于草稿树的KV槽位不连续，外部通过追溯parent_idx组装出k_ptrs和v_ptrs指针数组传入
 */
void TreeAttention_GQA_AVX2(
    const float* q_vec, //当前节点的Query [num_heads * head_dim]
    const float* const* k_ptrs, //追溯路径上的K指针数组，长度seq_len
    const float* const* v_ptrs, //追溯路径上的V指针数组，长度seq_len
    float* out_vec, //输出的Attention结果
    float* score_workspace, //外部传入的高速对齐缓冲，size >= seq_len
    uint32_t seq_len,
    uint32_t num_heads,
    uint32_t num_kv_heads,
    uint32_t head_dim
);

/**
 *@brief 中间层激活与门控线性算子 (SwiGLU)
 *计算公式：out_gate[i] = ( gate[i] / (1 + exp(-gate[i])) ) * up[i]
 *覆盖gate_out缓冲区节省空间
 */
void SiLU_Mul_AVX2(
    float* __restrict gate_out, 
    const float* __restrict up_vec, 
    uint32_t dim
);

#ifdef __cplusplus
}
#endif

/**
 *@brief 单向无分支Top-3算子 (V0.3核心级联归约)
 *放弃O(3N)的多次扫描，利用AVX2的Blend掩码实现寄存器内级联淘汰
 */
void BranchlessTop3_AVX2(
    const float* __restrict logits, 
    uint32_t vocab_size, 
    uint32_t out_tokens[3]
);

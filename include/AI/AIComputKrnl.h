#pragma once

#include "AIPrimitives.h"
#include <cstdint>

//-------------------------------------------------------------------------
//V3核心运算引擎
//专为移动端与边缘节点设计的AVX2算子库（无VNNI）
//-------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

/**
 *@brief V3版Small-GEMV
 *采用INT16扩展法，纯AVX2指令集实现的高性能安全乘加。
 **更新：去除了内部thread_local隐患，要求外部调用方传入预先在HeteroMemPool中锁定的大页物理内存作为workspace
 *
 *@param quant_weights 已量化的INT8权重矩阵，行优先 (size=vocab_size*hidden_dim)
 *@param scales        权重对应的量化描述符数组 (Block Size = 64)
 *@param input         尚未量化的FP32隐状态，激活值 (size=hidden_dim)
 *@param output        输出的FP32 Logits (size=vocab_size)
 *@param workspace_i16 外部传入的动态量化缓冲区 (size>=hidden_dim*sizeof(int16_t))
 *@param hidden_dim    K维度，输入特征
 *@param vocab_size    M维度，词表大小或输出特征
 */
void DraftGEMV_INT8_AVX2(
    const int8_t* __restrict quant_weights, 
    const QuantDescriptor* __restrict scales,
    const float* __restrict input, 
    float* __restrict output, 
    int16_t* __restrict workspace_i16,
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

#ifdef __cplusplus
}
#endif
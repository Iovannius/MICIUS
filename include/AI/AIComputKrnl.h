#pragma once

#include "AIPrimitives.h"
#include <cstdint>

//-------------------------------------------------------------------------
//V2核心数学引擎
//专为移动端与边缘节点设计的无VNNI纯AVX2算子库
//-------------------------------------------------------------------------

#ifdef __cplusplus
extern "C" {
#endif

/**
 *@brief V2版Small-GEMV (INT8 Block-Wise Quantization)
 * INT16扩展法，纯AVX2指令集下实现安全乘加实现，规避通道乱序与算术溢出
 **
 *@param quant_weights 已量化的 INT8 权重矩阵，行优先 (size=vocab_size*hidden_dim)
 *@param scales        权重对应的量化描述符数组 (Block Size = 64)
 *@param input         尚未量化的FP32隐状态，激活值 (size=hidden_dim)
 *@param output        输出FP32 Logits (size=vocab_size)
 *@param hidden_dim    K维度（输入特征维度）
 *@param vocab_size    M维度（词表大小或输出特征维度）
 */
void DraftGEMV_INT8_AVX2(
    const int8_t* __restrict quant_weights, 
    const QuantDescriptor* __restrict scales,
    const float* __restrict input, 
    float* __restrict output, 
    uint32_t hidden_dim, 
    uint32_t vocab_size
);

/**
 *@brief 无分支ArgMax算子
 *停用标量分支预测，利用AVX2掩码与Blend在寄存器内寻找最大值
 **
 *@param logits     FP32 Logits数组首指针
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
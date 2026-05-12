#pragma once
#include <cstdint>
#include <immintrin.h>
#include <atomic>

//核心枚举与微型结构体
enum class MemTopology : uint8_t
{
    Host_L1_L2_Locked     = 0,
    Shared_Zero_Copy_LLC  = 1,
    iGPU_VRAM_Exclusive   = 2
};

enum class QuantType : uint8_t
{
    FP32      = 0,
    INT8_VNNI = 1,
    FP16      = 2
};

//量化描述符
//用于后续从FP32向INT8平滑迁移
struct alignas(16) QuantDescriptor {
    float scale;
    int32_t zero_point;

    inline QuantDescriptor(float s, int z) 
        : scale(s), zero_point(z) {}
};

//异构投机采样-解码核心协议
//草稿提案（64字节对齐）
//CPU填充，iGPU验证
struct alignas(64) DraftProposal 
{
    static constexpr uint32_t MAX_K_DRAFTS = 7; //7个Token+1个状态标识=8个32bit=32 Bytes，刚好填满半条缓存行
    uint32_t draft_tokens[MAX_K_DRAFTS];        //预测的Token ID
    uint32_t num_valid_drafts;                  //有效的草稿数量
    float    draft_logits[MAX_K_DRAFTS];        //用于iGPU拒绝采样的对应概率分布
    std::atomic<uint32_t> state_flags;          //0: 空闲, 1: CPU正在写入, 2: iGPU读取待命
    
    inline void Reset()
    {
        num_valid_drafts = 0;
        //使用relaxed语义重置，稍后会有显式的release屏障
        state_flags.store(0, std::memory_order_relaxed);
    }

    inline void PushToken(uint32_t token, float logit)
    {
        if (num_valid_drafts < MAX_K_DRAFTS) {
            draft_tokens[num_valid_drafts] = token;
            draft_logits[num_valid_drafts] = logit;
            num_valid_drafts++;
        }
    }
};

//SoAoSoA布局的微Chunk，目前适配YMM寄存器
struct alignas(64) TensorMicroChunk {
    __m256 val[2]; //64B: 填满一条L1D缓存行
    inline void Load(const float* ptr)
    {
        val[0] = _mm256_load_ps(ptr);
        val[1] = _mm256_load_ps(ptr + 8);
    }

    inline void Store(float* ptr)
    {
        _mm256_store_ps(ptr, val[0]);
        _mm256_store_ps(ptr + 8, val[1]);
    }
};
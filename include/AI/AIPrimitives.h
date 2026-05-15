#pragma once
#include <cstdint>
#include <immintrin.h>
#include <atomic>

//核心枚举
enum class MemTopology : uint8_t {
    Host_L1_L2_Locked     = 0,
    Shared_Zero_Copy_LLC  = 1,
    iGPU_VRAM_Exclusive   = 2
};

enum class QuantType : uint8_t {
    FP32      = 0,
    INT8_AVX2 = 1, 
    INT8_VNNI = 2
};

//量化描述符
//内存对齐16字节存放反量化参数
struct alignas(16) QuantDescriptor {
    float scale;
    int32_t zero_point;
    uint32_t padding[2]; //填充8字节，确保在数组中连续排列时不会打破16字节对齐

    inline QuantDescriptor(float s = 1.0f, int z = 0) : scale(s), zero_point(z) {
        padding[0] = 0;
        padding[1] = 0;
    }
};

//V2版草稿提案
//支持多核树状拓扑探索，依然保证64字节对齐以填满单条L1D缓存行
struct alignas(64) DraftProposal {
    static constexpr uint32_t MAX_K_DRAFTS = 7;
    
    uint32_t draft_tokens[MAX_K_DRAFTS];   //预测Token ID (7*4=28B)
    int32_t  parent_indices[MAX_K_DRAFTS]; //父节点索引，-1代表根节点 (7*4=28B)
    
    uint32_t num_valid_drafts;             //有效草稿 (4B)
    std::atomic<uint32_t> state_flags;     //状态机锁 (4B)
    //28+28+4+4=64B，刚好填满一条缓存行
    
    inline void Reset()
    {
        num_valid_drafts = 0;
        state_flags.store(0, std::memory_order_relaxed);
    }

    inline void PushTreeToken(uint32_t token, int32_t parent_idx)
    {
        if (num_valid_drafts < MAX_K_DRAFTS) {
            draft_tokens[num_valid_drafts] = token;
            parent_indices[num_valid_drafts] = parent_idx;
            num_valid_drafts++;
        }
    }
};

//INT8微块布局，每条缓存行64个INT8参数
//两条指令装载，AVX-512下可简单衔接
struct alignas(64) TensorMicroChunkINT8 {
    __m256i val[2]; 
    
    inline void Load(const int8_t* ptr)
    {
        //利用非对齐加载或转换指针强制读取，此处假设底层指针已对齐
        val[0] = _mm256_load_si256((const __m256i*)ptr);
        val[1] = _mm256_load_si256((const __m256i*)(ptr + 32));
    }
};

//保留FP32微块，用于维护不量化的隐藏状态
struct alignas(64) TensorMicroChunk {
    __m256 val[2]; 
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
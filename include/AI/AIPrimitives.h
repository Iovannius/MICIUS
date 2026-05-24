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
    uint32_t padding[2];

    inline QuantDescriptor(float s = 1.0f, int z = 0) : scale(s), zero_point(z)
    {
        padding[0] = 0; padding[1] = 0;
    }
};

class alignas(64) NanoSpinLock
{
    std::atomic_flag locked = ATOMIC_FLAG_INIT;
public:
    inline void lock()
    {
        while (locked.test_and_set(std::memory_order_acquire)) {
            _mm_pause(); 
        }
    }
    inline void unlock()
    {
        locked.clear(std::memory_order_release);
    }
};

//-------------------------------------------------------------------------
//V3草稿提案 (256B爆炸版)
//扩容至30个节点，适配Top-3的三层深度扇出 (1+3+9=13<30)
//-------------------------------------------------------------------------
//v0.3超标量并发提案树升级 (256B)
struct alignas(256) DraftProposal {
    static constexpr uint32_t MAX_K_DRAFTS = 30;
    uint32_t draft_tokens[MAX_K_DRAFTS];   
    int32_t  parent_indices[MAX_K_DRAFTS]; 
    uint32_t slot_mapping[MAX_K_DRAFTS];   //Worker将申请的物理Slot注册于此供HQ进行回收

    std::atomic<uint32_t> num_valid_drafts; //原子化游标，允许多核Worker无锁并发生成树节点
    int32_t  accepted_tree_node_idx;       
    std::atomic<uint32_t> state_flags;     
    
    inline void Reset()
    {
        num_valid_drafts.store(0, std::memory_order_relaxed);
        accepted_tree_node_idx = -1;
        state_flags.store(0, std::memory_order_relaxed);
    }
    
    //运算完成后调用此函数将节点挂载到树上
    inline int32_t SpawnNodeConcurrent(uint32_t token, int32_t parent_idx, uint32_t phys_slot)
    {
        uint32_t current = num_valid_drafts.load(std::memory_order_relaxed);
        while (current < MAX_K_DRAFTS) {
            if (num_valid_drafts.compare_exchange_weak(current, current + 1, std::memory_order_acquire, std::memory_order_relaxed)) {
                //抢占写入安全槽位
                draft_tokens[current] = token;
                parent_indices[current] = parent_idx;
                slot_mapping[current] = phys_slot;
                
                return current;
            }
        }

        return -1; //树已满，修剪叶节点
    }
};

struct alignas(64) TensorMicroChunkINT8 {
    __m256i val[2]; 
    inline void Load(const int8_t* ptr)
    {
        val[0] = _mm256_load_si256((const __m256i*)ptr);
        val[1] = _mm256_load_si256((const __m256i*)(ptr + 32));
    }
};

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
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
//V4草稿提案
//扩容至30个节点，适配Top-3的三层深度扇出 (1+3+9=13<30)
//-------------------------------------------------------------------------
//缓存一致性与真路径升级
struct alignas(256) DraftProposal {

    static constexpr uint32_t MAX_K_DRAFTS = 30;

    uint32_t draft_tokens[MAX_K_DRAFTS];   
    int32_t  parent_indices[MAX_K_DRAFTS]; 
    uint32_t slot_mapping[MAX_K_DRAFTS];

    alignas(64) std::atomic<uint32_t> num_valid_drafts; //仅作为Allocator
    alignas(64) std::atomic<uint32_t> commit_counter;   //真正的数据就绪启动成员
    alignas(64) int32_t accepted_tree_node_idx;
    alignas(64) std::atomic<uint32_t> state_flags;
    alignas(64) std::atomic<int32_t> golden_leaf_idx;
    
    inline void Reset()
    {
        num_valid_drafts.store(0, std::memory_order_relaxed);
        commit_counter.store(0, std::memory_order_relaxed); //同步重置发令成员
        accepted_tree_node_idx = -1;
        golden_leaf_idx.store(0, std::memory_order_relaxed);
        state_flags.store(0, std::memory_order_relaxed);
    }

    inline int32_t SpawnNodeConcurrent(uint32_t token, int32_t parent_idx, uint32_t phys_slot, bool is_golden = false)
    {
        uint32_t current = num_valid_drafts.load(std::memory_order_relaxed);

        while (current < MAX_K_DRAFTS) {
            if (num_valid_drafts.compare_exchange_weak(current, current + 1, std::memory_order_acquire, std::memory_order_relaxed)) {
                
                draft_tokens[current] = token;
                parent_indices[current] = parent_idx;
                slot_mapping[current] = phys_slot;
                
                // 【关键修复】在发令枪扣响前，将真路径标识安全落盘
                if (is_golden) {
                    golden_leaf_idx.store(current, std::memory_order_relaxed);
                }
                
                // Release 屏障，保证上述所有写入（包括 golden_leaf_idx）对 HQ 绝对可见
                commit_counter.fetch_add(1, std::memory_order_release);
                
                return current;
            }
        }
        return -1;
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
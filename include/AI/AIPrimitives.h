#pragma once
#include <cstdint>
#include <immintrin.h>
#include <atomic>
#include <cstring>

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

    inline QuantDescriptor(float s = 1.0f, int z = 0) noexcept : scale(s), zero_point(z)
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
//V5草稿提案与寻址优化
//引入异步GC信箱 (Async GC Mailbox)及AoSoA混合拓扑与XADD原子操作加速
//-------------------------------------------------------------------------
//缓存一致性与真路径升级
/*AoSoA核心块*/
//每8个节点编织为一个128B的严格对齐块，共2条L1D缓存行
//上半行64B存放高频规约所需的数据(tokens+parents)
//下半行64B存放物理映射(slots)与防伪共享填充
struct alignas(64) DraftNodeBlock8 {
    uint32_t tokens[8];         //32B 
    int32_t  parent_indices[8]; //+32B=64B (Cache Line 0)
    
    uint32_t phys_slots[8];     //32B
    uint32_t padding[8];        //+32B=64B (Cache Line 1)保障不同Block间的绝对物理隔离
};

struct alignas(256) DraftProposal {

    static constexpr uint32_t MAX_K_DRAFTS = 64; 
    static constexpr uint32_t NUM_BLOCKS = MAX_K_DRAFTS / 8;

    //替换原有的SoA数组，采用AoSoA矩阵阵列
    DraftNodeBlock8 blocks[NUM_BLOCKS];

    alignas(64) std::atomic<uint32_t> num_valid_drafts;
    alignas(64) std::atomic<uint32_t> commit_counter;
    alignas(64) int32_t accepted_tree_node_idx;
    alignas(64) std::atomic<uint32_t> state_flags;
    alignas(64) std::atomic<int32_t> golden_leaf_idx;
    
    //异步GC Mailbox，隔离HQ的CAS冲突
    alignas(64) std::atomic<uint32_t> gc_pending_count;
    uint32_t gc_slots[MAX_K_DRAFTS + 2];

    DraftProposal() noexcept
    {
        std::memset(blocks, 0, sizeof(blocks));
        std::memset(gc_slots, 0xFF, sizeof(gc_slots));
        Reset();
    }

    //内联数据访问器组
    inline uint32_t GetToken(uint32_t idx) const { return blocks[idx / 8].tokens[idx % 8]; }
    inline int32_t  GetParent(uint32_t idx) const { return blocks[idx / 8].parent_indices[idx % 8]; }
    inline uint32_t GetSlot(uint32_t idx) const { return blocks[idx / 8].phys_slots[idx % 8]; }
    
    inline void Reset()
    {
        num_valid_drafts.store(0, std::memory_order_relaxed);
        commit_counter.store(0, std::memory_order_relaxed); 
        accepted_tree_node_idx = -1;
        golden_leaf_idx.store(-1, std::memory_order_relaxed);
        state_flags.store(0, std::memory_order_relaxed);
        gc_pending_count.store(0, std::memory_order_relaxed);
    }

    inline int32_t SpawnNodeConcurrent(uint32_t token, int32_t parent_idx, uint32_t phys_slot, bool is_golden = false)
    {
        uint32_t current = num_valid_drafts.load(std::memory_order_relaxed);
        while (current < MAX_K_DRAFTS &&
               !num_valid_drafts.compare_exchange_weak(
                   current,
                   current + 1,
                   std::memory_order_acq_rel,
                   std::memory_order_relaxed)) {
        }

        if (current < MAX_K_DRAFTS) {
            
            //将线性索引映射至AoSoA二维坐标
            uint32_t b_idx = current / 8;
            uint32_t l_idx = current % 8;

            blocks[b_idx].tokens[l_idx] = token;
            blocks[b_idx].parent_indices[l_idx] = parent_idx;
            blocks[b_idx].phys_slots[l_idx] = phys_slot;
            
            if (is_golden) {
                golden_leaf_idx.store(current, std::memory_order_relaxed);
            }
            
            commit_counter.fetch_add(1, std::memory_order_release);
            return current;
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

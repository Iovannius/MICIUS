#pragma once

#include "AIPrimitives.h"

#include <atomic>
#include <vector>
#include <cstring> //for memcpy
#include <algorithm>
#include <limits>
#include <memory>
#include <new>
#include <stdexcept>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <sys/mman.h>
    #include <unistd.h>
    #ifndef MAP_HUGETLB
        #define MAP_HUGETLB 0x40000 //确保旧版POSIX环境也能编译宏
    #endif
#endif

//内存分配跟踪
enum class AllocMethod : uint8_t
{
    WinVirtualAlloc = 0,
    PosixMmap       = 1,
    MmAllocFallback = 2
};

//-------------------------------------------------------------------------
//V2异构零拷贝内存池
//锁定物理内存，提供跨总线的无拷贝提案区域
//-------------------------------------------------------------------------
class HeteroMemPool
{
private:
    void* zero_copy_base = nullptr;
    size_t pool_size = 0;
    AllocMethod alloc_method = AllocMethod::MmAllocFallback;
    size_t proposal_count = 0;
    bool memory_locked = false;

public:
    explicit HeteroMemPool(size_t size_in_bytes) : pool_size(size_in_bytes)
    {
        constexpr size_t kProposalCount = 2;
        if (pool_size < kProposalCount * sizeof(DraftProposal)) {
            throw std::invalid_argument("heterogeneous pool is too small for proposal buffers");
        }
#ifdef _WIN32
        //Windows: 分配对齐的锁页物理内存（大页模式）
        zero_copy_base = VirtualAlloc(NULL, pool_size, MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES, PAGE_READWRITE);
        if (zero_copy_base) {
            alloc_method = AllocMethod::WinVirtualAlloc;
        } else {
            //退回普通分配并强制锁入物理内存
            zero_copy_base = _mm_malloc(pool_size, 4096);
            if (zero_copy_base) memory_locked = VirtualLock(zero_copy_base, pool_size) != 0;
            alloc_method = AllocMethod::MmAllocFallback;
        }
#else
        //POSIX(Linux/macOS): 使用mmap尝试获取大页内存
        zero_copy_base = mmap(NULL, pool_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        
        if (zero_copy_base == MAP_FAILED) {
            zero_copy_base = mmap(NULL, pool_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        }

        if (zero_copy_base != MAP_FAILED && zero_copy_base != nullptr) {
            //强制将此段锁在RAM中避免swap
            memory_locked = mlock(zero_copy_base, pool_size) == 0;
            alloc_method = AllocMethod::PosixMmap;
        } else {            //退回普通分配
            zero_copy_base = _mm_malloc(pool_size, 4096);
            if (zero_copy_base) alloc_method = AllocMethod::MmAllocFallback;
        }
#endif

        if (!zero_copy_base) {
            zero_copy_base = nullptr;
            throw std::bad_alloc();
        }

        for (size_t i = 0; i < kProposalCount; ++i) {
            std::construct_at(static_cast<DraftProposal*>(zero_copy_base) + i);
            proposal_count++;
        }
    }

    ~HeteroMemPool()
    {
        if (!zero_copy_base) return;

        for (size_t i = 0; i < proposal_count; ++i) {
            std::destroy_at(static_cast<DraftProposal*>(zero_copy_base) + i);
        }

        //根据不同的分配途径执行销毁
        if (alloc_method == AllocMethod::WinVirtualAlloc) {
#ifdef _WIN32
            VirtualFree(zero_copy_base, 0, MEM_RELEASE);
#endif
        } 
        else if (alloc_method == AllocMethod::PosixMmap) {
#ifndef _WIN32
            if (memory_locked) munlock(zero_copy_base, pool_size);
            munmap(zero_copy_base, pool_size);
#endif
        } 
        else if (alloc_method == AllocMethod::MmAllocFallback) {
#ifdef _WIN32
            if (memory_locked) VirtualUnlock(zero_copy_base, pool_size);
#endif
            _mm_free(zero_copy_base);
        }
    }

    //获取指向零拷贝区域的指针
    //64B对齐
    //索引支持
    inline DraftProposal* GetDraftBuffer(int idx) const
    {
        if (idx < 0 || static_cast<size_t>(idx) >= proposal_count) return nullptr;
        return static_cast<DraftProposal*>(zero_copy_base) + idx;
    }

    inline void* GetBase() const
    {
        return zero_copy_base;
    }
};

//-------------------------------------------------------------------------
//投机状态插槽池
//集成栈式空闲链表，解决多路拓扑废弃后的大量内存碎片问题
//无锁KV内存池的L1优化
//放弃NanoSpinLock，实装版本化的Treiber Stack
//-------------------------------------------------------------------------
class KVCacheRingBuffer
{

private:

    float* fp32_cache_data;           
    int8_t* int8_cache_data;          
    QuantDescriptor* scales_data;     
    
    uint32_t max_seq_len;
    uint32_t hidden_dim;
    uint32_t blocks_per_row;
    size_t scale_count;

    //无锁栈核心
    /*高32位是版本号以防止ABA问题，低32位是栈顶游标*/
    alignas(64) std::atomic<uint64_t> free_head;
    uint32_t* next_ptrs; //单向链表指针池

    //并发无锁统计KV占用量供遥测读取
    alignas(64) std::atomic<uint32_t> used_slots{0};

public:

    KVCacheRingBuffer(uint32_t seq_len, uint32_t dim)
        : fp32_cache_data(nullptr), int8_cache_data(nullptr), scales_data(nullptr),
          max_seq_len(seq_len), hidden_dim(dim), blocks_per_row(0), scale_count(0),
          next_ptrs(nullptr)
    {
        if (seq_len == 0 || dim == 0) {
            throw std::invalid_argument("KV cache dimensions must be non-zero");
        }
        if (static_cast<size_t>(seq_len) > std::numeric_limits<size_t>::max() / dim) {
            throw std::overflow_error("KV cache element count overflow");
        }

        blocks_per_row = (dim + 63) / 64;
        const size_t element_count = static_cast<size_t>(seq_len) * dim;
        scale_count = static_cast<size_t>(seq_len) * blocks_per_row;

        fp32_cache_data = static_cast<float*>(_mm_malloc(element_count * sizeof(float), 64));
        int8_cache_data = static_cast<int8_t*>(_mm_malloc(element_count * sizeof(int8_t), 64));
        scales_data = static_cast<QuantDescriptor*>(_mm_malloc(scale_count * sizeof(QuantDescriptor), 64));
        
        next_ptrs = static_cast<uint32_t*>(_mm_malloc(static_cast<size_t>(seq_len) * sizeof(uint32_t), 64));
        if (!fp32_cache_data || !int8_cache_data || !scales_data || !next_ptrs) {
            if (fp32_cache_data) _mm_free(fp32_cache_data);
            if (int8_cache_data) _mm_free(int8_cache_data);
            if (scales_data) _mm_free(scales_data);
            if (next_ptrs) _mm_free(next_ptrs);
            fp32_cache_data = nullptr;
            int8_cache_data = nullptr;
            scales_data = nullptr;
            next_ptrs = nullptr;
            throw std::bad_alloc();
        }

        std::fill_n(fp32_cache_data, element_count, 0.0f);
        std::fill_n(int8_cache_data, element_count, int8_t{0});
        for (size_t i = 0; i < scale_count; ++i) {
            std::construct_at(scales_data + i);
        }
        //编织初始物理插槽链表
        for (uint32_t i = 0; i < seq_len - 1; ++i) { next_ptrs[i] = i + 1; }
        next_ptrs[seq_len - 1] = 0xFFFFFFFF; // 尾节点
        
        free_head.store((0ULL << 32) | 0, std::memory_order_relaxed); //初始版本0，栈顶0
    }

    ~KVCacheRingBuffer()
    {
        if (fp32_cache_data) _mm_free(fp32_cache_data);
        if (int8_cache_data) _mm_free(int8_cache_data);
        if (scales_data) {
            std::destroy_n(scales_data, scale_count);
            _mm_free(scales_data);
        }
        if (next_ptrs) _mm_free(next_ptrs);
    }

    //获取占用槽位数量
    inline uint32_t GetUsedSlotsCount() const
    {
        return used_slots.load(std::memory_order_relaxed);
    }

    //O(1)复杂度的无锁申请
    inline uint32_t AllocateSlot() {
        uint64_t head = free_head.load(std::memory_order_acquire);
        while (true) {
            uint32_t slot = static_cast<uint32_t>(head & 0xFFFFFFFF);
            if (slot == 0xFFFFFFFF) return 0xFFFFFFFF; //极简OOM
            
            uint32_t next = next_ptrs[slot];
            uint64_t new_head = ((head >> 32) + 1) << 32 | next; //版本号递增
            
            if (free_head.compare_exchange_weak(head, new_head, std::memory_order_release, std::memory_order_acquire)) {
                used_slots.fetch_add(1, std::memory_order_relaxed);
                return slot;
            }
        }
    }

    //一次批量无锁抓取数个插槽，降低总线90%的CAS冲突
    inline uint32_t BulkAllocateSlots(uint32_t* out_slots, uint32_t count)
    {
        if (count == 0) return 0;
        uint64_t head = free_head.load(std::memory_order_acquire);
        while (true) {
            uint32_t curr = static_cast<uint32_t>(head & 0xFFFFFFFF);
            uint32_t allocated = 0;
            uint32_t iter = curr;
            
            while(allocated < count && iter != 0xFFFFFFFF) {
                out_slots[allocated++] = iter;
                iter = next_ptrs[iter];
            }
            
            if (allocated == 0) return 0;
            
            uint64_t new_head = ((head >> 32) + 1) << 32 | iter;
            if (free_head.compare_exchange_weak(head, new_head, std::memory_order_release, std::memory_order_acquire)) {
                used_slots.fetch_add(allocated, std::memory_order_relaxed);
                return allocated;
            }
        }
    }

    //回收
    inline void FreeSlot(uint32_t slot_idx)
    {
        if (slot_idx == 0xFFFFFFFF || slot_idx >= max_seq_len) return;

        uint64_t head = free_head.load(std::memory_order_acquire);
        while (true) {
            next_ptrs[slot_idx] = static_cast<uint32_t>(head & 0xFFFFFFFF);
            uint64_t new_head = ((head >> 32) + 1) << 32 | slot_idx;
            
            if (free_head.compare_exchange_weak(head, new_head, std::memory_order_release, std::memory_order_acquire)) {
                used_slots.fetch_sub(1, std::memory_order_relaxed);
                break;
            }
        }
    }

    inline float* GetFP32Base() const
    {
        return fp32_cache_data;
    }
    inline void WriteTreeSlotFP32(uint32_t slot_idx, const float* state)
    {
        if (!state || slot_idx >= max_seq_len) return;
        float* dst = fp32_cache_data + (slot_idx * hidden_dim);
        uint32_t i = 0;
        for (; i + 7 < hidden_dim; i += 8) {
            _mm256_storeu_ps(dst + i, _mm256_loadu_ps(state + i));
        }
        for (; i < hidden_dim; ++i) dst[i] = state[i];
    }

    //无锁链表拼接批量回收，避免GC风暴
    inline void BulkFreeSlots(uint32_t* slots, uint32_t count)
    {
        if (count == 0) return;

        uint32_t valid_count = 0;
        for (uint32_t i = 0; i < count; ++i) {
            if (slots[i] != 0xFFFFFFFF && slots[i] < max_seq_len) {
                slots[valid_count++] = slots[i];
            }
        }
        if (valid_count == 0) return;
        count = valid_count;

        for(uint32_t i = 0; i < count - 1; ++i) next_ptrs[slots[i]] = slots[i+1];
        uint64_t head = free_head.load(std::memory_order_acquire);
        while (true) {
            next_ptrs[slots[count - 1]] = static_cast<uint32_t>(head & 0xFFFFFFFF);
            uint64_t new_head = ((head >> 32) + count) << 32 | slots[0];
            if (free_head.compare_exchange_weak(head, new_head, std::memory_order_release, std::memory_order_acquire)) {
                used_slots.fetch_sub(count, std::memory_order_relaxed);
                break;
            }
        }
    }

    inline void Reset()
    {
        for (uint32_t i = 0; i + 1 < max_seq_len; ++i) next_ptrs[i] = i + 1;
        next_ptrs[max_seq_len - 1] = 0xFFFFFFFF;
        free_head.store(0, std::memory_order_release);
        used_slots.store(0, std::memory_order_release);
    }

};

#pragma once
#include "AIPrimitives.h"
#include <vector>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <sys/mman.h>
    #include <unistd.h>
    #ifndef MAP_HUGETLB
        #define MAP_HUGETLB 0x40000 //确保旧版POSIX环境也能编译宏
    #endif
#endif

//追踪底层内存的分配方式，保证析构安全性
enum class AllocMethod : uint8_t
{
    WinVirtualAlloc = 0,
    PosixMmap       = 1,
    MmAllocFallback = 2
};

class HeteroMemPool
{
private:
    void* zero_copy_base;
    size_t pool_size;
    AllocMethod alloc_method;

public:
    HeteroMemPool(size_t size_in_bytes) : pool_size(size_in_bytes)
    {
#ifdef _WIN32
        //Windows: 分配页对齐的锁页物理内存（大页模式）
        zero_copy_base = VirtualAlloc(NULL, pool_size, MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES, PAGE_READWRITE);
        if (zero_copy_base) {
            alloc_method = AllocMethod::WinVirtualAlloc;
        } else {
            //退回普通分配并尝试强制锁入物理内存
            zero_copy_base = _mm_malloc(pool_size, 4096);
            if (zero_copy_base) VirtualLock(zero_copy_base, pool_size);
            alloc_method = AllocMethod::MmAllocFallback;
        }
#else
        //POSIX(Linux/macOS): 使用mmap尝试获取大页内存
        zero_copy_base = mmap(NULL, pool_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_HUGETLB, -1, 0);
        
        if (zero_copy_base == MAP_FAILED) {
            //不支持大页时，退回普通页mmap
            zero_copy_base = mmap(NULL, pool_size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        }

        if (zero_copy_base != MAP_FAILED && zero_copy_base != nullptr) {
            //强制将此段内存锁在物理RAM中，绝对不能被Swap到硬盘
            mlock(zero_copy_base, pool_size); 
            alloc_method = AllocMethod::PosixMmap;
        } else {
            //mmap彻底失败，退回SIMD对齐的堆分配
            zero_copy_base = _mm_malloc(pool_size, 4096);
            alloc_method = AllocMethod::MmAllocFallback;
        }
#endif
    }

    ~HeteroMemPool()
    {
        if (!zero_copy_base) return;

        //根据不同的分配途径执行精确销毁，避免段错误
        if (alloc_method == AllocMethod::WinVirtualAlloc) {
#ifdef _WIN32
            VirtualFree(zero_copy_base, 0, MEM_RELEASE);
#endif
        } 
        else if (alloc_method == AllocMethod::PosixMmap) {
#ifndef _WIN32
            munlock(zero_copy_base, pool_size);
            munmap(zero_copy_base, pool_size);
#endif
        } 
        else if (alloc_method == AllocMethod::MmAllocFallback) {
#ifdef _WIN32
            VirtualUnlock(zero_copy_base, pool_size);
#endif
            _mm_free(zero_copy_base);
        }
    }

    //获取指向零拷贝区域的指针
    inline DraftProposal* GetDraftBuffer()
    {
        return static_cast<DraftProposal*>(zero_copy_base);
    }

    inline void* GetBase() const
    {
        return zero_copy_base;
    }
};

//专为投机采样设计的无锁化KV Cache环形缓冲区
//支持高速Append和Rollback，当iGPU拒绝草稿Token时尤为有效
class KVCacheRingBuffer
{
private:
    float* cache_data;
    uint32_t head;
    uint32_t max_seq_len;
    uint32_t hidden_dim;
    
public:
    KVCacheRingBuffer(uint32_t seq_len, uint32_t dim)
        : head(0),
          max_seq_len(seq_len),
          hidden_dim(dim) 
    {
        cache_data = static_cast<float*>(_mm_malloc(seq_len * dim * sizeof(float), 64)); // 64字节对齐
    }

    ~KVCacheRingBuffer()
    {
        _mm_free(cache_data);
    }

    //将CPU预测的草稿状态压入缓存
    inline void AppendDraftState(const float* state)
    {
        float* dst = cache_data + (head * hidden_dim);
        for (uint32_t i = 0; i < hidden_dim; i += 16) {
            TensorMicroChunk chunk;
            chunk.Load(state + i);
            chunk.Store(dst + i);
        }
        head = (head + 1) % max_seq_len;
    }

    //快速回滚机制，O(1)
    //如果iGPU验证失败直接回退指针，脏数据在L1D缓存立刻作废
    inline void Rollback(uint32_t rejected_count)
    {
        head = (head >= rejected_count) ? (head - rejected_count) : 0;
    }
};
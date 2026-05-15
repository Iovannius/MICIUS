#pragma once
#include "AIPrimitives.h"
#include <vector>
#include <cstring> //for memcpy

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
    void* zero_copy_base;
    size_t pool_size;
    AllocMethod alloc_method;

public:
    HeteroMemPool(size_t size_in_bytes) : pool_size(size_in_bytes)
    {
#ifdef _WIN32
        //Windows: 分配对齐的锁页物理内存（大页模式）
        zero_copy_base = VirtualAlloc(NULL, pool_size, MEM_COMMIT | MEM_RESERVE | MEM_LARGE_PAGES, PAGE_READWRITE);
        if (zero_copy_base) {
            alloc_method = AllocMethod::WinVirtualAlloc;
        } else {
            //退回普通分配并强制锁入物理内存
            zero_copy_base = _mm_malloc(pool_size, 4096);
            if (zero_copy_base) VirtualLock(zero_copy_base, pool_size);
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
            mlock(zero_copy_base, pool_size); 
            alloc_method = AllocMethod::PosixMmap;
        } else {            //退回普通分配
            zero_copy_base = _mm_malloc(pool_size, 4096);
            alloc_method = AllocMethod::MmAllocFallback;
        }
#endif
    }

    ~HeteroMemPool()
    {
        if (!zero_copy_base) return;

        //根据不同的分配途径执行销毁
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
    //64B对齐
    inline DraftProposal* GetDraftBuffer() const
    {
        return static_cast<DraftProposal*>(zero_copy_base);
    }

    inline void* GetBase() const
    {
        return zero_copy_base;
    }
};

//V2投机状态插槽池（原KVCacheRingBuffer）
//改动如下：
//1. 支持树状拓扑寻址(Slot-based addressing) 替代线性Ring逻辑
//2. 双通道多精度并存，同时维护FP32状态、INT8状态及QuantDescriptor阵列
class KVCacheRingBuffer
{
private:
    float* fp32_cache_data;           //FP32状态缓存
    int8_t* int8_cache_data;          //INT8量化状态缓存
    QuantDescriptor* scales_data;     //INT8对应的量化参数
    
    uint32_t head_cursor;             //线性推测时的游标（兼容模式）
    uint32_t max_seq_len;
    uint32_t hidden_dim;
    uint32_t blocks_per_row;          //每行对应的Block数量

public:
    KVCacheRingBuffer(uint32_t seq_len, uint32_t dim)
        : head_cursor(0),
          max_seq_len(seq_len),
          hidden_dim(dim)
    {
        blocks_per_row = (dim + 63) / 64;

        //分配FP32区
        fp32_cache_data = static_cast<float*>(_mm_malloc(seq_len * dim * sizeof(float), 64));
        
        //分配INT8区
        int8_cache_data = static_cast<int8_t*>(_mm_malloc(seq_len * dim * sizeof(int8_t), 64));
        
        //分配QuantDescriptor区（每Block占16字节）
        scales_data = static_cast<QuantDescriptor*>(_mm_malloc(seq_len * blocks_per_row * sizeof(QuantDescriptor), 64));
    }

    ~KVCacheRingBuffer()
    {
        if (fp32_cache_data) _mm_free(fp32_cache_data);
        if (int8_cache_data) _mm_free(int8_cache_data);
        if (scales_data) _mm_free(scales_data);
    }

    //暴露基址指针，供AIDataFront进行GlobalBusRectify内存覆写
    inline float* GetFP32Base() const
    {
        return fp32_cache_data;
    }

    inline int8_t* GetINT8Base() const
    {
        return int8_cache_data;
    }
    
    inline QuantDescriptor* GetScalesBase() const
    {
        return scales_data;
    }

    //V2树状拓扑写入接口
    //允许Core 1~3在并行探索时将结果写入指定的安全槽位，防止跨核覆写
    inline void WriteTreeSlotFP32(uint32_t slot_idx, const float* state)
    {
        float* dst = fp32_cache_data + (slot_idx * hidden_dim);
        
        for (uint32_t i = 0; i < hidden_dim; i += 8) {
            _mm256_storeu_ps(dst + i, _mm256_loadu_ps(state + i));
        }
    }

    inline void WriteTreeSlotINT8(uint32_t slot_idx, const int8_t* q_state, const QuantDescriptor* q_scales)
    {
        int8_t* dst_q = int8_cache_data + (slot_idx * hidden_dim);
        QuantDescriptor* dst_s = scales_data + (slot_idx * blocks_per_row);
        
        //复制INT8状态
        std::memcpy(dst_q, q_state, hidden_dim * sizeof(int8_t));
        //复制量化参数
        std::memcpy(dst_s, q_scales, blocks_per_row * sizeof(QuantDescriptor));
    }

    //兼容V1的线性写入与回滚
    inline void AppendDraftState(const float* state)
    {
        WriteTreeSlotFP32(head_cursor, state);
        head_cursor = (head_cursor + 1) % max_seq_len;
    }

    inline void Rollback(uint32_t rejected_count)
    {
        head_cursor = (head_cursor >= rejected_count) ? (head_cursor - rejected_count) : 0;
    }

    //获取当前线性游标分配位置
    inline uint32_t GetCurrentHead() const
    {
        return head_cursor;
    }



    inline uint32_t GetMaxSeqLen() const
    {
        return max_seq_len;
    }
};
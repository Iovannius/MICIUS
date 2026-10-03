#pragma once

#include <memory>
#include <cstdlib>
#include <atomic>
#include <limits>
#include <new>

#ifdef _WIN32
    #include <malloc.h>
    #include <intrin.h>
#else
    #include <immintrin.h>
#endif

//对齐内存分配
template <typename T>
struct AlignedDeleter {
    void operator()(T* ptr) const noexcept
    {
        if (ptr) {
#ifdef _WIN32
            _aligned_free(ptr);
#else
            free(ptr);
#endif
        }
    }
};

template <typename T>
using AlignedArray = std::unique_ptr<T[], AlignedDeleter<T>>;

template <typename T>
inline AlignedArray<T> MakeAlignedArray(size_t count, size_t alignment = 64)
{
    if (count == 0 || count > std::numeric_limits<size_t>::max() / sizeof(T)) {
        throw std::bad_alloc();
    }
    size_t bytes = count * sizeof(T);
    void* ptr = nullptr;
#ifdef _WIN32
    ptr = _aligned_malloc(bytes, alignment);
#else
    if (posix_memalign(&ptr, alignment, bytes) != 0) {
        ptr = nullptr;
    }
#endif
    if (!ptr) {
        throw std::bad_alloc(); 
    }
    return AlignedArray<T>(static_cast<T*>(ptr));
}

//无锁Epoch同步屏障
class EpochSynchronizer
{

private:
    alignas(64) std::atomic<uint64_t> current_epoch{0};

public:
    inline void AdvanceEpoch() noexcept
    {
        current_epoch.fetch_add(1, std::memory_order_release);
    }

    inline void WaitUntilEpoch(uint64_t target_epoch) const noexcept
    {
        while (current_epoch.load(std::memory_order_acquire) < target_epoch) {
            _mm_pause();
        }
    }

    inline uint64_t GetCurrentEpoch() const noexcept
    {
        return current_epoch.load(std::memory_order_relaxed);
    }
};

//推理调度状态机定义组
enum class PipelineState {
    IDLE,
    DRAFTING,     
    VERIFYING,    
    COMMIT_REJECT 
};

//预声明AIEngineFront以便后续访问其队列和上下文
class AIEngineFront;

class SpeculativeScheduler
{
private:
    PipelineState current_state = PipelineState::IDLE;
    EpochSynchronizer sync_barrier;
    
    AIEngineFront* engine_ctx; //弱引用宿主引擎

public:
    SpeculativeScheduler(AIEngineFront* engine) : engine_ctx(engine) {}

    //分离的阶段化逻辑
    void DispatchDraftTasks();
    void DispatchVerificationTasks();
    int  EvaluateTreeAndCommit(); 

    inline EpochSynchronizer& GetBarrier()
    {
        return sync_barrier;
    }

    int PumpPipeline() {
        current_state = PipelineState::DRAFTING;
        DispatchDraftTasks();
        sync_barrier.AdvanceEpoch(); 
        
        current_state = PipelineState::VERIFYING;
        DispatchVerificationTasks();
        sync_barrier.AdvanceEpoch(); 

        current_state = PipelineState::COMMIT_REJECT;
        return EvaluateTreeAndCommit(); 
    }
};

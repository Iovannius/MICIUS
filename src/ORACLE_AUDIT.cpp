#include "ORACLE_AUDIT.h"
#include <thread>
#include <atomic>
#include <random>
#include <immintrin.h>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <pthread.h>
#endif

//封装为独立的实例上下文
struct OracleContext {
    std::thread hardware_thread;
    alignas(64) std::atomic<bool> terminate{false};
    alignas(64) std::atomic<MiciusDraftTree*> pending_tree{nullptr};
    int core_id;
};

static void ORACLE_PinThreadToCore(int cpu_id)
{
    #ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (1ULL << cpu_id));
#else
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

//传入独立的实例指针
static void ORACLE_Routine(OracleContext* octx)
{
    ORACLE_PinThreadToCore(octx->core_id);
    std::mt19937 rng(42);
    int idle_spins = 0;

    while (!octx->terminate.load(std::memory_order_acquire)) {
        
        //使用exchange一次性取出树句柄并瞬间清空邮箱，避免HQ线程过快醒来投递新树被Oracle错误覆写的风险
        MiciusDraftTree* active_tree = octx->pending_tree.exchange(nullptr, std::memory_order_acquire);
        
        if (active_tree != nullptr) {
            idle_spins = 0;
            
            uint32_t valid_count = Micius_GetTreeValidCount(active_tree);
            int32_t winning_idx = -1;
            
            if (valid_count > 0) {
                std::uniform_int_distribution<int> dist(0, valid_count - 1);
                winning_idx = dist(rng);
                if (dist(rng) % 5 == 0) winning_idx = -1; 
            }

            for(int p = 0; p < 100; ++p) _mm_pause();

            //Mailbox已在上方通过exchange清空，此处安全唤醒HQ
            Micius_AcceptTreeNode(active_tree, winning_idx);
            
        } else {
            int power = std::min(idle_spins / 10, 10);
            int pause_count = 1 << power; 
            for (int p = 0; p < pause_count; ++p) _mm_pause();
            idle_spins++;
        }
    }
}

extern "C" {

OracleContext* ORACLE_InitHardwareThread(int target_core_id)
{
    OracleContext* octx = new OracleContext();
    octx->core_id = target_core_id;
    octx->terminate.store(false, std::memory_order_relaxed);
    octx->pending_tree.store(nullptr, std::memory_order_relaxed);
    
    octx->hardware_thread = std::thread(ORACLE_Routine, octx);
    return octx;
}

void ORACLE_ShutdownHardwareThread(OracleContext* octx)
{
    if (!octx) return;
    octx->terminate.store(true, std::memory_order_release);
    if (octx->hardware_thread.joinable()) {
        octx->hardware_thread.join();
    }
    delete octx;
}

void ORACLE_VerificationCallback(MiciusContext* ctx, MiciusDraftTree* tree_handle, void* user_data)
{
    //user_data强转回OracleContext实例
    OracleContext* octx = static_cast<OracleContext*>(user_data);
    octx->pending_tree.store(tree_handle, std::memory_order_release);
}

}
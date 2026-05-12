#include "AICoreEngine.h"

#include <iostream>
#include <vector>
#include <thread>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <sched.h>
    #include <pthread.h>
    #include <fstream>
    #include <string>
    #include <set>
    #include <time.h>
#endif

//外部核心算子声明
extern void DraftGEMV_AVX2(const float* __restrict, const float* __restrict, float* __restrict, uint32_t, uint32_t);
extern uint32_t BranchlessArgMax_AVX2(const float*, uint32_t);

//-------------------------------------------------------------------------
//获取高精度时间戳
//-------------------------------------------------------------------------
inline double AIEngineFront::GetTime() const
{
#ifdef _WIN32
    LARGE_INTEGER count, freq;
    QueryPerformanceCounter(&count); 
    QueryPerformanceFrequency(&freq);
    return (double)count.QuadPart / freq.QuadPart;
#else
    struct timespec ts; 
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

//-------------------------------------------------------------------------
//拓扑探针：剥离超线程以提取物理核心ID
//-------------------------------------------------------------------------
std::vector<int> AIEngineFront::ProbePhysicalCores()
{
    std::vector<int> physical_cores;

#ifdef _WIN32
    DWORD length = 0; 
    GetLogicalProcessorInformation(nullptr, &length);
    if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
        std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(length / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
        if (GetLogicalProcessorInformation(buffer.data(), &length)) {
            for (const auto& info : buffer) {
                if (info.Relationship == RelationProcessorCore) {
                    ULONG_PTR mask = info.ProcessorMask;
                    ULONG_PTR lowest_bit = mask & ((~mask) + 1); //丢弃HT
                    int idx = 0; 
                    while ((lowest_bit & 1) == 0) { lowest_bit >>= 1; idx++; }
                    physical_cores.push_back(idx);
                }
            }
        }
    }
#else
    int num_logical = std::thread::hardware_concurrency();
    std::set<std::string> seen_cores;
    for (int i = 0; i < num_logical; ++i) {
        std::ifstream pkg("/sys/devices/system/cpu/cpu" + std::to_string(i) + "/topology/physical_package_id");
        std::ifstream core("/sys/devices/system/cpu/cpu" + std::to_string(i) + "/topology/core_id");
        if (pkg.is_open() && core.is_open()) {
            std::string p, c; pkg >> p; core >> c;
            if (seen_cores.insert(p + "_" + c).second) physical_cores.push_back(i);
        }
    }
#endif
    return physical_cores;
}

//-------------------------------------------------------------------------
//锁定物理核心
//-------------------------------------------------------------------------
void AIEngineFront::SetThreadAffinity(int cpu_id) {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (1ULL << cpu_id));
#else
    cpu_set_t cpuset; 
    CPU_ZERO(&cpuset); 
    CPU_SET(cpu_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

//-------------------------------------------------------------------------
//构造函数修复：采用传参动态分配，取代硬编码
//-------------------------------------------------------------------------
AIEngineFront::AIEngineFront(size_t pool_size, uint32_t max_seq_len, uint32_t hidden_dim) {
    mem_pool = new HeteroMemPool(pool_size); 
    kv_cache = new KVCacheRingBuffer(max_seq_len, hidden_dim);    
    shared_proposal = mem_pool->GetDraftBuffer();
    
    data_front = new AiDataFront(kv_cache, shared_proposal);

    std::vector<int> phys_cores = ProbePhysicalCores();
    primary_physical_core = phys_cores.empty() ? 0 : phys_cores.back(); 
}

AIEngineFront::~AIEngineFront()
{
    delete data_front;
    delete kv_cache;
    delete mem_pool;
}

//-------------------------------------------------------------------------
//异构投机解码主频管线
//-------------------------------------------------------------------------
void AIEngineFront::RunInferencePipeline(const float* draft_weights, float* current_hidden_state, uint32_t target_tokens)
{
    SetThreadAffinity(primary_physical_core);
    
    uint32_t step = 0;
    double start_time = GetTime();

    while (step < target_tokens) {
        shared_proposal->Reset();
        shared_proposal->state_flags.store(1, std::memory_order_release);
        
        for(int k = 0; k < DraftProposal::MAX_K_DRAFTS; ++k) {
            alignas(64) float logits[32000]; 
            
            DraftGEMV_AVX2(draft_weights, current_hidden_state, logits, 128, 32000);
            uint32_t next_token = BranchlessArgMax_AVX2(logits, 32000);
            
            data_front->PrepareDraftForiGPU(next_token, logits[next_token]);
            kv_cache->AppendDraftState(current_hidden_state);
        }

#ifdef TARGET_ARCH_DGPU_PCIE
        _mm_clflush(&shared_proposal->draft_tokens);
        _mm_clflush(&shared_proposal->draft_logits);
        _mm_mfence(); 
#endif
        
        shared_proposal->state_flags.store(2, std::memory_order_release);

        while(shared_proposal->state_flags.load(std::memory_order_acquire) != 0) { 
            _mm_pause(); 
        } 

        uint32_t accepted_count = shared_proposal->num_valid_drafts; 
        step += accepted_count;
        
        if(accepted_count < DraftProposal::MAX_K_DRAFTS) {
            kv_cache->Rollback(DraftProposal::MAX_K_DRAFTS - accepted_count);
        }
    }
}
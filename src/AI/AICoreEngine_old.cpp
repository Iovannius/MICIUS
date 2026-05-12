#include "AIPrimitives.h"
#include "AIHetMemPool.h"
#include "AiDataFront.h" 

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

//外部张量计算内核声明
extern void DraftGEMV_AVX2(const float* __restrict, const float* __restrict, float* __restrict, uint32_t, uint32_t);
extern uint32_t BranchlessArgMax_AVX2(const float*, uint32_t);

//-------------------------------------------------------------------------
//异构调度主控 (Heterogeneous Dispatcher)
//实现硬件拓扑探测、线程绑核死锁、跨总线握手时序控制、流水线重叠
//-------------------------------------------------------------------------
class AIEngineFront
{

private:
    HeteroMemPool* mem_pool;
    KVCacheRingBuffer* kv_cache;
    AiDataFront* data_front;       //数据整流前端
    DraftProposal* shared_proposal;
    int primary_physical_core;     //锁定的主物理核

    inline double GetTime() const {
#ifdef _WIN32
        LARGE_INTEGER count, freq;
        QueryPerformanceCounter(&count); QueryPerformanceFrequency(&freq);
        return (double)count.QuadPart / freq.QuadPart;
#else
        struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
        return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
    }

    //“拓扑探针”（初版）
    //剥离超线程以提取物理核心ID
    std::vector<int> ProbePhysicalCores()
    {
        std::vector<int> physical_cores;
#ifdef _WIN32
        DWORD length = 0; GetLogicalProcessorInformation(nullptr, &length);
        if (GetLastError() == ERROR_INSUFFICIENT_BUFFER) {
            std::vector<SYSTEM_LOGICAL_PROCESSOR_INFORMATION> buffer(length / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION));
            if (GetLogicalProcessorInformation(buffer.data(), &length)) {
                for (const auto& info : buffer) {
                    if (info.Relationship == RelationProcessorCore) {
                        ULONG_PTR mask = info.ProcessorMask;
                        ULONG_PTR lowest_bit = mask & ((~mask) + 1); //丢弃HT
                        int idx = 0; while ((lowest_bit & 1) == 0) { lowest_bit >>= 1; idx++; }
                        physical_cores.push_back(idx);
                    }
                }
            }
        }
#else
        //Linux端通过sysfs解析物理包与核心映射
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

    void PinToPhysicalCore(int cpu_id)
    {
#ifdef _WIN32
        SetThreadAffinityMask(GetCurrentThread(), (1ULL << cpu_id));
#else
        cpu_set_t cpuset; CPU_ZERO(&cpuset); CPU_SET(cpu_id, &cpuset);
        pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
    }

public:
    AIEngineFront()
    {
        //初始化底层资源
        mem_pool = new HeteroMemPool(1024 * 1024 * 16); //16MB无拷贝区
        kv_cache = new KVCacheRingBuffer(2048, 128);    //环形缓存结构
        shared_proposal = mem_pool->GetDraftBuffer();
        
        //挂载数据整流前端，把将脏活累活全部委托出去
        data_front = new AiDataFront(kv_cache, shared_proposal);

        //获取物理拓扑并锁定最高优先级的独立核心
        std::vector<int> phys_cores = ProbePhysicalCores();
        primary_physical_core = phys_cores.empty() ? 0 : phys_cores.back(); 
    }

    ~AIEngineFront() {
        delete data_front;
        delete kv_cache;
        delete mem_pool;
    }

    //异构投机解码主频管线
    void RunInferencePipeline(const float* draft_weights, float* current_hidden_state, uint32_t target_tokens) {
        //锁定物理核心，进入高性能独占模式
        PinToPhysicalCore(primary_physical_core);
        
        uint32_t step = 0;
        double start_time = GetTime();

        while (step < target_tokens) {
            shared_proposal->Reset();
            shared_proposal->state_flags = 1;
            //草稿的互斥锁，微观上略微损失吞吐量但整体极大挽回确定性，联合其他部分实现高速忙等同步
            
            //CPU草稿流水线
            //由外部内核算子执行计算，数据整流前端装填
            for(int k = 0; k < DraftProposal::MAX_K_DRAFTS; ++k) {
                alignas(64) float logits[32000]; 
                
                DraftGEMV_AVX2(draft_weights, current_hidden_state, logits, 128, 32000);
                uint32_t next_token = BranchlessArgMax_AVX2(logits, 32000);
                
                //交给AiDataFront.cpp决定是否装填并管理KV碎片
                data_front->PrepareDraftForiGPU(next_token, logits[next_token]);
                kv_cache->AppendDraftState(current_hidden_state);
                
                // TODO: Hidden State 前向推进计算
            }

            //PCIe及环形总线无拷贝输出
            _mm_mfence(); //刷出写缓冲区，确保物理内存立即可见
            shared_proposal->state_flags = 2; //Signal: iGPU Execute!

            //纳秒级自旋同步与仲裁
            //暴力无锁等待，循环检查这4字节的物理内存状态
            while(reinterpret_cast<volatile uint32_t*>(&shared_proposal->state_flags)[0] != 0) { 
                _mm_pause();
                //降频功能，防止超线程饿死及功耗过高
            } 

            //分支预测仲裁与回滚机制
            uint32_t accepted_count = shared_proposal->num_valid_drafts; 
            step += accepted_count;
            
            if(accepted_count < DraftProposal::MAX_K_DRAFTS) {
                //预测失败，交由数据前端执行O(1)极速回滚，抹除L1D脏数据
                kv_cache->Rollback(DraftProposal::MAX_K_DRAFTS - accepted_count);
                
                //（暂不实现）如果空洞过大，触发总线级重排
              //data_front->LocalMicroCompactKV(...);
            }
        }

        double elapsed = GetTime() - start_time;
      //std::cout << "Engine Halted. Overall Throughput: " << step / elapsed << " t/s\n";
    }
};
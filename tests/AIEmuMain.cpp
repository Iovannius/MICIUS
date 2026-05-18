#include "AIPrimitives.h"
#include "AICoreEngine.h"

#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <cstring>
#include <chrono>
#include <algorithm>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <pthread.h>
    #include <sched.h>
#endif

//声明给AICoreEngine::WorkerRoutine准备的全局伪参数
const int8_t* g_quant_weights = nullptr;
const QuantDescriptor* g_scales = nullptr;

//-------------------------------------------------------------------------
//跨平台物理绑核，为模拟核显的影子线程准备的辅助函数
//-------------------------------------------------------------------------
void PinThreadToCore(int cpu_id)
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

//-------------------------------------------------------------------------
//Memory Bandwidth Burn 😼
//模拟真实iGPU在验证大模型时对物理内存总线的占用
//-------------------------------------------------------------------------
void Simulate_iGPU_Memory_Burn(size_t mb_to_burn)
{
    if (mb_to_burn == 0) return;
    const size_t bytes = mb_to_burn * 1024 * 1024;
    
    static size_t current_capacity = 0;
    static float* dummy_src = nullptr;
    static float* dummy_dst = nullptr;

    if (bytes > current_capacity) {
        if (dummy_src) _mm_free(dummy_src);
        if (dummy_dst) _mm_free(dummy_dst);
        
        dummy_src = (float*)_mm_malloc(bytes, 64);
        dummy_dst = (float*)_mm_malloc(bytes, 64);
        current_capacity = bytes;
        
        std::memset(dummy_src, 1, bytes);
        std::memset(dummy_dst, 0, bytes);
    }
    
    std::memcpy(dummy_dst, dummy_src, bytes);
}

//-------------------------------------------------------------------------
//影子核显线程
//-------------------------------------------------------------------------
enum class OracleMode { HEAVEN, HELL, REALISTIC };

void ShadowiGPU_Routine(DraftProposal* proposal, int core_id, OracleMode mode)
{
    //强行绑定到离主引擎最远的物理核，触发真实的MESI跨核同步延迟
    PinThreadToCore(core_id);
    
    std::mt19937 rng(42);
    std::poisson_distribution<int> poisson_dist(2.5); 

    while (true) {
        uint32_t current_state = proposal->state_flags.load(std::memory_order_acquire);
        
        if (current_state == 3) {
            break; 
        }

        if (current_state == 2) {
            //模拟验证时的内存总线挤占
            Simulate_iGPU_Memory_Burn(0); // 设为 0 为纯净吞吐测试，可调大测试 MESI 压力
            
            uint32_t accepted = 0;
            switch (mode) {
                case OracleMode::HEAVEN: 
                    accepted = DraftProposal::MAX_K_DRAFTS; 
                    break;
                case OracleMode::HELL:   
                    /*大模型无论多糟，一定会吐出1个绝对正确的Token*/
                    accepted = 1; 
                    break;
                case OracleMode::REALISTIC: 
                    accepted = std::min(static_cast<uint32_t>(poisson_dist(rng)), DraftProposal::MAX_K_DRAFTS);
                    //保底一个Token
                    if (accepted == 0) accepted = 1; 
                    break;
            }

            proposal->num_valid_drafts = accepted;
            proposal->state_flags.store(0, std::memory_order_release);
        } else {
            _mm_pause();
        }
    }
}

//-------------------------------------------------------------------------
//主程序入口
//-------------------------------------------------------------------------
int main()
{
    std::cout << "========================================================\n";
    std::cout << "[Init] Starting V2 Heterogeneous Speculative Decoding Sandbox\n";
    std::cout << "========================================================\n";

    const uint32_t HIDDEN_DIM = 128;
    const uint32_t VOCAB_SIZE = 32000;
    const uint32_t TARGET_TOKENS = 2000;

    std::cout << "[Mem] Allocating INT8 Block-Quantized weights and states...\n";
    
    int8_t* quant_weights = (int8_t*)_mm_malloc(VOCAB_SIZE * HIDDEN_DIM * sizeof(int8_t), 64);
    std::memset(quant_weights, 1, VOCAB_SIZE * HIDDEN_DIM);
    
    uint32_t num_blocks = (HIDDEN_DIM + 63) / 64;
    QuantDescriptor* weight_scales = (QuantDescriptor*)_mm_malloc(VOCAB_SIZE * num_blocks * sizeof(QuantDescriptor), 64);
    for (size_t i = 0; i < VOCAB_SIZE * num_blocks; ++i) {
        weight_scales[i].scale = 0.01f;
        weight_scales[i].zero_point = 0;
    }

    float* current_hidden_state = (float*)_mm_malloc(HIDDEN_DIM * sizeof(float), 64);
    for (size_t i = 0; i < HIDDEN_DIM; ++i) current_hidden_state[i] = 0.1f;

    g_quant_weights = quant_weights;
    g_scales = weight_scales;

    AIEngineFront engine(1024 * 1024 * 16, 2048, HIDDEN_DIM);
    DraftProposal* shared_proposal = engine.GetSharedProposal(); 

    //将影子线程绑在系统最后一个逻辑核上
    int total_logical_cores = std::thread::hardware_concurrency();
    int shadow_core_id = (total_logical_cores > 1) ? (total_logical_cores - 1) : 1;
    
    OracleMode test_mode = OracleMode::REALISTIC;
    
    std::cout << "[Sys] HQ locked on Core 0. Spawning Shadow iGPU on Core " << shadow_core_id 
              << " | Mode: " << (int)test_mode << "\n";
    
    std::thread igpu_thread(ShadowiGPU_Routine, shared_proposal, shadow_core_id, test_mode);

    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::cout << "[Engine] Ignition. Target Tokens: " << TARGET_TOKENS << "\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    engine.RunTreeInferencePipeline(quant_weights, weight_scales, current_hidden_state, TARGET_TOKENS);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    shared_proposal->state_flags.store(3, std::memory_order_release);
    igpu_thread.join();

    _mm_free(quant_weights);
    _mm_free(weight_scales);
    _mm_free(current_hidden_state);

    double tokens_per_sec = TARGET_TOKENS / elapsed.count();
    std::cout << "========================================================\n";
    std::cout << "[Report] Engine Halted.\n";
    std::cout << "[Report] Elapsed Time : " << elapsed.count() << " seconds\n";
    std::cout << "[Report] Throughput   : " << tokens_per_sec << " Tokens/s\n";
    std::cout << "========================================================\n";

    return 0;
}
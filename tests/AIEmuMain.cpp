#include "AIPrimitives.h"
#include "AICoreEngine.h"

#include <iostream>
#include <vector>
#include <thread>
#include <random>
#include <cstring>
#include <chrono>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <pthread.h>
    #include <sched.h>
#endif

const int8_t* g_quant_weights = nullptr;
const QuantDescriptor* g_scales = nullptr;

void PinThreadToCore(int cpu_id) {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (1ULL << cpu_id));
#else
    cpu_set_t cpuset; CPU_ZERO(&cpuset); CPU_SET(cpu_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

void Simulate_iGPU_Memory_Burn(size_t mb_to_burn) {
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

enum class OracleMode { HEAVEN, HELL, REALISTIC };

void ShadowiGPU_Routine(DraftProposal* proposal, int core_id, OracleMode mode) {
    PinThreadToCore(core_id);
    std::mt19937 rng(42);

    while (true) {
        uint32_t current_state = proposal->state_flags.load(std::memory_order_acquire);
        if (current_state == 3) break; 

        if (current_state == 2) {
            Simulate_iGPU_Memory_Burn(0); 
            
            int32_t winning_idx = -1;
            uint32_t valid_count = proposal->num_valid_drafts.load(std::memory_order_acquire);

            if (valid_count > 0) {
                switch (mode) {
                    case OracleMode::HEAVEN:
                        winning_idx = valid_count - 1;
                        break;
                    case OracleMode::HELL:   
                        winning_idx = -1; //ALL GONE
                        break;
                    case OracleMode::REALISTIC: 
                        //在0到valid_count-1之间随机挑节点
                        std::uniform_int_distribution<int> dist(0, valid_count - 1);
                        winning_idx = dist(rng);
                        //20%概率全盘否决，模拟极低置信度
                        if (dist(rng) % 5 == 0) winning_idx = -1;
                        break;
                }
            }

            proposal->accepted_tree_node_idx = winning_idx;
            proposal->state_flags.store(0, std::memory_order_release);
        } else {
            _mm_pause();
        }
    }
}

int main() {
    std::cout << "========================================================\n";
    std::cout << "[Init] V0.3 Superscalar Speculative Fabric Sandbox\n";
    std::cout << "========================================================\n";

    const uint32_t HIDDEN_DIM = 128;
    const uint32_t VOCAB_SIZE = 32000;
    const uint32_t TARGET_TOKENS = 4000; // 提量压测

    std::cout << "[Mem] Allocating INT8 Block-Quantized structures...\n";
    int8_t* quant_weights = (int8_t*)_mm_malloc(VOCAB_SIZE * HIDDEN_DIM * sizeof(int8_t), 64);
    std::memset(quant_weights, 1, VOCAB_SIZE * HIDDEN_DIM);
    
    uint32_t num_blocks = (HIDDEN_DIM + 63) / 64;
    QuantDescriptor* weight_scales = (QuantDescriptor*)_mm_malloc(VOCAB_SIZE * num_blocks * sizeof(QuantDescriptor), 64);
    for (size_t i = 0; i < VOCAB_SIZE * num_blocks; ++i) { weight_scales[i] = QuantDescriptor(0.01f, 0); }

    float* current_hidden_state = (float*)_mm_malloc(HIDDEN_DIM * sizeof(float), 64);
    for (size_t i = 0; i < HIDDEN_DIM; ++i) current_hidden_state[i] = 0.1f;

    g_quant_weights = quant_weights;
    g_scales = weight_scales;

    //动态分配engine以精确控制销毁时机
    AIEngineFront* engine = new AIEngineFront(1024 * 1024 * 16, 2048, HIDDEN_DIM);
    DraftProposal* shared_proposal = engine->GetSharedProposal(); 

    int total_logical_cores = std::thread::hardware_concurrency();
    int shadow_core_id = (total_logical_cores > 1) ? (total_logical_cores - 1) : 1;
    OracleMode test_mode = OracleMode::REALISTIC;
    
    std::cout << "[Sys] HQ locked on Core 0. Shadow iGPU on Core " << shadow_core_id << "\n";
    std::thread igpu_thread(ShadowiGPU_Routine, shared_proposal, shadow_core_id, test_mode);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::cout << "[Engine] Ignition. Target Tokens: " << TARGET_TOKENS << "\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    engine->RunTreeInferencePipeline(quant_weights, weight_scales, current_hidden_state, TARGET_TOKENS);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    //抛出停机并回收影子iGPU审计线程
    shared_proposal->state_flags.store(3, std::memory_order_release);
    igpu_thread.join();

    //在这里显式销毁引擎
    //将触发~AIEngineFront()，将global_terminate置true并join所有Worker
    //确保运算队伍彻底退出，不再访问任何全局资源
    delete engine;

    _mm_free(quant_weights);
    _mm_free(weight_scales);
    _mm_free(current_hidden_state);

    std::cout << "========================================================\n";
    std::cout << "[Report] Engine Halted.\n";
    std::cout << "[Report] Elapsed Time : " << elapsed.count() << " seconds\n";
    std::cout << "[Report] Throughput   : " << (TARGET_TOKENS / elapsed.count()) << " Tokens/s\n";
    std::cout << "========================================================\n";

    return 0;
}
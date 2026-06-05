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

enum class OracleMode
{
    HEAVEN,
    HELL,
    REALISTIC
};

void ShadowiGPU_Routine(DraftProposal* proposals, int core_id, OracleMode mode)
{
    PinThreadToCore(core_id);
    std::mt19937 rng(42);

    int idle_spins = 0;
    
    while (true) {
        
        if (proposals[0].state_flags.load(std::memory_order_acquire) == 3 || proposals[1].state_flags.load(std::memory_order_acquire) == 3) break;

        bool found_work = false;

        for (int i = 0; i < 2; ++i) {
            if (proposals[i].state_flags.load(std::memory_order_acquire) == 2) {

                found_work = true;

                Simulate_iGPU_Memory_Burn(0);
                int32_t winning_idx = -1;

                uint32_t valid_count = proposals[i].commit_counter.load(std::memory_order_acquire);
                
                if (valid_count > 0) {
                    switch (mode) {
                        case OracleMode::HEAVEN: winning_idx = valid_count - 1;
                        break;
                        case OracleMode::HELL: winning_idx = -1;
                        break;
                        case OracleMode::REALISTIC: 
                            std::uniform_int_distribution<int> dist(0, valid_count - 1);
                            winning_idx = dist(rng);
                            if (dist(rng) % 5 == 0) winning_idx = -1;
                            break;
                    }
                }
                proposals[i].accepted_tree_node_idx = winning_idx;
                proposals[i].state_flags.store(0, std::memory_order_release);
            }
        }

        if (!found_work) {
            if (idle_spins < 100) {
                _mm_pause();
            } else {
                std::this_thread::yield(); 
            }
            idle_spins++;
        } else {
            idle_spins = 0;
        }

        _mm_pause();
    }
}

int main()
{
    std::cout << "========================================================\n[Init] V0.4 Superscalar Speculative Fabric Sandbox\n========================================================\n";
    const uint32_t HIDDEN_DIM = 768;
    const uint32_t VOCAB_SIZE = 32000;
    const uint32_t TARGET_TOKENS = 4000;

    std::cout << "[Mem] Allocating INT8 Block-Quantized structures...\n";
    int8_t* quant_weights = (int8_t*)_mm_malloc(VOCAB_SIZE * HIDDEN_DIM * sizeof(int8_t), 64);
    std::memset(quant_weights, 1, VOCAB_SIZE * HIDDEN_DIM);
    uint32_t num_blocks = (HIDDEN_DIM + 63) / 64;
    QuantDescriptor* weight_scales = (QuantDescriptor*)_mm_malloc(VOCAB_SIZE * num_blocks * sizeof(QuantDescriptor), 64);
    for (size_t i = 0; i < VOCAB_SIZE * num_blocks; ++i) weight_scales[i] = QuantDescriptor(0.01f, 0);

    float* current_hidden_state = (float*)_mm_malloc(HIDDEN_DIM * sizeof(float), 64);
    for (size_t i = 0; i < HIDDEN_DIM; ++i) current_hidden_state[i] = 0.1f;

    g_quant_weights = quant_weights;
    g_scales = weight_scales;

    AIEngineFront* engine = new AIEngineFront(1024 * 1024 * 16, 2048, HIDDEN_DIM);
    DraftProposal* shared_proposal_arr = engine->GetSharedProposal(0);

    int total_logical_cores = std::thread::hardware_concurrency();
    //沙箱中将影子iGPU绑定到逻辑核1，与HQ共享物理核心0
    int shadow_core_id = (total_logical_cores > 1) ? 1 : 0;
    
    std::cout << "[Sys] HQ locked on Core 0. Shadow iGPU on Core " << shadow_core_id << "\n";
    std::thread igpu_thread(ShadowiGPU_Routine, shared_proposal_arr, shadow_core_id, OracleMode::REALISTIC);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));

    std::cout << "[Engine] Ignition. Target Tokens: " << TARGET_TOKENS << "\n";
    auto start_time = std::chrono::high_resolution_clock::now();
    engine->RunTreeInferencePipeline(quant_weights, weight_scales, current_hidden_state, TARGET_TOKENS);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    engine->GetSharedProposal(0)->state_flags.store(3, std::memory_order_release);
    engine->GetSharedProposal(1)->state_flags.store(3, std::memory_order_release);
    igpu_thread.join();
    
    delete engine;
    
    _mm_free(quant_weights);
    _mm_free(weight_scales);
    _mm_free(current_hidden_state);

    std::cout << "========================================================\n[Report] Engine Halted.\n[Report] Elapsed Time : " << elapsed.count() << " seconds\n[Report] Throughput   : " << (TARGET_TOKENS / elapsed.count()) << " Tokens/s\n========================================================\n";
    
    return 0;
}
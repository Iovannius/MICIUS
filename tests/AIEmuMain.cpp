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
    //每次验证强制读写大量内存，人为制造总线拥堵
    const size_t bytes = mb_to_burn * 1024 * 1024;
    static float* dummy_src = (float*)_mm_malloc(bytes, 64);
    static float* dummy_dst = (float*)_mm_malloc(bytes, 64);
    
    //强行memcpy，绕过缓存直接污染主存总线
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
    std::poisson_distribution<int> poisson_dist(2.5); //平均接受2.5个Token的仿真

    while (true) {
        //等待CPU（Acquire语义对应CPU的Release）
        uint32_t current_state = proposal->state_flags.load(std::memory_order_acquire);
        
        //检测停机退出
        if (current_state == 3) {
            break; 
        }

        if (current_state == 2) {
            //iGPU验证开始
            
            //模拟1：执行高压内存负载
            //假设LLM跑一轮需要读取10MB权重
            Simulate_iGPU_Memory_Burn(0); 
            
            //模拟2：判决 
            //决定接受多少个Token
            uint32_t accepted = 0;
            switch (mode) {
                case OracleMode::HEAVEN: 
                    //天堂：全中，测试纯净的FMA GEMV极限吞吐
                    accepted = DraftProposal::MAX_K_DRAFTS; 
                    break;
                case OracleMode::HELL:   
                    //地狱：只接受1个，疯狂触发AiDataFront中的KV回滚和碎片管理
                    accepted = 1; 
                    break;
                case OracleMode::REALISTIC: 
                    //人间：泊松分布，模拟真实自然语言概率
                    accepted = std::min(static_cast<uint32_t>(poisson_dist(rng)), DraftProposal::MAX_K_DRAFTS);
                    //至少保证目标LLM自己能生成1个Token 🥺
                    if (accepted == 0) accepted = 1; 
                    break;
            }

            //写入判决结果
            proposal->num_valid_drafts = accepted;
            
            //验证完成，释放锁，交还给CPU
            proposal->state_flags.store(0, std::memory_order_release);
        } else {
            //防止过度自旋
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
    std::cout << "[Init] Starting Heterogeneous Speculative Decoding Sandbox\n";
    std::cout << "========================================================\n";

    //模型维度定义
    //草稿模型预设隐层128，词表32000
    const uint32_t HIDDEN_DIM = 128;
    const uint32_t VOCAB_SIZE = 32000;
    const uint32_t TARGET_TOKENS = 1000;

    //内存对齐分配
    //模拟模型权重与初始状态
    std::cout << "[Mem] Allocating SoAoSoA aligned weights and states...\n";
    float* draft_weights = (float*)_mm_malloc(VOCAB_SIZE * HIDDEN_DIM * sizeof(float), 64);
    float* current_hidden_state = (float*)_mm_malloc(HIDDEN_DIM * sizeof(float), 64);

    //填充随机数，避免浮点计算触发NaN或Subnormal导致异常掉速
    for (size_t i = 0; i < VOCAB_SIZE * HIDDEN_DIM; ++i) draft_weights[i] = 0.01f;
    for (size_t i = 0; i < HIDDEN_DIM; ++i) current_hidden_state[i] = 0.1f;

    //实例化主引擎
    //16MB零拷贝区，2048单位KV，128维度
    AIEngineFront engine(1024 * 1024 * 16, 2048, HIDDEN_DIM);
    DraftProposal* shared_proposal = engine.GetSharedProposal(); // 假设你在 AIEngineFront 添加了此 Get 方法

    //发射影子线程
    //假设系统有8个物理核(0-7)，会将主引擎绑在7，影子核显绑在0
    int shadow_core_id = 0; 
    OracleMode test_mode = OracleMode::REALISTIC;
    
    std::cout << "[Sys] Spawning Shadow iGPU on Core " << shadow_core_id 
              << " | Mode: " << (int)test_mode << "\n";
    
    std::thread igpu_thread(ShadowiGPU_Routine, shared_proposal, shadow_core_id, test_mode);

    //“点火！”
    //执行推理管线
    std::cout << "[Engine] Ignition. Target Tokens: " << TARGET_TOKENS << "\n";
    auto start_time = std::chrono::high_resolution_clock::now();

    //运行主循环
    //AIEngineFront内核将被锁定在它探测到的物理核心上执行
    engine.RunInferencePipeline(draft_weights, current_hidden_state, TARGET_TOKENS);

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    //停机与清理
    shared_proposal->state_flags.store(3, std::memory_order_release); //发送退出信号3
    igpu_thread.join();

    _mm_free(draft_weights);
    _mm_free(current_hidden_state);

    //完成报告
    double tokens_per_sec = TARGET_TOKENS / elapsed.count();
    std::cout << "========================================================\n";
    std::cout << "[Report] Engine Halted.\n";
    std::cout << "[Report] Elapsed Time : " << elapsed.count() << " seconds\n";
    std::cout << "[Report] Throughput   : " << tokens_per_sec << " Tokens/s\n";
    std::cout << "========================================================\n";

    return 0;
}
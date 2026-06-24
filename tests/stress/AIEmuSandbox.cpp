#include "AIPrimitives.h"

#include "micius_runtime.h"
#include "ORACLE_AUDIT.h"

#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <thread>
#include <atomic>
#include <cassert>
#include <cstring>

#ifdef _WIN32
    #include <windows.h>
    #include <malloc.h>
#else
    #include <stdlib.h>
#endif

//v0.5压力测试沙箱
//专门测算C-ABI接口在多线程竞态、高频拒绝、OOM情况下的抗压能力

void AllocateDummyWeights(uint32_t vocab_size, uint32_t hidden_dim, int8_t** out_w, void** out_s)
{
    size_t w_bytes = vocab_size * hidden_dim * sizeof(int8_t);
    uint32_t num_blocks = (hidden_dim + 63) / 64;
    size_t s_bytes = vocab_size * num_blocks * sizeof(QuantDescriptor);

#ifdef _WIN32
    *out_w = (int8_t*)_aligned_malloc(w_bytes, 64);
    *out_s = _aligned_malloc(s_bytes, 64);
#else
    posix_memalign((void**)out_w, 64, w_bytes);
    posix_memalign(out_s, 64, s_bytes);
#endif

    std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(-5, 5);
    int8_t* w_ptr = *out_w;
    for (size_t i = 0; i < w_bytes; ++i) w_ptr[i] = static_cast<int8_t>(dist(rng));
    QuantDescriptor* s_ptr = static_cast<QuantDescriptor*>(*out_s);
    for (size_t i = 0; i < vocab_size * num_blocks; ++i) { 
        s_ptr[i] = QuantDescriptor(0.05f, 0); 
    }
}

//模拟一个极端的外部大模型状态
void HostileOracleCallback(MiciusContext* ctx, MiciusDraftTree* tree_handle, void* user_data)
{
    //穿透C-ABI获取内部信息进行审计
    uint32_t valid_drafts = Micius_GetTreeValidCount(tree_handle);
    std::atomic<uint32_t>* reject_counter = static_cast<std::atomic<uint32_t>*>(user_data);

    static thread_local std::mt19937 rng(std::random_device{}());
    std::uniform_int_distribution<int> dist(0, 100);

    int chance = dist(rng);
    int32_t winning_idx = -1;

    if (valid_drafts > 0) {
        if (chance < 30) {
            //30%地狱模式
            //全盘拒绝整个草稿树，测试GC与队列清空
            winning_idx = -1;
            reject_counter->fetch_add(1, std::memory_order_relaxed);
        } else if (chance < 60) {
            //30%完美模式，只接受树根Golden Leaf
            winning_idx = valid_drafts - 1; //系统中最新的节点通常在末尾
        } else {
            //40%现实模式，随机接受树中某个分支节点
            std::uniform_int_distribution<int> node_dist(0, valid_drafts - 1);
            winning_idx = node_dist(rng);
        }
    }

    //模拟快速但波动的硬件延迟，约1~5微秒
    int pause_cycles = 100 * (dist(rng) % 5 + 1);
    for(int i = 0; i < pause_cycles; ++i) {
#ifdef _WIN32
        YieldProcessor();
#else
        __asm__ __volatile__("pause");
#endif
    }

    //将决断通过C-API反馈给引擎
    Micius_AcceptTreeNode(tree_handle, winning_idx);
}

int main(int argc, char** argv)
{
    std::cout << "========================================================\n";
    std::cout << "  MICIUS C-ABI Stress Sandbox [Destructive Testing]\n";
    std::cout << "========================================================\n";

    MiciusConfig config = {768, 32000, 256, 4, 0};
    MiciusContext* ctx = Micius_CreateEngine(&config);
    if (!ctx) return -1;
    
    int8_t* dummy_w = nullptr; void* dummy_s = nullptr;
    AllocateDummyWeights(config.vocab_size, config.hidden_dim, &dummy_w, &dummy_s);
    Micius_LoadQuantWeights(ctx, dummy_w, dummy_s);

    //记录拒绝次数
    std::atomic<uint32_t> hostile_rejections{0};

    //沙箱不走分离的ORACLE线程
    //在原地执行同步验证以施加最大主频压力
    Micius_RegisterVerifier(ctx, HostileOracleCallback, &hostile_rejections);

#ifdef _WIN32
    float* prompt_state = (float*)_aligned_malloc(config.hidden_dim * sizeof(float), 64);
#else
    float* prompt_state;
    posix_memalign((void**)&prompt_state, 64, config.hidden_dim * sizeof(float));
#endif
    for(uint32_t i=0; i<config.hidden_dim; ++i) prompt_state[i] = 0.5f;

    uint32_t current_token = 999; 
    uint32_t target_tokens = 5000;
    uint32_t generated_tokens = 0;
    
    std::cout << "[Sandbox] Ignition. KV Pool Size: 64 slots. Target: 5000 Tokens.\n";
    std::cout << "[Sandbox] Expecting brutal GC churn and potential OOMs...\n";
    
    auto start_time = std::chrono::high_resolution_clock::now();
    bool oom_triggered = false;

    //获取Token
    while (generated_tokens < target_tokens) {
        uint32_t out_token = 0;
        
        //这一步将高强度交替触发引擎的Worker阵列、HQ控制面与刚刚注入的HostileOracle
        MiciusStatus status = Micius_Step(ctx, current_token, prompt_state, &out_token);

        if (status == MICIUS_STATE_YIELD_TOKEN) {
            current_token = out_token;
            generated_tokens++;
            if (generated_tokens % 200 == 0) std::cout << "+" << std::flush;
        } 
        else if (status == MICIUS_STATE_TREE_REJECTED) {
            //草稿全毁
            //真实大模型会计算新Token，此处强行变更Token继续压力测试
            current_token = (current_token * 13) % 32000;

            generated_tokens++;
            
            if (hostile_rejections % 50 == 0) std::cout << "!" << std::flush;
        }
        else if (status == MICIUS_STATE_OOM_ERROR) {
            std::cerr << "\n[Sandbox] SUCCESS: Induced OOM correctly caught by ABI firewall at Token " << generated_tokens << "!\n";
            oom_triggered = true;
            break;
        }
    }

    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;

    std::cout << "\n========================================================\n";
    std::cout << "[Report] Survival Status  : " << (oom_triggered ? "Killed by OOM (Expected)" : "Survived") << "\n";
    std::cout << "[Report] Tokens Survived  : " << generated_tokens << "\n";
    std::cout << "[Report] Hostile Rejects  : " << hostile_rejections.load() << " Trees completely destroyed\n";
    std::cout << "[Report] Elapsed Time     : " << elapsed.count() << " seconds\n";
    std::cout << "[Report] Throughput       : " << (generated_tokens / elapsed.count()) << " Tokens/s\n";
    std::cout << "========================================================\n";

    Micius_DestroyEngine(ctx);

#ifdef _WIN32
    _aligned_free(dummy_w);
    _aligned_free(dummy_s);
    _aligned_free(prompt_state);
#else
    free(dummy_w);
    free(dummy_s);
    free(prompt_state);
#endif

    return 0;
}
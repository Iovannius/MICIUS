#include "AICoreEngine.h"
#include "AIComputKrnl.h"

#include <iostream>
#include <cstring>
#include <cmath>
#include <cfloat>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <sched.h>
    #include <pthread.h>
    #include <fstream>
    #include <string>
    #include <set>
#endif

void AIEngineFront::SetThreadAffinity(int cpu_id)
{
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (1ULL << cpu_id));
#else
    cpu_set_t cpuset; CPU_ZERO(&cpuset); CPU_SET(cpu_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

std::vector<int> AIEngineFront::ProbePhysicalCores()
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
                    ULONG_PTR lowest_bit = mask & ((~mask) + 1);
                    int idx = 0; while ((lowest_bit & 1) == 0) { lowest_bit >>= 1; idx++; }
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

AIEngineFront::AIEngineFront(size_t pool_size, uint32_t max_seq_len, uint32_t h_dim)
    : hidden_dim(h_dim)
{
    mem_pool = new HeteroMemPool(pool_size); 
    kv_cache = new KVCacheRingBuffer(max_seq_len, hidden_dim);    
    shared_proposal = mem_pool->GetDraftBuffer();
    data_front = new AiDataFront(kv_cache, shared_proposal);

    std::vector<int> phys_cores = ProbePhysicalCores();
    hq_core_id = phys_cores.empty() ? 0 : phys_cores[0];
    
    for (size_t i = 1; i < phys_cores.size() && i <= 7; ++i) { 
        worker_cores.push_back(phys_cores[i]);
    }

    local_queues = std::vector<WorkStealQueue<FrontierTask, 1024>>(worker_cores.size());
    
    for (size_t i = 0; i < worker_cores.size(); ++i) {
        WorkerContext* ctx = static_cast<WorkerContext*>(_mm_malloc(sizeof(WorkerContext), 64));
        ctx->workspace_i16 = static_cast<int16_t*>(_mm_malloc(hidden_dim * sizeof(int16_t), 64));
        ctx->logits = static_cast<float*>(_mm_malloc(32000 * sizeof(float), 64));
        
        std::memset(ctx->workspace_i16, 0, hidden_dim * sizeof(int16_t));
        std::memset(ctx->logits, 0, 32000 * sizeof(float));

        worker_ctxs.push_back(ctx);
        worker_threads.emplace_back(&AIEngineFront::WorkerRoutine, this, i);
    }
}

AIEngineFront::~AIEngineFront()
{
    global_terminate.store(true, std::memory_order_release);
    for (auto& t : worker_threads) { if (t.joinable()) t.join(); }
    for (auto ctx : worker_ctxs) {
        _mm_free(ctx->workspace_i16);
        _mm_free(ctx->logits);
        _mm_free(ctx);
    }
    delete data_front; delete kv_cache; delete mem_pool;
}

inline void AIEngineFront::ComputeStateTransition(uint32_t token, const float* parent_state, float* new_state, uint32_t dim)
{
    for (uint32_t i = 0; i < dim; ++i) { new_state[i] = parent_state[i] + (token * 0.0001f); }
}

//核心重建 
//-------------------------------------------------------------------------
//AVX2向量化Top-K提取
//彻底消灭标量分支，利用掩码在纯整型/浮点指令域内完成O(3N)级别的扫描
//-------------------------------------------------------------------------
inline void AIEngineFront::ExtractTop3_AVX2(float* logits, uint32_t vocab_size, uint32_t out_tokens[3])
{
    //将复用高度优化的无分支整型域ArgMax算子
    for (int k = 0; k < 3; ++k) {
        uint32_t best_idx = BranchlessArgMax_AVX2(logits, vocab_size);
        out_tokens[k] = best_idx;
        //将其置为负无穷，保证下一趟扫描能够找出次大值
        logits[best_idx] = -FLT_MAX; 
    }
}

//v0.3更新
//重构版运算核心的无锁热路径与向量化降维
void AIEngineFront::WorkerRoutine(int w_idx)
{
    SetThreadAffinity(worker_cores[w_idx]);
    WorkerContext* ctx = worker_ctxs[w_idx];
    extern const int8_t* g_quant_weights;
    extern const QuantDescriptor* g_scales;

    while(!global_terminate.load(std::memory_order_acquire)) {
        FrontierTask task; bool got = local_queues[w_idx].pop_bottom(task); //先吃自己的
        if(!got) { //饿死偷别人的
            for (size_t i=1; i<worker_cores.size(); ++i) {
                if(local_queues[(w_idx+i)%worker_cores.size()].steal_top(task)) {
                    got=true;
                    break;
                }
            }
        }
        if(!got) {
            _mm_pause();
            continue;
        }

        //惰性作废
        //Generation改变或分支失效时扔掉任务，不冲刷流水线
        if(task.gen_id != global_generation.load() || branch_tombstones[task.branch_id].load()) continue;

        float *p_state = kv_cache->GetFP32Base()+(task.parent_slot*hidden_dim), *m_state = kv_cache->GetFP32Base()+(task.alloc_slot*hidden_dim);
        ComputeStateTransition(task.input_token, p_state, m_state, hidden_dim);
        DraftGEMV_INT8_AVX2(g_quant_weights, g_scales, m_state, ctx->logits, ctx->workspace_i16, hidden_dim, 32000);
        
        //双校验
        if(task.gen_id != global_generation.load() || branch_tombstones[task.branch_id].load()) continue;
        kv_cache->WriteTreeSlotFP32(task.alloc_slot, m_state);

        //启用级联向量归约算子
        //排空流水线气泡的同时彻底抛弃标量扫描
        uint32_t top_k[3];
        BranchlessTop3_AVX2(ctx->logits, 32000, top_k);

        for (int k=0; k<3; ++k) {
            uint32_t new_slot = kv_cache->AllocateSlot();
            
            //执行原子化操作抢占共享树索引，同时上报物理Slot至HQ
            int32_t new_tree_idx = shared_proposal->SpawnNodeConcurrent(top_k[k], task.parent_tree_idx, new_slot);
            if(new_tree_idx == -1) {
                kv_cache->FreeSlot(new_slot);
                break; //树已达30节点上限，停止扇出
            }

            FrontierTask child;
            child.gen_id = task.gen_id;
            child.parent_slot = task.alloc_slot;
            child.alloc_slot = new_slot;
            child.input_token = top_k[k];
            child.parent_tree_idx = new_tree_idx;
            child.branch_id = (k==0) ? task.branch_id : new_tree_idx;

            local_queues[w_idx].push_bottom(child);
            /*塞进自己的队列，实际上消除了通信开销！*/
        }
    }
}

//v0.3升级主频管线
//发端、监控与回收
void AIEngineFront::RunTreeInferencePipeline(const int8_t* quant_weights, const QuantDescriptor* weight_scales, float* root_hidden_state, uint32_t target_tokens)
{
    SetThreadAffinity(hq_core_id);
    uint32_t step = 0;
    
    uint32_t root_slot = kv_cache->AllocateSlot();
    kv_cache->WriteTreeSlotFP32(root_slot, root_hidden_state);
    uint32_t accepted_token = 0; 

    while(step < target_tokens) {
        shared_proposal->Reset();
        std::memset((void*)branch_tombstones, 0, sizeof(branch_tombstones));
        uint32_t epoch = global_generation.load();
        uint32_t epoch_base_slot = kv_cache->AllocateSlot();

        //启动根节点并交给0号Worker后待命
        FrontierTask seed; seed.gen_id=epoch; seed.parent_slot=root_slot; seed.alloc_slot=epoch_base_slot;
        seed.input_token=accepted_token; seed.parent_tree_idx=-1; seed.branch_id=0;
        local_queues[0].push_bottom(seed);

        //HQ在休眠状态监控树是否达到30个节点
        while(shared_proposal->num_valid_drafts.load(std::memory_order_acquire) < DraftProposal::MAX_K_DRAFTS) {
            _mm_pause();
        }

        //树的节点满，纪元结束
        //交由iGPU审计
        global_generation.fetch_add(1, std::memory_order_release);
        for(auto& q : local_queues) q.clear();

#ifdef TARGET_ARCH_DGPU_PCIE

        _mm_clflush(&shared_proposal->draft_tokens);
        _mm_clflush(&shared_proposal->parent_indices);
        _mm_clflush(&shared_proposal->slot_mapping);
        _mm_mfence();

#endif

        shared_proposal->state_flags.store(2, std::memory_order_release);
        while(shared_proposal->state_flags.load() != 0) _mm_pause();

        //利用Worker留下的slot_mapping账本进行回收
        int32_t accepted_idx = shared_proposal->accepted_tree_node_idx;
        if(accepted_idx >= 0) {
            uint32_t depth = 0;
            int32_t curr = accepted_idx;
            while(curr != -1) {
                depth++;
                curr = shared_proposal->parent_indices[curr];
            }
            
            kv_cache->FreeSlot(root_slot);
            kv_cache->FreeSlot(epoch_base_slot);
            for(uint32_t i=0; i<shared_proposal->num_valid_drafts; ++i) {
                if(i != (uint32_t)accepted_idx) kv_cache->FreeSlot(shared_proposal->slot_mapping[i]);
            }
            root_slot = shared_proposal->slot_mapping[accepted_idx];
            step += (depth + 1);
        } else {
            for(uint32_t i=0; i<shared_proposal->num_valid_drafts; ++i) kv_cache->FreeSlot(shared_proposal->slot_mapping[i]);
            kv_cache->FreeSlot(root_slot);
            root_slot = epoch_base_slot;
            step += 1;
        }
    }
}
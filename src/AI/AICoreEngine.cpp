#include "AICoreEngine.h"
#include "AIComputKrnl.h"

#include <iostream>
#include <cstring>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <sched.h>
    #include <pthread.h>
    #include <fstream>
    #include <string>
    #include <set>
#endif

// 获取高精度时间戳
inline double AIEngineFront::GetTime() const {
#ifdef _WIN32
    LARGE_INTEGER count, freq;
    QueryPerformanceCounter(&count); QueryPerformanceFrequency(&freq);
    return (double)count.QuadPart / freq.QuadPart;
#else
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
#endif
}

//物理核心探针
std::vector<int> AIEngineFront::ProbePhysicalCores() {
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

//绑核操作
void AIEngineFront::SetThreadAffinity(int cpu_id) {
#ifdef _WIN32
    SetThreadAffinityMask(GetCurrentThread(), (1ULL << cpu_id));
#else
    cpu_set_t cpuset; CPU_ZERO(&cpuset); CPU_SET(cpu_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

//核心初始化与基本部署
AIEngineFront::AIEngineFront(size_t pool_size, uint32_t max_seq_len, uint32_t h_dim)
    : hidden_dim(h_dim)
{
    mem_pool = new HeteroMemPool(pool_size); 
    kv_cache = new KVCacheRingBuffer(max_seq_len, hidden_dim);    
    shared_proposal = mem_pool->GetDraftBuffer();
    data_front = new AiDataFront(kv_cache, shared_proposal);

    std::vector<int> phys_cores = ProbePhysicalCores();
    hq_core_id = phys_cores.empty() ? 0 : phys_cores[0];
    
    global_terminate.store(false, std::memory_order_release);

    //剥离总控，剩下的核全部分配供运算
    for (size_t i = 1; i < phys_cores.size() && i <= 3; ++i) {
        worker_cores.push_back(phys_cores[i]);
    }

    //初始化任务信箱并启动运算线程
    for (size_t i = 0; i < worker_cores.size(); ++i) {
        WorkerTaskBox* box = static_cast<WorkerTaskBox*>(_mm_malloc(sizeof(WorkerTaskBox), 64));
        
        //内存池外置分配
        //为每个Worker分配专属的量化Workspace
        size_t ws_size = hidden_dim * sizeof(int16_t);
        box->workspace_i16 = static_cast<int16_t*>(_mm_malloc(ws_size, 64));
        
        //强制触页
        //写入0以强制OS映射真实物理页，消除首Token延迟尖峰
        std::memset(box->workspace_i16, 0, ws_size);
        
        box->state_flag.store(0, std::memory_order_relaxed);
        task_boxes.push_back(box);
        worker_threads.emplace_back(&AIEngineFront::WorkerRoutine, this, i);
    }
}

AIEngineFront::~AIEngineFront()
{
    global_terminate.store(true, std::memory_order_release);
    for (auto box : task_boxes) {
        box->state_flag.store(1, std::memory_order_release); //唤醒沉睡的线程以退出
    }
    for (auto& t : worker_threads) {
        if (t.joinable()) t.join();
    }
    for (auto box : task_boxes) {
        _mm_free(box->workspace_i16); //释放物理页
        _mm_free(box);
    }
    delete data_front; delete kv_cache; delete mem_pool;
}

// 状态转移引擎 (Transformer/RNN的抽象)
inline void AIEngineFront::ComputeStateTransition(uint32_t token, const float* parent_state, float* new_state, uint32_t dim)
{
    /*真正的模型中，这里会用Embedding(token)与parent_state进行注意力机制或线性变换*/
    for (uint32_t i = 0; i < dim; ++i) {
        new_state[i] = parent_state[i] + (token * 0.0001f);
    }
}

//运算核心的无锁热路径
void AIEngineFront::WorkerRoutine(int worker_idx)
{
    SetThreadAffinity(worker_cores[worker_idx]);
    WorkerTaskBox* my_box = task_boxes[worker_idx];

    //分配属于该线程的Logits缓存，避免破坏L1D连续性
    alignas(64) float logits[32000];

    while (true) {
        while (my_box->state_flag.load(std::memory_order_acquire) != 1) {
            if (global_terminate.load(std::memory_order_acquire)) return;
            _mm_pause(); 
        }
        if (global_terminate.load(std::memory_order_acquire)) return;

        //读取依赖的父节点隐状态
        float* parent_state = kv_cache->GetFP32Base() + (my_box->parent_slot * hidden_dim);
        float* my_state = kv_cache->GetFP32Base() + (my_box->allocated_slot * hidden_dim);

        //修改的状态注入
        ComputeStateTransition(my_box->input_token, parent_state, my_state, hidden_dim);

        //接入重构算子
        //传入专用物理页workspace_i16以避免线程内部malloc
        extern const int8_t* g_quant_weights;
        extern const QuantDescriptor* g_scales;
        DraftGEMV_INT8_AVX2(
            g_quant_weights, 
            g_scales, 
            my_state, 
            logits, 
            my_box->workspace_i16, 
            hidden_dim, 
            32000
        );

        //选择最优分支
        my_box->predicted_token = BranchlessArgMax_AVX2(logits, 32000);

        //将结果写入自己的KV槽位 (FP32 & INT8)
        kv_cache->WriteTreeSlotFP32(my_box->allocated_slot, my_state);

        //交还控制权
        my_box->state_flag.store(2, std::memory_order_release);
    }
}

//总控主频管线
/*动态深度扇出版*/
void AIEngineFront::RunTreeInferencePipeline(
    const int8_t* quant_weights, const QuantDescriptor* weight_scales,
    float* root_hidden_state, uint32_t target_tokens) 
{
    SetThreadAffinity(hq_core_id);
    uint32_t step = 0;
    uint32_t root_slot = kv_cache->GetCurrentHead();
    kv_cache->WriteTreeSlotFP32(root_slot, root_hidden_state);

    while (step < target_tokens) {
        shared_proposal->Reset();
        shared_proposal->state_flags.store(1, std::memory_order_release);

        //动态深度扇出
        //支持任意数量的Worker，形成深度推测链条
        uint32_t current_parent_slot = root_slot;
        uint32_t current_input_token = 0; //WIP: 真实系统中应传入上一步确定的Token
        int32_t current_parent_idx = -1;

        for (size_t i = 0; i < worker_cores.size(); ++i) {
            if (shared_proposal->num_valid_drafts >= DraftProposal::MAX_K_DRAFTS) break;

            //给Worker分配任务
            task_boxes[i]->parent_slot = current_parent_slot;
            task_boxes[i]->allocated_slot = (root_slot + i + 1) % kv_cache->GetMaxSeqLen();
            task_boxes[i]->input_token = current_input_token;
            task_boxes[i]->tree_node_idx = i;
            
            //启动指令
            task_boxes[i]->state_flag.store(1, std::memory_order_release); 

            //等待当前Worker算完（下一层推测需要它的输出Token）
            //若要完全重叠计算，可引入Top-K候选分支并行，此处为单链深度推测
            while (task_boxes[i]->state_flag.load(std::memory_order_acquire) != 2) { 
                _mm_pause(); 
            }

            //压入共享提案区，形成树的拓扑边
            shared_proposal->PushTreeToken(task_boxes[i]->predicted_token, current_parent_idx);

            //为下一个Worker更新上下文
            current_parent_slot = task_boxes[i]->allocated_slot;
            current_input_token = task_boxes[i]->predicted_token;
            current_parent_idx = i; 

            //重置Worker状态机
            task_boxes[i]->state_flag.store(0, std::memory_order_relaxed);
        }

        //iGPU验证与回滚
#ifdef TARGET_ARCH_DGPU_PCIE
        _mm_clflush(&shared_proposal->draft_tokens);
        _mm_clflush(&shared_proposal->parent_indices);
        _mm_mfence(); 
#endif
        shared_proposal->state_flags.store(2, std::memory_order_release);

        //等待iGPU审计
        while (shared_proposal->state_flags.load(std::memory_order_acquire) != 0) { _mm_pause(); }

        uint32_t accepted_count = shared_proposal->num_valid_drafts;
        step += accepted_count;

        //O(1)
        //如果推测全错，root_slot不变，同时下次直接覆盖脏数据
        //如果对了N个，Root推进N步，旧数据失效
        if (accepted_count > 0) {
            root_slot = (root_slot + accepted_count) % kv_cache->GetMaxSeqLen();
        }
    }
}
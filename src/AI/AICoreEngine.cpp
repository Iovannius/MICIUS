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
    cpu_set_t cpuset;
    CPU_ZERO(&cpuset);
    CPU_SET(cpu_id, &cpuset);
    pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
#endif
}

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
                    ULONG_PTR lowest_bit = mask & ((~mask) + 1);
                    
                    int idx = 0;
                    while ((lowest_bit & 1) == 0) {
                        lowest_bit >>= 1;
                        idx++;
                    }
                    
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
    shared_proposal[0] = mem_pool->GetDraftBuffer(0);
    shared_proposal[1] = mem_pool->GetDraftBuffer(1);
    data_front = new AiDataFront(kv_cache, shared_proposal[0]);
    
    std::vector<int> phys_cores = ProbePhysicalCores();
    int total_logical_cores = std::thread::hardware_concurrency();

    //若全是P核关闭超线程：logical == phys
    //若全是P核开启超线程：logical == phys * 2
    //异构大小核(P+E)的logical必然介于两者之间
    bool is_hetero = (total_logical_cores != phys_cores.size()) && (total_logical_cores != phys_cores.size() * 2);

    if (phys_cores.size() < 2) {
        int logical_count = std::thread::hardware_concurrency();
        if (logical_count < 2) logical_count = 2; 
        phys_cores.clear();
        for (int i = 0; i < logical_count; ++i) phys_cores.push_back(i);
        std::cout << "[Kernel] OS Topology Probe failed. Fallback to " << phys_cores.size() << " logical cores.\n";
        
        hq_core_id = phys_cores[0];
        recommended_shadow_core = (phys_cores.size() > 1) ? phys_cores[1] : phys_cores[0];
        for (size_t i = 1; i < phys_cores.size(); ++i) {
            if (phys_cores[i] != recommended_shadow_core) worker_cores.push_back(phys_cores[i]);
        }

    } else if (phys_cores.size() == 2) {
        //双核处理器的策略，保证唯一Worker的算力绝对纯净
        hq_core_id = phys_cores[0]; 
        recommended_shadow_core = -1;
        for (int id = 0; id < total_logical_cores; ++id) {
            bool is_phys_head = false;
            for (int p_core : phys_cores) { if (p_core == id) is_phys_head = true; }
            if (!is_phys_head) {
                recommended_shadow_core = id;
                break;
            }
        }
        /*极端情况下拓扑失败，回到传统模式*/
        if (recommended_shadow_core == -1) recommended_shadow_core = hq_core_id + 1;
        
        worker_cores.push_back(phys_cores[1]);

    } else if (is_hetero) {
        //异构大小核的策略，保障Oracle响应优先级

        //Total_Logical = P * 2 + E
        //Total_Physical = P + E
        //=> P = Total_Logical - Total_Physical
        int num_p_cores = total_logical_cores - phys_cores.size();
        int total_e_cores = phys_cores.size() - num_p_cores;

        hq_core_id = phys_cores[0];              //HQ占用P核0
        recommended_shadow_core = phys_cores[1]; //iGPU占用P核1

        //识别并剔除SoC岛上的LP-E核
        int valid_e_cores = total_e_cores;
        if (total_e_cores % 4 == 2) {
            valid_e_cores -= 2; 
            std::cout << "[Kernel] Meteor Lake detected. Dropping 2 SoC LP-E cores.\n";
        }

        //将Worker阵列全部且仅限绑定至E核
        for (int i = 0; i < valid_e_cores; ++i) {
            worker_cores.push_back(phys_cores[num_p_cores + i]); 
        }

    } else {
        //同构多核的策略
        hq_core_id = phys_cores[0];                 //HQ占用Core 0
        recommended_shadow_core = phys_cores.back();//iGPU强行踢到最后一个物理核
        for (size_t i = 1; i < phys_cores.size() - 1; ++i) {
            worker_cores.push_back(phys_cores[i]);  //中间的物理核成为Worker阵列
        }
    }

    std::cout << "[Kernel] Worker Array size: " << worker_cores.size() << "\n";

    local_queues = std::vector<ChaseLevDeque<FrontierTask, 1024>>(worker_cores.size());
    
    for (size_t i = 0; i < worker_cores.size(); ++i) {

        worker_gen[i].val.store(0, std::memory_order_relaxed);
        WorkerContext* ctx = static_cast<WorkerContext*>(_mm_malloc(sizeof(WorkerContext), 64));
        
        ctx->workspace_i8 = static_cast<int8_t*>(_mm_malloc(hidden_dim * sizeof(int8_t), 64));
        ctx->logits = static_cast<float*>(_mm_malloc(32000 * sizeof(float), 64));
        
        ctx->tlab_count = 0;
        ctx->prng_state = (uint32_t)(1337 + i * 19);
        
        std::memset(ctx->workspace_i8, 0, hidden_dim * sizeof(int8_t));
        std::memset(ctx->logits, 0, 32000 * sizeof(float));
        
        worker_ctxs.push_back(ctx); worker_threads.emplace_back(&AIEngineFront::WorkerRoutine, this, i);
    }
}

AIEngineFront::~AIEngineFront()
{
    global_terminate.store(true, std::memory_order_release);

    for (auto& t : worker_threads) {
        if (t.joinable()) t.join();
    }

    for (auto ctx : worker_ctxs) {
        _mm_free(ctx->workspace_i8);
        _mm_free(ctx->logits);
        _mm_free(ctx);
    }

    delete data_front;
    delete kv_cache;
    delete mem_pool;
}

inline void AIEngineFront::ComputeStateTransition(uint32_t token, const float* parent_state, float* new_state, uint32_t dim)
{
    for (uint32_t i = 0; i < dim; ++i) new_state[i] = parent_state[i] + (token * 0.0001f);
}

void AIEngineFront::WorkerRoutine(int w_idx)
{

    SetThreadAffinity(worker_cores[w_idx]);
    
    WorkerContext* ctx = worker_ctxs[w_idx];
    
    extern const int8_t* g_quant_weights;
    extern const QuantDescriptor* g_scales;
    
    int idle_spins = 0;
    uint32_t local_gen = global_generation.load(std::memory_order_relaxed);

    while(!global_terminate.load(std::memory_order_acquire)) {
        uint32_t current_gen = global_generation.load(std::memory_order_relaxed);
        worker_gen[w_idx].val.store(current_gen, std::memory_order_release); 
        if (local_gen != current_gen) {
            local_gen = current_gen;
        }

        FrontierTask task;
        bool got = false;
        
        if (seed_claimed.load(std::memory_order_acquire) == 0) {
            
            int expected = 0;
            if (seed_claimed.compare_exchange_strong(expected, 1, std::memory_order_acq_rel)) {
                task = global_seed_task;
                got = true;
            }
        }

        if(!got) got = local_queues[w_idx].pop_bottom(task);
        
        if(!got && worker_cores.size() > 1) {
            /*Xorshift32极速伪随机发生器*/
            ctx->prng_state ^= ctx->prng_state << 13;
            ctx->prng_state ^= ctx->prng_state >> 17;
            ctx->prng_state ^= ctx->prng_state << 5;
            
            //每次失败随机挑一个Victim献祭（窃取），修复前版本O(N^2)的缓存行广播性能灾难
            size_t victim = ctx->prng_state % worker_cores.size();
            if (victim != (size_t)w_idx) {
                got = local_queues[victim].steal_top(task);
            }
        }

        if(!got) {

            if (idle_spins < 50) {
                //阶段1
                //极短停留（约200纳秒内响应）应对正常内存延迟
                _mm_pause();
            } else if (idle_spins < 200) {
                //阶段2
                //中等停留，降低取指和发射单元压力，降低局部热点
                for(int p = 0; p < 8; ++p) _mm_pause();
            } else {
                //阶段3
                //让出核心执行权降低硅片功耗，防止过流
                std::this_thread::yield();
            }
            
            idle_spins++;
            continue; 
        }
        
        //抢到任务立刻清空休眠计数器，恢复高速轮询状态
        idle_spins = 0; 
        
        //任务过滤与纪元强制同步
        uint32_t current_global_gen = global_generation.load(std::memory_order_relaxed);
        if (task.gen_id != current_global_gen) {
            continue; //丢弃彻底过期的任务
        }
        
        //如果抢到了合法新任务但本地纪元还没跟上，强制提前同步防止下一次循环误删队列
        if (local_gen != task.gen_id) {
            local_gen = task.gen_id;
            worker_gen[w_idx].val.store(local_gen, std::memory_order_release);
        }

        DraftProposal* active_prop = shared_proposal[task.tree_idx];
        
        if (active_prop->num_valid_drafts.load(std::memory_order_relaxed) >= DraftProposal::MAX_K_DRAFTS) {
            continue;
        }

        task.fusion_depth = 4;
        bool fusion_continue = true;

        auto tlab_alloc = [&]() -> uint32_t {
            if (ctx->tlab_count == 0) {
                //通过总线一次性获取16个插槽
                ctx->tlab_count = kv_cache->BulkAllocateSlots(ctx->tlab_slots, 16);
                if (ctx->tlab_count == 0) return 0xFFFFFFFF;
            }
            return ctx->tlab_slots[--ctx->tlab_count];
        };
        
        auto tlab_free = [&](uint32_t slot) {
            //本地还有空间
            //塞回本地，填满则使用原子指令操作归还全局
            if (ctx->tlab_count < 16) {
                ctx->tlab_slots[ctx->tlab_count++] = slot;
            } else {
                kv_cache->FreeSlot(slot);
            }
        };
        
        while(fusion_continue && task.fusion_depth > 0) {

            fusion_continue = false; 
            if(task.gen_id != global_generation.load(std::memory_order_relaxed)) break; 
            
            float *p_state = kv_cache->GetFP32Base()+(task.parent_slot*hidden_dim), *m_state = kv_cache->GetFP32Base()+(task.alloc_slot*hidden_dim);
            ComputeStateTransition(task.input_token, p_state, m_state, hidden_dim);
            DraftGEMV_INT8_AVX2(g_quant_weights, g_scales, m_state, ctx->logits, ctx->workspace_i8, hidden_dim, 32000);
            
            if(task.gen_id != global_generation.load(std::memory_order_relaxed)) break;
            kv_cache->WriteTreeSlotFP32(task.alloc_slot, m_state);
            
            uint32_t top_k[3];
            BranchlessTop3_AVX2(ctx->logits, 32000, top_k); 
            
            //独立提取k=0优先执行，避免30个插槽容量上限的反向瓶颈
            bool golden_survived = false;
            FrontierTask child_golden;
            
            uint32_t slot_0 = tlab_alloc();
            if (slot_0 == 0xFFFFFFFF) break;

            //对于k=0 (可能携带Golden属性)
            int32_t tree_idx_0 = active_prop->SpawnNodeConcurrent(top_k[0], task.parent_tree_idx, slot_0, task.is_golden);            
            
            if(tree_idx_0 != -1) {
                
                child_golden.gen_id = task.gen_id;
                child_golden.parent_slot = task.alloc_slot;
                child_golden.alloc_slot = slot_0;
                child_golden.input_token = top_k[0];
                child_golden.parent_tree_idx = tree_idx_0; 
                child_golden.branch_id = (task.branch_id == 0) ? tree_idx_0 : task.branch_id; 
                child_golden.fusion_depth = task.fusion_depth - 1;
                child_golden.tree_idx = task.tree_idx;
                child_golden.is_golden = task.is_golden;
                
                golden_survived = true;

            } else {
                tlab_free(slot_0);
            }

            //处理k=2和k=1的旁支
            //推入队列的LIFO顺序保证k=1会比k=2先弹出
            for(int k=2; k>=1; --k) { 
                
                uint32_t new_slot = tlab_alloc();
                if (new_slot == 0xFFFFFFFF) break;
                
                //对于旁支k=1, k=2传false或依赖默认参数
                int32_t new_tree_idx = active_prop->SpawnNodeConcurrent(top_k[k], task.parent_tree_idx, new_slot, false);
                if(new_tree_idx == -1) {
                    tlab_free(new_slot);
                    break;
                } //容量超限，静默丢弃旁支

                FrontierTask child;

                child.gen_id=task.gen_id;
                child.parent_slot=task.alloc_slot;
                child.alloc_slot=new_slot;
                child.input_token=top_k[k];
                child.parent_tree_idx=new_tree_idx;
                child.branch_id=new_tree_idx; 
                child.fusion_depth = task.fusion_depth - 1;
                child.tree_idx = task.tree_idx;
                child.is_golden = false;

                local_queues[w_idx].push_bottom(child);
            }

            //真路径状态机继承
            if (golden_survived) {
                if (child_golden.fusion_depth > 0) {

                    task = child_golden;
                    fusion_continue = true; //深度足够，寄存器内继续

                } else {
                    
                    local_queues[w_idx].push_bottom(child_golden); //深度耗尽，推入队列等待下次弹出
                
                }
            }
        }
    }
}

//新Spearhead GC管线
void AIEngineFront::RunTreeInferencePipeline(const int8_t* quant_weights, const QuantDescriptor* weight_scales, float* root_hidden_state, uint32_t target_tokens)
{
    SetThreadAffinity(hq_core_id);
    
    uint32_t step = 0;
    uint32_t cpu_tree = 0;
    uint32_t accepted_token = 0; 
    
    uint32_t roots[2];
    uint32_t base_slots[2];

    roots[0] = kv_cache->AllocateSlot();
    kv_cache->WriteTreeSlotFP32(roots[0], root_hidden_state);
    base_slots[0] = kv_cache->AllocateSlot();

    shared_proposal[cpu_tree]->Reset();
    uint32_t epoch = global_generation.load();
    
    FrontierTask seed;

    seed.gen_id=epoch;
    seed.parent_slot=roots[0];
    seed.alloc_slot=base_slots[0];
    seed.input_token=accepted_token;
    seed.parent_tree_idx=-1;
    seed.branch_id=0;
    seed.fusion_depth=4;
    seed.tree_idx=cpu_tree;
    seed.is_golden=true;
    global_seed_task = seed;
    seed_claimed.store(0, std::memory_order_release);

    while(step < target_tokens) {

        //将HQ的屏障从num_valid_drafts切换至commit_counter
        //std::memory_order_acquire会与Worker的release握手，读到30则Worker写入的数组数据在内存中必定可见
        while(shared_proposal[cpu_tree]->commit_counter.load(std::memory_order_acquire) < DraftProposal::MAX_K_DRAFTS) {
            _mm_pause();
        }

        shared_proposal[cpu_tree]->state_flags.store(2, std::memory_order_release);
        uint32_t gpu_tree = cpu_tree;
        
        uint32_t next_tree = 1 - cpu_tree; shared_proposal[next_tree]->Reset();
        int32_t golden_idx = shared_proposal[gpu_tree]->golden_leaf_idx.load(std::memory_order_relaxed);
        if (golden_idx < 0) golden_idx = DraftProposal::MAX_K_DRAFTS - 1;

        roots[next_tree] = shared_proposal[gpu_tree]->slot_mapping[golden_idx];
        base_slots[next_tree] = kv_cache->AllocateSlot();
        uint32_t next_token = shared_proposal[gpu_tree]->draft_tokens[golden_idx];

        global_generation.fetch_add(1, std::memory_order_release);

        FrontierTask next_seed;
        
        next_seed.gen_id = global_generation.load();
        next_seed.parent_slot = roots[next_tree];
        next_seed.alloc_slot = base_slots[next_tree];
        next_seed.input_token = next_token;
        next_seed.parent_tree_idx = -1;
        next_seed.branch_id = 0;
        next_seed.fusion_depth = 4;
        next_seed.tree_idx = next_tree;
        next_seed.is_golden = true;
        global_seed_task = next_seed;
        seed_claimed.store(0, std::memory_order_release);

        while(shared_proposal[gpu_tree]->state_flags.load(std::memory_order_acquire) != 0) _mm_pause();

        int32_t accepted_idx = shared_proposal[gpu_tree]->accepted_tree_node_idx;
        uint32_t num_gpu_drafts = shared_proposal[gpu_tree]->num_valid_drafts.load(std::memory_order_relaxed);

        if (accepted_idx == golden_idx) {

            uint32_t depth = 0;
            int32_t curr = accepted_idx;
            
            while(curr != -1) {
                depth++;
                curr = shared_proposal[gpu_tree]->parent_indices[curr];
            }
            
            step += (depth + 1);
            
            uint32_t rej_slots[DraftProposal::MAX_K_DRAFTS + 2];
            uint32_t rc = 0;
            
            rej_slots[rc++] = roots[gpu_tree];
            rej_slots[rc++] = base_slots[gpu_tree];
            
            for(uint32_t i=0; i<num_gpu_drafts; ++i) if (i != (uint32_t)accepted_idx) rej_slots[rc++] = shared_proposal[gpu_tree]->slot_mapping[i];
            
            kv_cache->BulkFreeSlots(rej_slots, rc);
            cpu_tree = next_tree;

        } else {

            global_generation.fetch_add(1, std::memory_order_release);
            
            uint32_t target_gen = global_generation.load();
            
            for (size_t w = 0; w < worker_cores.size(); ++w) {
                while(worker_gen[w].val.load(std::memory_order_acquire) < target_gen) {
                    _mm_pause();
                }
            }
            uint32_t abort_slots[DraftProposal::MAX_K_DRAFTS + 1];
            uint32_t ac = 0;
            
            abort_slots[ac++] = base_slots[next_tree];
            
            uint32_t num_next = shared_proposal[next_tree]->num_valid_drafts.load(std::memory_order_relaxed);
            
            for(uint32_t i=0; i<num_next; ++i) abort_slots[ac++] = shared_proposal[next_tree]->slot_mapping[i];
            
            kv_cache->BulkFreeSlots(abort_slots, ac);

            uint32_t rej_slots[DraftProposal::MAX_K_DRAFTS + 2];
            uint32_t rc = 0;
            
            rej_slots[rc++] = roots[gpu_tree];
            
            if (accepted_idx >= 0) {

                uint32_t depth = 0;
                int32_t curr = accepted_idx;
                
                while(curr != -1) {
                    depth++;
                    curr = shared_proposal[gpu_tree]->parent_indices[curr];
                }
                
                step += (depth + 1);
                
                rej_slots[rc++] = base_slots[gpu_tree];

                for(uint32_t i=0; i<num_gpu_drafts; ++i) if (i != (uint32_t)accepted_idx) rej_slots[rc++] = shared_proposal[gpu_tree]->slot_mapping[i];
                
                roots[next_tree] = shared_proposal[gpu_tree]->slot_mapping[accepted_idx];
                accepted_token = shared_proposal[gpu_tree]->draft_tokens[accepted_idx];
                
            } else {
                step += 1; 
                for(uint32_t i=0; i<num_gpu_drafts; ++i) rej_slots[rc++] = shared_proposal[gpu_tree]->slot_mapping[i];
                roots[next_tree] = base_slots[gpu_tree]; 
            }

            kv_cache->BulkFreeSlots(rej_slots, rc);

            shared_proposal[next_tree]->Reset();
            base_slots[next_tree] = kv_cache->AllocateSlot();
            next_seed.gen_id = global_generation.load();
            next_seed.parent_slot = roots[next_tree];
            next_seed.alloc_slot = base_slots[next_tree];
            next_seed.input_token = accepted_token;
            next_seed.parent_tree_idx = -1;
            next_seed.tree_idx = next_tree;
            next_seed.is_golden = true;
            global_seed_task = next_seed;
            seed_claimed.store(0, std::memory_order_release);
            
            cpu_tree = next_tree;
        }
    }
}
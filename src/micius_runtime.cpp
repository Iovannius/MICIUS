#include "micius_runtime.h"
#include "AICoreEngine.h"

//不透明句柄的具体实现（封装引擎实例）
struct MiciusContext {
    AIEngineFront* core_engine;
    MiciusConfig config;
    bool pipeline_initialized;
};

extern "C" {

MiciusContext* Micius_CreateEngine(const MiciusConfig* config)
{
    if (!config) return nullptr;

    MiciusContext* ctx = new MiciusContext();
    ctx->config = *config;
    ctx->pipeline_initialized = false;

    //实例化内核（池大小MB->Bytes）
    size_t pool_bytes = (size_t)config->kv_pool_size_mb * 1024 * 1024;
    ctx->core_engine = new AIEngineFront(pool_bytes, config->max_seq_len, config->hidden_dim);

    return ctx;
}

void Micius_DestroyEngine(MiciusContext* ctx)
{
    if (ctx) {
        if (ctx->core_engine) {
            delete ctx->core_engine;
        }
        delete ctx;
    }
}

bool Micius_LoadQuantWeights(MiciusContext* ctx, const int8_t* quant_weights, const void* scales)
{
    if (!ctx || !ctx->core_engine) return false;
    ctx->core_engine->LoadWeights(quant_weights, static_cast<const QuantDescriptor*>(scales));
    return true;
}

void Micius_RegisterVerifier(MiciusContext* ctx, MiciusVerifierCallback verifier_cb, void* user_data)
{
    if (!ctx || !ctx->core_engine) return;
    
    //必须将外部的C回调桥接入核心引擎回调，因为引擎定义的cb签名使用void*来掩盖类型
    ctx->core_engine->RegisterVerifier(
        reinterpret_cast<void (*)(void*, void*, void*)>(verifier_cb), 
        ctx,
        user_data
    );
}

MiciusStatus Micius_Step(MiciusContext* ctx, uint32_t input_token, const float* hidden_state, uint32_t* out_accepted_token)
{
    if (!ctx || !ctx->core_engine) return MICIUS_STATE_OOM_ERROR;

    if (!ctx->pipeline_initialized) {
        ctx->core_engine->InitPipeline(const_cast<float*>(hidden_state), input_token);
        ctx->pipeline_initialized = true;
    }

    PumpResult result = ctx->core_engine->PumpSingleToken(input_token, hidden_state, out_accepted_token);
    
    if (result == PumpResult::YIELD_TOKEN) {
        return MICIUS_STATE_YIELD_TOKEN;
    } else if (result == PumpResult::TREE_REJECTED) {
        return MICIUS_STATE_TREE_REJECTED;
    } else {
        return MICIUS_STATE_OOM_ERROR;
    }
}

uint32_t Micius_GetTreeValidCount(MiciusDraftTree* tree)
{
    if (!tree) return 0;
    //解析内部类型
    DraftProposal* proposal = reinterpret_cast<DraftProposal*>(tree);
    return proposal->commit_counter.load(std::memory_order_acquire);
}

void Micius_AcceptTreeNode(MiciusDraftTree* tree, int32_t node_idx)
{
    if (!tree) return;
    DraftProposal* proposal = reinterpret_cast<DraftProposal*>(tree);
    
    //写入真路径索引
    proposal->accepted_tree_node_idx = node_idx;
    //释放状态机锁，唤醒正在自旋的HQ控制面线程
    proposal->state_flags.store(0, std::memory_order_release);
}

} //extern "C"
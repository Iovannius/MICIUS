#ifndef MICIUS_RUNTIME_H
#define MICIUS_RUNTIME_H

#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

//Opaque Handles
//外部调用无法探明MiciusContext内部包含ChaseLevDeque还是std::atomic
typedef struct MiciusContext MiciusContext;
typedef struct MiciusDraftTree MiciusDraftTree;

//拓扑与模型配置，动态化替代硬编码
typedef struct {
    uint32_t hidden_dim;
    uint32_t vocab_size;
    uint32_t max_seq_len; //环形KV Cache的深度
    uint32_t kv_pool_size_mb; //异构内存池预留大页大小(MB)
    
    int worker_threads; //Worker线程数，传入0表示启用引擎的自动物理核探测与绑定
} MiciusConfig;

//Oracle预言回调接口（控制反转，剥离ShadowiGPU）
//外部的真实大模型（如TensorRT和Llama.cpp）通过此接口接收树形草稿进行验证
//ctx: 引擎上下文
//tree_handle: 当前回合生成的推测树句柄
//user_data: 外部的自定义指针，可用于传递自定义GPU流等
typedef void (*MiciusVerifierCallback)(MiciusContext* ctx, MiciusDraftTree* tree_handle, void* user_data);

//生命周期管理API
MiciusContext* Micius_CreateEngine(const MiciusConfig* config);
void           Micius_DestroyEngine(MiciusContext* ctx);

//权重与内存装载API
//允许外部Loader（如解析GGUF的模块）直接将权重拷贝到引擎锁定的物理页中
bool Micius_LoadQuantWeights(MiciusContext* ctx, const int8_t* quant_weights, const void* scales);

//非阻塞单步驱动执行流状态机
//取代while(step < target_tokens)
typedef enum {
    MICIUS_STATE_DRAFTING,        //Worker阵列正在进行展开推测树过程（暂不使用，同步模式下被隐藏）
    MICIUS_STATE_AWAITING_VERIFY, //树已生成，挂起等待Oracle预言回调（暂不使用，同步模式下被隐藏）
    MICIUS_STATE_YIELD_TOKEN,     //引擎产出了新的确定性Token供输出
    MICIUS_STATE_TREE_REJECTED,   //草稿树被预言否决，外部需补发目标Token
    MICIUS_STATE_OOM_ERROR        //致命错误，例如KV槽位耗尽或内存崩溃
} MiciusStatus;

//提交新的Prompt隐状态，启动或推进引擎状态机
MiciusStatus Micius_Step(MiciusContext* ctx, 
                         uint32_t input_token, 
                         const float* hidden_state, 
                         uint32_t* out_accepted_token);

//注册外部的预言验证逻辑
void Micius_RegisterVerifier(MiciusContext* ctx, MiciusVerifierCallback verifier_cb, void* user_data);

//预言审计API
//获取当前树中成功生成的草稿节点总数
uint32_t Micius_GetTreeValidCount(MiciusDraftTree* tree);

//预言验证完成后，提交被接受的节点索引（传入-1表示全盘拒绝）并交回控制权
void Micius_AcceptTreeNode(MiciusDraftTree* tree, int32_t node_idx);


#ifdef __cplusplus
}
#endif

#endif //MICIUS_RUNTIME_H
#ifndef MICIUS_RUNTIME_H
#define MICIUS_RUNTIME_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

//Opaque Handles
//外部调用无法探明MiciusContext内部包含ChaseLevDeque还是std::atomic
typedef struct MiciusContext MiciusContext;
typedef struct MiciusDraftTree MiciusDraftTree;

#define MICIUS_ABI_VERSION UINT32_C(0x00060003)

typedef enum {
    MICIUS_EXECUTION_REFERENCE = 0,
    MICIUS_EXECUTION_SPECULATIVE = 1
} MiciusExecutionMode;

//拓扑与模型配置，动态化替代硬编码
typedef struct {
    uint32_t hidden_dim;
    uint32_t vocab_size;
    uint32_t max_seq_len; //环形KV Cache的深度
    uint32_t kv_pool_size_mb; //异构内存池预留大页大小(MB)
    
    int worker_threads; //Worker线程数，传入0表示启用引擎的自动物理核探测与绑定

    uint32_t num_layers;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint32_t intermediate_dim;
    uint32_t execution_mode;
    float rms_norm_epsilon;
    float rope_freq_base;
    uint32_t rope_dimension_count;
} MiciusConfig;

typedef enum {
    MICIUS_ERROR_NONE = 0,
    MICIUS_ERROR_INVALID_ARGUMENT,
    MICIUS_ERROR_UNSUPPORTED_ABI,
    MICIUS_ERROR_UNSUPPORTED_MODE,
    MICIUS_ERROR_OUT_OF_MEMORY,
    MICIUS_ERROR_MODEL_NOT_BOUND,
    MICIUS_ERROR_INTERNAL,
    MICIUS_ERROR_CONTEXT_FULL,
    MICIUS_ERROR_INVALID_MODEL_DATA,
    MICIUS_ERROR_NUMERIC,
    MICIUS_ERROR_NOT_READY
} MiciusErrorCode;

//张量桥接端
typedef struct {
    const char* tensor_name;
    const void* tensor_data;
} MiciusTensorEntry;

// Retained as a fail-closed ABI symbol. This legacy entry has no byte size,
// shape, or type and cannot safely bind a model in v0.6-rebuild.
bool Micius_BindModelTensors(MiciusContext* ctx, const MiciusTensorEntry* entries, uint32_t count);

typedef enum {
    MICIUS_TENSOR_F32 = 0,
    MICIUS_TENSOR_Q8_32_WEIGHT = 1,
    MICIUS_TENSOR_Q8_32_SCALE = 2
} MiciusTensorDataType;

// Scale descriptor for one symmetric 32-element weight block. The block width
// intentionally matches GGUF Q8_0, so loading does not merge scales or requantize
// values. zero_point and both padding words must be zero.
typedef struct {
    float scale;
    int32_t zero_point;
    uint32_t padding[2];
} MiciusQ8Block32Scale;

typedef struct {
    const char* tensor_name;
    const void* tensor_data;
    size_t byte_size;
    uint32_t data_type;
    uint32_t rank;
    // Logical row-major dimensions: [rows, columns]. Q/K rows must already use
    // GGUF's RoPE-ready adjacent-pair order; the engine never permutes them.
    uint64_t dimensions[4];
    size_t row_stride_bytes;
    uint32_t flags;         // zero means borrowed for the engine lifetime
} MiciusTensorEntryV1;

bool Micius_BindModelTensorsV1(
    MiciusContext* ctx,
    const MiciusTensorEntryV1* entries,
    uint32_t count,
    size_t entry_size,
    uint32_t abi_version);

//Oracle预言回调接口（控制反转，剥离ShadowiGPU）
//外部的真实大模型（如TensorRT和Llama.cpp）通过此接口接收树形草稿进行验证
//ctx: 引擎上下文
//tree_handle: 当前回合生成的推测树句柄
//user_data: 外部的自定义指针，可用于传递自定义GPU流等
typedef void (*MiciusVerifierCallback)(MiciusContext* ctx, MiciusDraftTree* tree_handle, void* user_data);

//生命周期管理API
MiciusContext* Micius_CreateEngine(const MiciusConfig* config);
MiciusContext* Micius_CreateEngineV1(const MiciusConfig* config,
                                     size_t config_size,
                                     uint32_t abi_version);
bool           Micius_InitConfig(MiciusConfig* config, size_t config_size);
void           Micius_DestroyEngine(MiciusContext* ctx);
void           Micius_ResetEngine(MiciusContext* ctx);
bool           Micius_ResetEngineV1(MiciusContext* ctx);
MiciusErrorCode Micius_GetLastErrorCode(void);
const char*     Micius_GetLastError(void);

//Legacy fail-closed symbol; use Micius_BindModelTensorsV1 for checked loading.
bool Micius_LoadQuantWeights(MiciusContext* ctx, const int8_t* quant_weights, const void* scales);

//非阻塞单步驱动执行流状态机
//取代while(step < target_tokens)
typedef enum {
    MICIUS_STATE_DRAFTING,        //Worker阵列正在进行展开推测树过程（暂不使用，同步模式下被隐藏）
    MICIUS_STATE_AWAITING_VERIFY, //树已生成，挂起等待Oracle预言回调（暂不使用，同步模式下被隐藏）
    MICIUS_STATE_YIELD_TOKEN,     //引擎产出了新的确定性Token供输出
    MICIUS_STATE_TREE_REJECTED,   //草稿树被预言否决，外部需补发目标Token
    MICIUS_STATE_OOM_ERROR,       //致命错误，例如KV槽位耗尽或内存崩溃
    MICIUS_STATE_NOT_READY,       //模型尚未完成绑定，或reference路径尚未就绪
    MICIUS_STATE_INVALID_ARGUMENT,//调用参数不满足公开契约
    MICIUS_STATE_CONTEXT_FULL,    //已达到配置的最大权威上下文长度
    MICIUS_STATE_TOKEN_ACCEPTED,  //真实Prefill Token已提交，未请求输出Token
    MICIUS_STATE_NUMERIC_ERROR    //模型数据或中间值出现非有限数
} MiciusStatus;

//提交新的Prompt隐状态，启动或推进引擎状态机
MiciusStatus Micius_Step(MiciusContext* ctx, 
                         uint32_t input_token, 
                         const float* hidden_state, 
                         uint32_t* out_accepted_token);

// Authoritative single-thread reference APIs. Prefill commits exactly one real
// token. Decode commits one input token and returns the greedy next token.
MiciusStatus Micius_PrefillTokenV1(MiciusContext* ctx, uint32_t input_token);
MiciusStatus Micius_DecodeOneV1(
    MiciusContext* ctx,
    uint32_t input_token,
    uint32_t* out_token);
uint32_t Micius_GetSequenceLengthV1(MiciusContext* ctx);
bool Micius_CopyLastLogitsV1(
    MiciusContext* ctx,
    float* out_logits,
    size_t logits_count);
bool Micius_CopyReferenceKVV1(
    MiciusContext* ctx,
    uint32_t layer,
    uint32_t position,
    float* out_key,
    float* out_value,
    // Number of float elements available in each output buffer.
    size_t element_count);

//Speculative ABI symbols are retained but fail closed while that mode is disabled.
void Micius_RegisterVerifier(MiciusContext* ctx, MiciusVerifierCallback verifier_cb, void* user_data);

//预言审计API
//获取当前树中成功生成的草稿节点总数
uint32_t Micius_GetTreeValidCount(MiciusDraftTree* tree);

//预言验证完成后，提交被接受的节点索引（传入-1表示全盘拒绝）并交回控制权
void Micius_AcceptTreeNode(MiciusDraftTree* tree, int32_t node_idx);

//遥测与监控API
//引擎全局计数器
typedef struct {
    uint32_t total_tokens_generated; //总产出Token数
    uint32_t tree_reject_count;      //推测树被全盘拒绝的次数
    uint32_t kv_slots_used;          //当前占用的KV槽数
    uint32_t kv_slots_total;         //内存池总KV槽容量
} MiciusRuntimeStats;

//Worker阵列元数据
typedef struct {
    int core_id;              //绑定的物理/逻辑核心ID
    uint32_t tasks_processed; //本地队列处理的任务数
    uint32_t tasks_stolen;    //窃取的任务数（压力指示器）
} MiciusWorkerInfo;

//遥测接口
void Micius_GetRuntimeStats(MiciusContext* ctx, MiciusRuntimeStats* out_stats);
uint32_t Micius_GetWorkerCount(MiciusContext* ctx);
bool Micius_GetWorkerInfo(MiciusContext* ctx, uint32_t worker_index, MiciusWorkerInfo* out_info);

#ifdef __cplusplus
}
#endif

#endif //MICIUS_RUNTIME_H

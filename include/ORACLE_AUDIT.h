#pragma once
#include "micius_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

//暴露不透明句柄取代单例
typedef struct OracleContext OracleContext;

//传入绑定的核心并返回独立Oracle实例句柄
OracleContext* ORACLE_InitHardwareThread(int target_core_id);

//销毁指定Oracle实例
void ORACLE_ShutdownHardwareThread(OracleContext* octx);

//注册给Micius引擎的C-ABI回调入口，此时user_data将传入OracleContext*
void ORACLE_VerificationCallback(MiciusContext* ctx, MiciusDraftTree* tree_handle, void* user_data);

#ifdef __cplusplus
}
#endif
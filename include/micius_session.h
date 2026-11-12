#ifndef MICIUS_SESSION_H
#define MICIUS_SESSION_H
#include "micius_gguf.h"
#ifdef __cplusplus
extern "C" {
#endif
#define MICIUS_SESSION_ABI_V1 UINT32_C(0x00010000)
typedef struct MiciusChatSession MiciusChatSession;
/* Borrows file/engine until close. One chat per file engine, single-thread calls.
 * While open, do not drive/reset/rebind the borrowed engine directly.
 * NULL id uses the file's directory, restoring it if already saved. Explicit id
 * must be an existing 32-digit lowercase hex context ID for this model.
 * The session holds an OS file lock. No stale lock removal is needed on crash.
 * Open failure preserves the engine. Directory-free file models are rejected. */
MiciusChatSession* Micius_OpenChatV1(MiciusFileModel*,const char* session_id,char* error,size_t capacity);
void Micius_CloseChatV1(MiciusChatSession*);
const char* Micius_ChatDirectoryV1(const MiciusChatSession*);
/* Callback bytes can split UTF-8 characters. Return 0 to cancel this turn.
 * A completed reply is committed in memory; explicitly save it to disk.
 * Reload sampling.conf happens before any forward. Same seed preserves RNG;
 * a changed seed restarts it. Cancellation/errors do not commit RNG/history;
 * partial engine KV is rebuilt from committed history at the next reply/save.
 * The cancelled result's context_length may reflect that partial engine state.
 * Close does not save. Save before close to retain the committed conversation. */
typedef int (*MiciusChatEmitV1)(const char* bytes,size_t length,void* user);
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    uint32_t finish; /* 0=eot, 1=reply limit, 2=context limit, 3=cancelled */
    uint32_t tokens;
    uint32_t context_length;
} MiciusChatResultV1;
int Micius_ChatReplyV1(MiciusChatSession*,const char* text,size_t length,
    uint32_t max_new_override,MiciusChatEmitV1,void* user,MiciusChatResultV1*,char* error,size_t capacity);
/* The only cross-thread operation: request cancellation, not parallel inference. */
void Micius_CancelChatV1(MiciusChatSession*);
int Micius_ResetChatV1(MiciusChatSession*,char* error,size_t capacity);
int Micius_SaveChatV1(MiciusChatSession*,char* error,size_t capacity);
int Micius_ReloadChatConfigV1(MiciusChatSession*,char* error,size_t capacity);
#ifdef __cplusplus
}
#endif
#endif

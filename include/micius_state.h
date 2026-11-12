#ifndef MICIUS_STATE_H
#define MICIUS_STATE_H
#include "micius_runtime.h"
#ifdef __cplusplus
extern "C" {
#endif
#define MICIUS_STATE_ABI_V1 UINT32_C(0x00010000)
/* Raw engine state, no pointers/weights/scratch. Opaque, versioned, little-endian.
 * Only the active K/V prefix and last logits are stored. The destination must
 * already have the SAME tensor model bound. The session module verifies GGUF
 * content identity; raw API clients must enforce identity themselves.
 * No concurrent inference/rebinding is allowed. Import validates the complete
 * buffer before modifying the engine. A failed import preserves live state.
 * Export: 1=success, 0=capacity query/short buffer, -1=error. */
const char* Micius_StateContractV1(void);
int Micius_ExportStateV1(MiciusContext*, void* out, size_t capacity, size_t* required);
int Micius_ImportStateV1(MiciusContext*, const void* data, size_t size, uint32_t expected_sequence_length);
#ifdef __cplusplus
}
#endif
#endif

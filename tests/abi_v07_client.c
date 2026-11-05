/* Compiled against the archived v0.7 public header. This object is also
 * linked unchanged to both installed profiles in the delivery audit. */
#include "micius_runtime.h"
#include <stddef.h>
#include <stdio.h>

_Static_assert(MICIUS_ABI_VERSION == 0x00060003, "V1 version");
_Static_assert(sizeof(MiciusConfig) == 56 && offsetof(MiciusConfig, rope_dimension_count) == 52, "V1 config");
_Static_assert(sizeof(MiciusQ8Block32Scale) == 16 && offsetof(MiciusQ8Block32Scale,padding) == 8, "scale ABI");
#if UINTPTR_MAX == UINT64_MAX
_Static_assert(sizeof(MiciusTensorEntryV1) == 80 && offsetof(MiciusTensorEntryV1,flags) == 72, "64-bit tensor ABI");
#endif
int main(void)
{
    MiciusConfig c;
    if (!Micius_InitConfig(&c,sizeof(c))) return 1;
    c.hidden_dim = c.head_dim = c.intermediate_dim = 32;
    c.num_layers = c.num_heads = c.num_kv_heads = 1;
    c.rope_dimension_count = c.head_dim;
    c.max_seq_len = 8;
    c.vocab_size = 8;
    if (Micius_CreateEngineV1(&c,sizeof(c)-1,MICIUS_ABI_VERSION) ||
        Micius_CreateEngineV1(&c,sizeof(c),MICIUS_ABI_VERSION+1)) return 2;
    MiciusContext* ctx = Micius_CreateEngineV1(&c,sizeof(c),MICIUS_ABI_VERSION);
    if (!ctx) return 3;
    int ok = Micius_GetWorkerCount(ctx)==0 && Micius_GetSequenceLengthV1(ctx)==0 &&
             Micius_PrefillTokenV1(ctx,1)==MICIUS_STATE_NOT_READY;
    Micius_DestroyEngine(ctx);
    if (!ok) return 4;
    puts("v0.7 C object: PASS");
    return 0;
}

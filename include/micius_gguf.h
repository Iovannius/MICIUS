#ifndef MICIUS_GGUF_H
#define MICIUS_GGUF_H

#include "micius_runtime.h"

#ifdef __cplusplus
extern "C" {
#endif

/* Static architecture module contract. These versions are independent of V1. */
#define MICIUS_GGUF_PARSER_ABI UINT32_C(0x00010000)
#define MICIUS_GGUF_FILE_ABI   UINT32_C(0x00010000)
#define MICIUS_GGUF_SPLIT_HALF_ROPE UINT32_C(1)
#define MICIUS_GGUF_QKV_BIAS        UINT32_C(2)

/* All metadata pointers are borrowed for the describe call. Getters return
 * 1 on success, 0 for an absent key, -1 for a present key of another type. */
typedef struct {
    const void* reader;
    uint32_t vocab_size;
    int (*get_uint)(const void*, const char*, uint64_t*);
    int (*get_float)(const void*, const char*, double*);
    int (*get_string)(const void*, const char*, const char**);
} MiciusGGUFMetadataV1;

typedef struct {
    MiciusConfig config;
    uint32_t features;
} MiciusGGUFPlanV1;

typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    const char* architecture; /* exact general.architecture */
    int (*describe)(const MiciusGGUFMetadataV1*, MiciusGGUFPlanV1*);
    /* Return the runtime tensor name (static string or the input), or NULL
     * for unsupported tensors. The host copies it before the next call. */
    const char* (*map_tensor)(const char* source_name, const MiciusGGUFPlanV1*);
} MiciusGGUFParserV1;

/* Each factory belongs to its own optional static library. */
const MiciusGGUFParserV1* Micius_LlamaParserV1(void);
const MiciusGGUFParserV1* Micius_Qwen2ParserV1(void);

typedef struct MiciusFileModel MiciusFileModel;
typedef struct {
    uint32_t abi_version;
    uint32_t struct_size;
    const char* path;       /* UTF-8 GGUF path */
    const char* cache_root; /* UTF-8, NULL for pure in-memory operation */
    const MiciusGGUFParserV1* const* parsers;
    uint32_t parser_count;
    uint32_t context_length; /* 0 uses the model limit, otherwise 1..limit */
} MiciusGGUFOpenOptionsV1;

/* On failure returns NULL and writes a NUL-terminated diagnostic if the
 * caller supplies a nonempty buffer. No exception crosses this boundary. */
MiciusFileModel* Micius_OpenGGUFV1(const MiciusGGUFOpenOptionsV1* options,
                                    char* error, size_t error_capacity);
void Micius_CloseGGUFV1(MiciusFileModel* model);
/* Borrowed until close. Use the engine for inference, KV inspection and reset.
 * The file model owns its binding. Close it with Micius_CloseGGUFV1. */
MiciusContext* Micius_FileEngineV1(const MiciusFileModel* model);
const MiciusConfig* Micius_FileConfigV1(const MiciusFileModel* model);
const char* Micius_FileArchitectureV1(const MiciusFileModel* model);
const char* Micius_FileContextDirectoryV1(const MiciusFileModel* model);
/* SHA-256 of the complete GGUF content, borrowed until close. */
const char* Micius_FileIdentityV1(const MiciusFileModel* model);

#ifdef __cplusplus
}
#endif
#endif

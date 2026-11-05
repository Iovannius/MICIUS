#ifndef MICIUS_TEXT_H
#define MICIUS_TEXT_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define MICIUS_TEXT_ABI_V1 UINT32_C(0x00010000)
typedef struct MiciusTokenizer MiciusTokenizer;
/* Reads GGUF metadata only. The returned tokenizer owns its vocabulary. */
MiciusTokenizer* Micius_OpenTokenizerV1(const char* utf8_path, char* error, size_t capacity);
void Micius_CloseTokenizerV1(MiciusTokenizer* tokenizer);
/* Encode ordinary UTF-8, with no implicit BOS/EOS or trusted control tokens.
 * Returns 1 on success, 0 for insufficient capacity (required is set), -1 on
 * an invalid request. NULL output with 0 capacity is a supported size query.
 * A failed or short call never writes a partial token sequence. */
int Micius_EncodeTextV1(const MiciusTokenizer* tokenizer, const char* utf8, size_t length,
                        uint32_t* ids, size_t capacity, size_t* required);
/* Ordinary output bytes, possibly part of a UTF-8 character, borrowed until
 * tokenizer close. Control/unused/out-of-range IDs return 0. */
int Micius_TokenBytesV1(const MiciusTokenizer* tokenizer, uint32_t id,
                        const char** bytes, size_t* length);
#ifdef __cplusplus
}
#endif
#endif

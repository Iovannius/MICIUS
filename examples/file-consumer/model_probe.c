/* External C client. Input is 1..2048 little-endian u32 token IDs, output is
 * one complete F32 logits row per ID. Reset/prefill replay is kept separate. */
#include "micius_gguf.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int check(MiciusFileModel* model, const uint32_t* ids, size_t n, FILE* output)
{
    MiciusContext* engine = Micius_FileEngineV1(model);
    const MiciusConfig* config = Micius_FileConfigV1(model);
    if (!engine || !config || config->max_seq_len != n || Micius_GetWorkerCount(engine) != 0) return 0;
    const size_t vocab = config->vocab_size;
    float* row = (float*)calloc(vocab, sizeof(float));
    float* replay = (float*)calloc(vocab, sizeof(float));
    int ok = row && replay;
    uint32_t prediction = 0;
    for (size_t i = 0; ok && i < n; ++i) {
        ok = Micius_DecodeOneV1(engine,ids[i],&prediction) == MICIUS_STATE_YIELD_TOKEN &&
             Micius_GetSequenceLengthV1(engine) == i+1 &&
             Micius_CopyLastLogitsV1(engine,row,vocab);
        if (!ok) break;
        uint32_t best = 0;
        for (uint32_t k = 0; k < vocab; ++k) {
            if (!isfinite(row[k])) { ok = 0; break; }
            if (row[k] > row[best]) best = k;
        }
        ok = ok && best == prediction && fwrite(row,sizeof(float),vocab,output) == vocab;
    }
    uint32_t sentinel = UINT32_MAX;
    ok = ok && Micius_DecodeOneV1(engine,ids[0],&sentinel) == MICIUS_STATE_CONTEXT_FULL &&
         sentinel == UINT32_MAX && Micius_GetSequenceLengthV1(engine) == n &&
         Micius_ResetEngineV1(engine) && Micius_GetSequenceLengthV1(engine) == 0 &&
         !Micius_CopyLastLogitsV1(engine,replay,vocab);
    for (size_t i = 0; ok && i+1 < n; ++i)
        ok = Micius_PrefillTokenV1(engine,ids[i]) == MICIUS_STATE_TOKEN_ACCEPTED;
    ok = ok && Micius_DecodeOneV1(engine,ids[n-1],&sentinel) == MICIUS_STATE_YIELD_TOKEN &&
         sentinel == prediction && Micius_CopyLastLogitsV1(engine,replay,vocab) &&
         memcmp(row,replay,vocab*sizeof(float)) == 0;
    free(row);
    free(replay);
    return ok;
}

int main(int argc, char** argv)
{
    if (argc != 5) {
        fprintf(stderr,"usage: micius_model_probe MODEL CACHE_ROOT INPUT_U32 OUTPUT_F32\n");
        return 2;
    }
    FILE* input = fopen(argv[3],"rb");
    if (!input) return 2;
    uint32_t ids[2048];
    size_t bytes = fread(ids,1,sizeof(ids),input);
    int valid = bytes && bytes%4 == 0 && fgetc(input)==EOF && !ferror(input);
    fclose(input);
    if (!valid) { fprintf(stderr,"invalid token file\n"); return 2; }
    const MiciusGGUFParserV1* parsers[] = {
#ifdef LINK_LLAMA
        Micius_LlamaParserV1(),
#endif
#ifdef LINK_QWEN2
        Micius_Qwen2ParserV1(),
#endif
    };
    const uint32_t steps = (uint32_t)(bytes/sizeof(ids[0]));
    MiciusGGUFOpenOptionsV1 options = {MICIUS_GGUF_FILE_ABI,sizeof(options),argv[1],argv[2],
        parsers,(uint32_t)(sizeof(parsers)/sizeof(parsers[0])),steps};
    char diagnostic[512] = {0};
    MiciusFileModel* model = Micius_OpenGGUFV1(&options,diagnostic,sizeof(diagnostic));
    if (!model) { fprintf(stderr,"%s\n",diagnostic); return 1; }
    printf("architecture=%s\ncontext_directory=%s\n",Micius_FileArchitectureV1(model),Micius_FileContextDirectoryV1(model));
    FILE* out = fopen(argv[4],"wb");
    int ok = out && check(model,ids,steps,out);
    if (out && fclose(out) != 0) ok = 0;
    if (!ok) fprintf(stderr,"model functional check failed: %s\n",Micius_GetLastError());
    Micius_CloseGGUFV1(model);
    if (ok) printf("steps=%u\nreset_replay=PASS\n",steps);
    return ok ? 0 : 1;
}

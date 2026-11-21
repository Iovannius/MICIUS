#include "DecoderParser.h"

namespace {
int Describe(const MiciusGGUFMetadataV1* view, MiciusGGUFPlanV1* plan) noexcept
{
    try {
        return micius::gguf::DescribeDecoder(view, plan, "llama", MICIUS_GGUF_ROPE_FACTORS);
    } catch (...) {
        return 0;
    }
}
const MiciusGGUFParserV1 parser{
    MICIUS_GGUF_PARSER_ABI, sizeof(MiciusGGUFParserV1), "llama",
    Describe, micius::gguf::StandardTensor
};
}
extern "C" const MiciusGGUFParserV1* Micius_LlamaParserV1(void) { return &parser; }

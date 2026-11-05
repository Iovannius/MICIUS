#include "DecoderParser.h"

namespace {
int Describe(const MiciusGGUFMetadataV1* view, MiciusGGUFPlanV1* plan) noexcept
{
    try {
        return micius::gguf::DescribeDecoder(view, plan, "qwen2", MICIUS_GGUF_SPLIT_HALF_ROPE | MICIUS_GGUF_QKV_BIAS);
    } catch (...) {
        return 0;
    }
}
const MiciusGGUFParserV1 parser{
    MICIUS_GGUF_PARSER_ABI, sizeof(MiciusGGUFParserV1), "qwen2",
    Describe, micius::gguf::StandardTensor
};
}
extern "C" const MiciusGGUFParserV1* Micius_Qwen2ParserV1(void) { return &parser; }

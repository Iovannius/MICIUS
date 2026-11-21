#include "DecoderParser.h"

namespace {
int Describe(const MiciusGGUFMetadataV1* view, MiciusGGUFPlanV1* plan) noexcept
{
    try {
        return micius::gguf::DescribeDecoder(view, plan, "qwen3", MICIUS_GGUF_SPLIT_HALF_ROPE | MICIUS_GGUF_QK_NORM);
    } catch (...) {
        return 0;
    }
}
const MiciusGGUFParserV1 parser{
    MICIUS_GGUF_PARSER_ABI, sizeof(MiciusGGUFParserV1), "qwen3",
    Describe, micius::gguf::StandardTensor
};
}
extern "C" const MiciusGGUFParserV1* Micius_Qwen3ParserV1(void) { return &parser; }

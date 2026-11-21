#pragma once
#include "quant/WeightFormat.h"
#include "micius_runtime.h"
#include <vector>

namespace micius::internal {
struct FileLayer {
    quant::Matrix q,k,v,o,gate,up,down;
    const float* norm_attn{};const float* norm_ffn{};
    const float* qb{};const float* kb{};const float* vb{};
    const float* qnorm{};const float* knorm{};
};
struct FileWeights {
    std::vector<FileLayer> layers;
    quant::Matrix embedding, output;
    const float* final_norm{};
    const float* rope_factors{};
    bool split_half{},qk_norm{};
};
MiciusContext* CreateFileEngine(const MiciusConfig& config) noexcept;
bool BindFileWeights(MiciusContext*,FileWeights&&) noexcept;
bool UsesFileWeights(const MiciusContext*) noexcept;
}

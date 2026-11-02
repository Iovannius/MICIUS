#pragma once
#include "micius_runtime.h"
#include <cstdint>
#include <span>
#include <string>
#include <vector>

namespace micius::i4 {
struct Score {
    double nll=0,logsumexp=0,target_logit=0;
    uint32_t top1=0;
};
bool ScoreLogits(std::span<const float>,uint32_t target,Score&) noexcept;
std::vector<std::vector<uint32_t>> ReadTokens(const std::string&,uint32_t vocab,uint32_t context);
// Uses the existing model-owned context and public V1 reset/prefill/logits API.
// Tokens are fixed by the caller; a prediction is never fed back as an input.
int EvaluateTokens(MiciusContext*,const MiciusConfig&,const std::string& input,const std::string& output) noexcept;
} // namespace micius::i4

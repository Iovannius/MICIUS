#pragma once
#include "micius_runtime.h"

namespace micius::internal {
struct ProjectionSemantics {
    bool split_half_rope;
    bool qkv_bias;
};
bool BindTypedTensors(MiciusContext*, const MiciusTensorEntryV1*, uint32_t, size_t,
                      uint32_t abi_version, ProjectionSemantics);
}

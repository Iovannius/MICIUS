#pragma once
#include "micius_runtime.h"
#include <immintrin.h>
#include <algorithm>
#include <cstddef>
#include <vector>

namespace micius::i22 {
struct ReleaseStats { size_t buffers=0, bytes=0; };
// Private Shell-only operation after a successful V1 bind. The runtime copies
// every consumed public scale into its own validated descriptor allocation.
// Do not apply this operation to raw weights/F32 tensors, which remain borrowed.
// Validate ownership before any free; erase each owner to prevent double free
// during reload, failure cleanup or destruction. No allocations on this path.
inline bool ReleasePublicScales(std::vector<void*>& owners,
    const std::vector<MiciusTensorEntryV1>& entries, ReleaseStats& stats) noexcept {
    stats={};
    for(size_t i=0;i<entries.size();++i) {
        const auto& e=entries[i];
        if(e.data_type!=MICIUS_TENSOR_Q8_32_SCALE) continue;
        if(!e.tensor_data || !e.byte_size ||
           std::count(owners.begin(),owners.end(),e.tensor_data)!=1) return false;
        for(size_t j=0;j<entries.size();++j)
            if(j!=i && entries[j].tensor_data==e.tensor_data) return false;
        if(e.byte_size>SIZE_MAX-stats.bytes) return false;
        stats.bytes+=static_cast<size_t>(e.byte_size);++stats.buffers;
    }
    for(const auto& e:entries) {
        if(e.data_type!=MICIUS_TENSOR_Q8_32_SCALE) continue;
        auto it=std::find(owners.begin(),owners.end(),e.tensor_data);
        _mm_free(*it);owners.erase(it);
    }
    return true;
}
}

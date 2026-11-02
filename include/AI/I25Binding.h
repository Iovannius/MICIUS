#pragma once
#include "micius_runtime.h"
#include <cstddef>
#include <cstdint>
#include <span>
namespace micius::i25 {
// Internal Shell/benchmark bridge. No change to the public C ABI.
struct RetiredWeight { uintptr_t address=0; size_t bytes=0; };
struct Stats {
    bool active=false;
    size_t matrices=0, payload_bytes=0, capacity_bytes=0, scratch_bytes=0;
    size_t retired_weight_bytes=0, retired_weight_buffers=0;
    size_t legacy_consumer_pointers=0, legacy_scale_buffers=0;
};
Stats BindingStats(const MiciusContext*) noexcept;
std::span<const RetiredWeight> RetirableWeights(const MiciusContext*) noexcept;
}

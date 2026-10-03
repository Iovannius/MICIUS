#pragma once

#include "micius_runtime.h"

#include <cstddef>
#include <cstring>

namespace micius::internal {

// The caller must first validate every public descriptor. This helper performs
// no arithmetic conversion: each accepted binary32 scale is copied byte-for-byte
// into the private compact stream.
inline void CopyCompactScaleBits(
    const MiciusQ8Block32Scale* source,
    float* destination,
    size_t count) noexcept
{
    for (size_t i = 0; i < count; ++i) {
        std::memcpy(destination + i, &source[i].scale, sizeof(float));
    }
}

} // namespace micius::internal

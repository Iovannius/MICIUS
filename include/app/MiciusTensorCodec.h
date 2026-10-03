#pragma once

#include "../micius_runtime.h"

#include <cstddef>
#include <cstdint>

namespace MiciusTensorCodec {

float DecodeFloat16(uint16_t value) noexcept;

bool DequantizeGGUFQ8_0ToF32(
    const void* source,
    size_t source_bytes,
    size_t rows,
    size_t columns,
    float* output,
    size_t output_count) noexcept;

bool ConvertGGUFQ8_0ToQ8_32(
    const void* source,
    size_t source_bytes,
    size_t rows,
    size_t columns,
    int8_t* output_weights,
    size_t output_weight_count,
    MiciusQ8Block32Scale* output_scales,
    size_t output_scale_count) noexcept;

} // namespace MiciusTensorCodec

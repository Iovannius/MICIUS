#include "MiciusTensorCodec.h"

#include <cmath>
#include <cstring>
#include <limits>

namespace {

#pragma pack(push, 1)
struct GGUFBlockQ8_0 {
    uint16_t scale_f16;
    int8_t quantized[32];
};
#pragma pack(pop)

static_assert(sizeof(GGUFBlockQ8_0) == 34);

bool CheckedProduct(size_t lhs, size_t rhs, size_t* result) noexcept
{
    if (!result || (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)) {
        return false;
    }
    *result = lhs * rhs;
    return true;
}

bool ValidateLayout(
    size_t source_bytes,
    size_t rows,
    size_t columns,
    size_t block_width,
    size_t* element_count,
    size_t* source_block_count) noexcept
{
    if (!element_count || !source_block_count || rows == 0 || columns == 0 ||
        columns % block_width != 0 ||
        !CheckedProduct(rows, columns, element_count) ||
        !CheckedProduct(rows, columns / 32U, source_block_count)) {
        return false;
    }
    size_t expected_source_bytes = 0;
    return CheckedProduct(
               *source_block_count,
               sizeof(GGUFBlockQ8_0),
               &expected_source_bytes) &&
           source_bytes == expected_source_bytes;
}

} // namespace

namespace MiciusTensorCodec {

float DecodeFloat16(uint16_t value) noexcept
{
    const uint32_t sign = static_cast<uint32_t>(value & 0x8000U) << 16U;
    uint32_t exponent = (value >> 10U) & 0x1FU;
    uint32_t mantissa = value & 0x03FFU;
    uint32_t bits = 0;

    if (exponent == 0) {
        if (mantissa == 0) {
            bits = sign;
        } else {
            int32_t unbiased_exponent = -14;
            while ((mantissa & 0x0400U) == 0) {
                mantissa <<= 1U;
                --unbiased_exponent;
            }
            mantissa &= 0x03FFU;
            bits = sign |
                   (static_cast<uint32_t>(unbiased_exponent + 127) << 23U) |
                   (mantissa << 13U);
        }
    } else if (exponent == 0x1FU) {
        bits = sign | 0x7F800000U | (mantissa << 13U);
    } else {
        exponent += 127U - 15U;
        bits = sign | (exponent << 23U) | (mantissa << 13U);
    }

    float result = 0.0f;
    std::memcpy(&result, &bits, sizeof(result));
    return result;
}

bool DequantizeGGUFQ8_0ToF32(
    const void* source,
    size_t source_bytes,
    size_t rows,
    size_t columns,
    float* output,
    size_t output_count) noexcept
{
    size_t element_count = 0;
    size_t source_block_count = 0;
    if (!source || !output ||
        !ValidateLayout(
            source_bytes,
            rows,
            columns,
            32U,
            &element_count,
            &source_block_count) ||
        output_count < element_count) {
        return false;
    }

    const auto* blocks = static_cast<const GGUFBlockQ8_0*>(source);
    for (size_t block = 0; block < source_block_count; ++block) {
        const float scale = DecodeFloat16(blocks[block].scale_f16);
        if (!std::isfinite(scale)) return false;
        for (size_t i = 0; i < 32U; ++i) {
            const float value = static_cast<float>(blocks[block].quantized[i]) * scale;
            if (!std::isfinite(value)) return false;
            output[block * 32U + i] = value;
        }
    }
    return true;
}

bool ConvertGGUFQ8_0ToQ8_32(
    const void* source,
    size_t source_bytes,
    size_t rows,
    size_t columns,
    int8_t* output_weights,
    size_t output_weight_count,
    MiciusQ8Block32Scale* output_scales,
    size_t output_scale_count) noexcept
{
    size_t element_count = 0;
    size_t source_block_count = 0;
    if (!source || !output_weights || !output_scales ||
        !ValidateLayout(
            source_bytes,
            rows,
            columns,
            32U,
            &element_count,
            &source_block_count) ||
        output_weight_count < element_count) {
        return false;
    }

    size_t target_block_count = 0;
    if (!CheckedProduct(rows, columns / 32U, &target_block_count) ||
        output_scale_count < target_block_count ||
        source_block_count != target_block_count) {
        return false;
    }

    const auto* blocks = static_cast<const GGUFBlockQ8_0*>(source);
    for (size_t block = 0; block < target_block_count; ++block) {
        const float scale = DecodeFloat16(blocks[block].scale_f16);
        if (!std::isfinite(scale)) return false;
        output_scales[block] = MiciusQ8Block32Scale{scale, 0, {0, 0}};
        std::memcpy(
            output_weights + block * 32U,
            blocks[block].quantized,
            sizeof(blocks[block].quantized));
    }
    return true;
}

} // namespace MiciusTensorCodec

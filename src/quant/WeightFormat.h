#pragma once
#include <cstddef>
#include <cstdint>

namespace micius::quant {
// GGML file type identifiers; Q4_K_M is a file recipe, not a tensor type.
enum class Type : uint32_t { F32=0, F16=1, Q8_0=8, Q4_K=12, Q6_K=14 };
struct Layout { size_t elements, bytes; };
Layout Block(Type type) noexcept;
size_t RowBytes(Type type, size_t columns) noexcept;
bool Decode(Type type, const void* source, size_t bytes, float* out, size_t elements) noexcept;
struct Matrix {
    const void* data{};
    size_t bytes{};
    uint32_t rows{}, cols{};
    Type type{};
    bool Valid() const noexcept;
    bool Row(uint32_t row, float* out) const noexcept;
    // Bounded block scratch, unchanged autoregressive decoder. No Q4 -> Q8 conversion.
    bool Multiply(const float* x, float* y) const noexcept;
};
}

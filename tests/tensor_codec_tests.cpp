#include "micius_runtime.h"
#include "MiciusTensorCodec.h"

#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <vector>

#if defined(NDEBUG)
#error "tensor_codec_tests requires assert() checks; build this target without NDEBUG"
#endif

namespace {

#pragma pack(push, 1)
struct TestGGUFBlockQ8_0 {
    uint16_t scale_f16;
    int8_t quantized[32];
};
#pragma pack(pop)

static_assert(sizeof(TestGGUFBlockQ8_0) == 34);

void FillBlock(TestGGUFBlockQ8_0* block, uint16_t scale, int8_t value)
{
    assert(block != nullptr);
    block->scale_f16 = scale;
    for (int8_t& element : block->quantized) element = value;
}

void TestFloat16Decode()
{
    using MiciusTensorCodec::DecodeFloat16;
    assert(DecodeFloat16(0x0000U) == 0.0f);
    assert(std::signbit(DecodeFloat16(0x8000U)));
    assert(DecodeFloat16(0x3C00U) == 1.0f);
    assert(DecodeFloat16(0xC000U) == -2.0f);
    assert(DecodeFloat16(0x0001U) == std::ldexp(1.0f, -24));
    assert(std::isinf(DecodeFloat16(0x7C00U)));
    assert(std::isnan(DecodeFloat16(0x7E00U)));
}

void TestQ8DequantizationContract()
{
    TestGGUFBlockQ8_0 source{};
    FillBlock(&source, 0x3800U, int8_t{-4}); // 0.5 * -4 == -2
    std::vector<float> output(32, 0.0f);
    assert(MiciusTensorCodec::DequantizeGGUFQ8_0ToF32(
        &source,
        sizeof(source),
        1,
        32,
        output.data(),
        output.size()));
    for (float value : output) assert(value == -2.0f);

    assert(!MiciusTensorCodec::DequantizeGGUFQ8_0ToF32(
        &source,
        sizeof(source) - 1U,
        1,
        32,
        output.data(),
        output.size()));
    source.scale_f16 = 0x7C00U;
    assert(!MiciusTensorCodec::DequantizeGGUFQ8_0ToF32(
        &source,
        sizeof(source),
        1,
        32,
        output.data(),
        output.size()));
}

void TestQ8ConversionRowsAndCapacity()
{
    constexpr size_t kRows = 2;
    constexpr size_t kColumns = 128;
    std::vector<TestGGUFBlockQ8_0> source(kRows * kColumns / 32U);
    const uint16_t scales[] = {
        0x3C00U, 0x4000U, 0x4400U, 0x4800U,
        0x4C00U, 0x5000U, 0x5400U, 0x5800U};
    for (size_t i = 0; i < source.size(); ++i) {
        FillBlock(&source[i], scales[i], int8_t{1});
    }

    std::vector<int8_t> weights(kRows * kColumns, int8_t{0});
    std::vector<MiciusQ8Block32Scale> descriptors(kRows * kColumns / 32U);
    assert(MiciusTensorCodec::ConvertGGUFQ8_0ToQ8_32(
        source.data(),
        source.size() * sizeof(TestGGUFBlockQ8_0),
        kRows,
        kColumns,
        weights.data(),
        weights.size(),
        descriptors.data(),
        descriptors.size()));

    const float expected_scales[] = {1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f, 64.0f, 128.0f};
    for (size_t i = 0; i < descriptors.size(); ++i) {
        assert(descriptors[i].scale == expected_scales[i]);
        assert(descriptors[i].zero_point == 0);
        assert(descriptors[i].padding[0] == 0);
        assert(descriptors[i].padding[1] == 0);
    }
    assert(weights[0] == 1);
    assert(weights[32] == 1);
    assert(weights[64] == 1);
    assert(weights[96] == 1);
    assert(weights[128] == 1);
    assert(weights[160] == 1);

    assert(!MiciusTensorCodec::ConvertGGUFQ8_0ToQ8_32(
        source.data(),
        source.size() * sizeof(TestGGUFBlockQ8_0),
        kRows,
        kColumns,
        weights.data(),
        weights.size() - 1U,
        descriptors.data(),
        descriptors.size()));
    assert(!MiciusTensorCodec::ConvertGGUFQ8_0ToQ8_32(
        source.data(),
        source.size() * sizeof(TestGGUFBlockQ8_0),
        kRows,
        48,
        weights.data(),
        weights.size(),
        descriptors.data(),
        descriptors.size()));
}

void TestQ8ConversionIsBitExact()
{
    TestGGUFBlockQ8_0 source[2]{};
    FillBlock(&source[0], 0xBC00U, int8_t{-128}); // -1 * -128 == +128
    FillBlock(&source[1], 0x0000U, int8_t{42});
    int8_t weights[64]{};
    MiciusQ8Block32Scale descriptors[2]{};
    assert(MiciusTensorCodec::ConvertGGUFQ8_0ToQ8_32(
        source,
        sizeof(source),
        1,
        64,
        weights,
        64,
        descriptors,
        2));
    assert(descriptors[0].scale == -1.0f);
    assert(descriptors[1].scale == 0.0f);
    assert(weights[0] == int8_t{-128});
    assert(weights[31] == int8_t{-128});
    assert(weights[32] == int8_t{42});
    assert(weights[63] == int8_t{42});
}

void TestQ8ThirtyTwoBlockShapes()
{
    TestGGUFBlockQ8_0 source[3]{};
    FillBlock(&source[0], 0x3C00U, int8_t{3});
    FillBlock(&source[1], 0x4000U, int8_t{-5});
    FillBlock(&source[2], 0x3800U, int8_t{7});
    int8_t weights[96]{};
    MiciusQ8Block32Scale descriptors[3]{};
    assert(MiciusTensorCodec::ConvertGGUFQ8_0ToQ8_32(
        source,
        sizeof(source),
        1,
        96,
        weights,
        96,
        descriptors,
        3));
    assert(descriptors[0].scale == 1.0f);
    assert(descriptors[1].scale == 2.0f);
    assert(descriptors[2].scale == 0.5f);
    assert(weights[0] == 3);
    assert(weights[32] == -5);
    assert(weights[64] == 7);
}

} // namespace

int main()
{
    TestFloat16Decode();
    TestQ8DequantizationContract();
    TestQ8ConversionRowsAndCapacity();
    TestQ8ConversionIsBitExact();
    TestQ8ThirtyTwoBlockShapes();
    std::cout << "tensor codec tests passed\n";
    return 0;
}

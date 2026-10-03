#include "MiciusGGUFParser.h"

#include <cassert>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(NDEBUG)
#error "gguf_parser_tests requires assert() checks; build this target without NDEBUG"
#endif

namespace {

template <typename T>
void AppendScalar(std::vector<uint8_t>& bytes, const T& value)
{
    const auto* source = reinterpret_cast<const uint8_t*>(&value);
    bytes.insert(bytes.end(), source, source + sizeof(T));
}

void AppendString(std::vector<uint8_t>& bytes, const std::string& value)
{
    const uint64_t length = value.size();
    AppendScalar(bytes, length);
    bytes.insert(bytes.end(), value.begin(), value.end());
}

void AppendU32Metadata(
    std::vector<uint8_t>& bytes,
    const std::string& key,
    uint32_t value)
{
    AppendString(bytes, key);
    AppendScalar(bytes, static_cast<uint32_t>(GGUFValueType::UINT32));
    AppendScalar(bytes, value);
}

void AppendU64Metadata(
    std::vector<uint8_t>& bytes,
    const std::string& key,
    uint64_t value)
{
    AppendString(bytes, key);
    AppendScalar(bytes, static_cast<uint32_t>(GGUFValueType::UINT64));
    AppendScalar(bytes, value);
}

void AppendFloatMetadata(
    std::vector<uint8_t>& bytes,
    const std::string& key,
    float value)
{
    AppendString(bytes, key);
    AppendScalar(bytes, static_cast<uint32_t>(GGUFValueType::FLOAT32));
    AppendScalar(bytes, value);
}

void AppendStringMetadata(
    std::vector<uint8_t>& bytes,
    const std::string& key,
    const std::string& value)
{
    AppendString(bytes, key);
    AppendScalar(bytes, static_cast<uint32_t>(GGUFValueType::STRING));
    AppendString(bytes, value);
}

void AppendTokenizerTokens(std::vector<uint8_t>& bytes)
{
    AppendString(bytes, "tokenizer.ggml.tokens");
    AppendScalar(bytes, static_cast<uint32_t>(GGUFValueType::ARRAY));
    AppendScalar(bytes, static_cast<uint32_t>(GGUFValueType::STRING));
    AppendScalar(bytes, uint64_t{2});
    AppendString(bytes, "<unk>");
    AppendString(bytes, "token");
}

std::vector<uint8_t> MakeConfigFixture(
    uint32_t alignment,
    const std::string& rope_scaling = "",
    bool add_large_unrelated_uint64 = false,
    uint32_t sliding_window = 0,
    uint32_t key_length = 0,
    bool alignment_is_uint64 = false)
{
    const uint64_t metadata_count =
        12U +
        (rope_scaling.empty() ? 0U : 1U) +
        (add_large_unrelated_uint64 ? 1U : 0U) +
        (sliding_window == 0 ? 0U : 1U) +
        (key_length == 0 ? 0U : 1U);
    std::vector<uint8_t> bytes;
    AppendScalar(bytes, GGUF_MAGIC);
    AppendScalar(bytes, uint32_t{3});
    AppendScalar(bytes, uint64_t{0});
    AppendScalar(bytes, metadata_count);

    AppendStringMetadata(bytes, "general.architecture", "llama");
    AppendTokenizerTokens(bytes);
    AppendU32Metadata(bytes, "llama.embedding_length", 64);
    AppendU32Metadata(bytes, "llama.context_length", 8);
    AppendU32Metadata(bytes, "llama.block_count", 1);
    AppendU32Metadata(bytes, "llama.attention.head_count", 1);
    AppendU32Metadata(bytes, "llama.attention.head_count_kv", 1);
    AppendU32Metadata(bytes, "llama.feed_forward_length", 64);
    AppendU32Metadata(bytes, "llama.rope.dimension_count", 64);
    AppendFloatMetadata(bytes, "llama.attention.layer_norm_rms_epsilon", 1.0e-5f);
    AppendFloatMetadata(bytes, "llama.rope.freq_base", 10000.0f);
    if (alignment_is_uint64) {
        AppendU64Metadata(bytes, "general.alignment", alignment);
    } else {
        AppendU32Metadata(bytes, "general.alignment", alignment);
    }
    if (!rope_scaling.empty()) {
        AppendStringMetadata(bytes, "llama.rope.scaling.type", rope_scaling);
    }
    if (add_large_unrelated_uint64) {
        AppendU64Metadata(bytes, "general.parameter_count", UINT64_C(8000000000));
    }
    if (sliding_window != 0) {
        AppendU32Metadata(bytes, "llama.attention.sliding_window", sliding_window);
    }
    if (key_length != 0) {
        AppendU32Metadata(bytes, "llama.attention.key_length", key_length);
    }

    if (alignment != 0) {
        while (bytes.size() % alignment != 0) bytes.push_back(0);
    }
    return bytes;
}

std::filesystem::path WriteFixture(
    const std::string& name,
    const std::vector<uint8_t>& bytes)
{
    const auto nonce = std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path path =
        std::filesystem::temp_directory_path() /
        ("micius_" + name + "_" + std::to_string(nonce) + ".gguf");
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    assert(stream.is_open());
    stream.write(reinterpret_cast<const char*>(bytes.data()),
                 static_cast<std::streamsize>(bytes.size()));
    stream.close();
    return path;
}

template <typename Function>
void ExpectThrows(Function&& function)
{
    bool threw = false;
    try {
        function();
    } catch (const std::exception&) {
        threw = true;
    }
    assert(threw);
}

void TestValidConfigAndRepeatParse()
{
    const auto path = WriteFixture("valid", MakeConfigFixture(32));
    {
        GGUFParser parser(path.string());
        assert(parser.ParseHeaders());
        MiciusConfig config{};
        assert(parser.ExtractEngineConfig(config));
        assert(config.hidden_dim == 64);
        assert(config.vocab_size == 2);
        assert(config.num_layers == 1);
        assert(config.rope_dimension_count == 64);
        assert(config.rope_freq_base == 10000.0f);
        assert(config.rms_norm_epsilon == 1.0e-5f);

        assert(parser.ParseHeaders());
        assert(parser.GetVocab().size() == 2);
        assert(parser.GetTensors().empty());
    }
    std::filesystem::remove(path);
}

void TestUnsupportedVersionAndTruncation()
{
    std::vector<uint8_t> unsupported;
    AppendScalar(unsupported, GGUF_MAGIC);
    AppendScalar(unsupported, uint32_t{4});
    const auto unsupported_path = WriteFixture("version", unsupported);
    {
        GGUFParser parser(unsupported_path.string());
        assert(!parser.ParseHeaders());
    }
    std::filesystem::remove(unsupported_path);

    std::vector<uint8_t> truncated;
    AppendScalar(truncated, GGUF_MAGIC);
    const auto truncated_path = WriteFixture("truncated", truncated);
    {
        GGUFParser parser(truncated_path.string());
        ExpectThrows([&] { (void)parser.ParseHeaders(); });
    }
    std::filesystem::remove(truncated_path);
}

void TestAlignmentAndScalingRejection()
{
    const auto alignment_path = WriteFixture("alignment", MakeConfigFixture(0));
    {
        GGUFParser parser(alignment_path.string());
        ExpectThrows([&] { (void)parser.ParseHeaders(); });
    }
    std::filesystem::remove(alignment_path);

    const auto scaling_path = WriteFixture("scaling", MakeConfigFixture(32, "linear"));
    {
        GGUFParser parser(scaling_path.string());
        assert(parser.ParseHeaders());
        MiciusConfig config{};
        config.hidden_dim = 0xA5A5A5A5U;
        assert(!parser.ExtractEngineConfig(config));
        assert(config.hidden_dim == 0xA5A5A5A5U);
    }
    std::filesystem::remove(scaling_path);
}

void TestUint64MetadataAndArchitectureSubset()
{
    const auto large_metadata_path = WriteFixture(
        "large_u64",
        MakeConfigFixture(32, "", true, 0, 0, true));
    {
        GGUFParser parser(large_metadata_path.string());
        assert(parser.ParseHeaders());
        MiciusConfig config{};
        assert(parser.ExtractEngineConfig(config));
        assert(config.hidden_dim == 64);
    }
    std::filesystem::remove(large_metadata_path);

    const auto sliding_path = WriteFixture(
        "sliding_window",
        MakeConfigFixture(32, "", false, 4));
    {
        GGUFParser parser(sliding_path.string());
        assert(parser.ParseHeaders());
        MiciusConfig config{};
        assert(!parser.ExtractEngineConfig(config));
    }
    std::filesystem::remove(sliding_path);

    const auto key_length_path = WriteFixture(
        "key_length",
        MakeConfigFixture(32, "", false, 0, 32));
    {
        GGUFParser parser(key_length_path.string());
        assert(parser.ParseHeaders());
        MiciusConfig config{};
        assert(!parser.ExtractEngineConfig(config));
    }
    std::filesystem::remove(key_length_path);
}

void TestDuplicateMetadataRejection()
{
    std::vector<uint8_t> duplicate;
    AppendScalar(duplicate, GGUF_MAGIC);
    AppendScalar(duplicate, uint32_t{3});
    AppendScalar(duplicate, uint64_t{0});
    AppendScalar(duplicate, uint64_t{2});
    AppendU32Metadata(duplicate, "general.alignment", 32);
    AppendU32Metadata(duplicate, "general.alignment", 32);
    while (duplicate.size() % 32U != 0) duplicate.push_back(0);
    const auto path = WriteFixture("duplicate_metadata", duplicate);
    {
        GGUFParser parser(path.string());
        ExpectThrows([&] { (void)parser.ParseHeaders(); });
    }
    std::filesystem::remove(path);
}

void TestTensorRankAndRangeRejection()
{
    std::vector<uint8_t> rank;
    AppendScalar(rank, GGUF_MAGIC);
    AppendScalar(rank, uint32_t{3});
    AppendScalar(rank, uint64_t{1});
    AppendScalar(rank, uint64_t{0});
    AppendString(rank, "bad.weight");
    AppendScalar(rank, uint32_t{5});
    const auto rank_path = WriteFixture("rank", rank);
    {
        GGUFParser parser(rank_path.string());
        ExpectThrows([&] { (void)parser.ParseHeaders(); });
    }
    std::filesystem::remove(rank_path);

    std::vector<uint8_t> range;
    AppendScalar(range, GGUF_MAGIC);
    AppendScalar(range, uint32_t{3});
    AppendScalar(range, uint64_t{1});
    AppendScalar(range, uint64_t{1});
    AppendU32Metadata(range, "general.alignment", 32);
    AppendString(range, "output.weight");
    AppendScalar(range, uint32_t{2});
    AppendScalar(range, uint64_t{32});
    AppendScalar(range, uint64_t{1});
    AppendScalar(range, static_cast<uint32_t>(GGMLType::Q8_0));
    AppendScalar(range, uint64_t{0});
    while (range.size() % 32U != 0) range.push_back(0);
    const auto range_path = WriteFixture("range", range);
    {
        GGUFParser parser(range_path.string());
        ExpectThrows([&] { (void)parser.ParseHeaders(); });
    }
    std::filesystem::remove(range_path);
}

} // namespace

int main()
{
    TestValidConfigAndRepeatParse();
    TestUnsupportedVersionAndTruncation();
    TestAlignmentAndScalingRejection();
    TestUint64MetadataAndArchitectureSubset();
    TestTensorRankAndRangeRejection();
    TestDuplicateMetadataRejection();
    std::cout << "GGUF parser tests passed\n";
    return 0;
}

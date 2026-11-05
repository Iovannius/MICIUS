#include "MiciusGGUFParser.h"
#include "gguf/NativePath.h"

#include <algorithm>
#include <iostream>
#include <filesystem>
#include <cmath>
#include <cstring>
#include <limits>
#include <sstream>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
    #include <windows.h>
#else
    #include <sys/mman.h>
    #include <sys/stat.h>
    #include <fcntl.h>
    #include <unistd.h>
#endif

GGUFParser::GGUFParser(const std::string& path)
    : file_path(path), mmap_addr(nullptr), file_size(0), cursor(0),
      data_offset(0), parsed(false)
{
#ifdef _WIN32
    hFile = INVALID_HANDLE_VALUE;
    hMapping = NULL;
    
    hFile = CreateFileW(micius::gguf::NativePath(std::filesystem::u8path(path)).c_str(), GENERIC_READ, FILE_SHARE_READ, NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (hFile == INVALID_HANDLE_VALUE) throw std::runtime_error(std::string("Failed to open file: ") + path);
    
    LARGE_INTEGER li{};
    if (!GetFileSizeEx(hFile, &li) || li.QuadPart <= 0 ||
        static_cast<uint64_t>(li.QuadPart) > std::numeric_limits<size_t>::max()) {
        CloseHandle(hFile);
        hFile = INVALID_HANDLE_VALUE;
        throw std::runtime_error("Failed to determine a supported GGUF file size");
    }
    file_size = static_cast<size_t>(li.QuadPart);

    hMapping = CreateFileMappingA(hFile, NULL, PAGE_READONLY, 0, 0, NULL);
    if (!hMapping) {
        CloseHandle(hFile);
        hFile = INVALID_HANDLE_VALUE;
        throw std::runtime_error("Failed to map file");
    }

    mmap_addr = MapViewOfFile(hMapping, FILE_MAP_READ, 0, 0, 0);
    if (!mmap_addr) {
        if (hMapping) CloseHandle(hMapping);
        if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
        throw std::runtime_error("Failed to create map view");
    }
#else
    fd = -1;

    fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) throw std::runtime_error(std::string("Failed to open file: ") + path);

    struct stat st{};
    if (fstat(fd, &st) != 0 || st.st_size <= 0) {
        close(fd);
        fd = -1;
        throw std::runtime_error("Failed to determine a non-empty GGUF file size");
    }
    file_size = static_cast<size_t>(st.st_size);

    mmap_addr = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mmap_addr == MAP_FAILED) {
        mmap_addr = nullptr; 
        if (fd >= 0) close(fd);
        throw std::runtime_error("Failed to map file");
    }
#endif
}

GGUFParser::~GGUFParser() 
{
#ifdef _WIN32
    if (mmap_addr) UnmapViewOfFile(mmap_addr);
    if (hMapping) CloseHandle(hMapping);
    if (hFile != INVALID_HANDLE_VALUE) CloseHandle(hFile);
#else
    if (mmap_addr && mmap_addr != MAP_FAILED) munmap(mmap_addr, file_size);
    if (fd >= 0) close(fd);
#endif
}

void GGUFParser::ReadBytes(void* dst, size_t len) 
{
    if (!dst || cursor > file_size || len > file_size - cursor) {
        throw std::runtime_error("Unexpected EOF during GGUF parsing");
    }
    std::memcpy(dst, static_cast<const char*>(mmap_addr) + cursor, len);
    cursor += len;
}

void GGUFParser::SkipBytes(size_t len)
{
    if (cursor > file_size || len > file_size - cursor) {
        throw std::runtime_error("Unexpected EOF while skipping GGUF metadata");
    }
    cursor += len;
}

uint32_t GGUFParser::ReadU32()
{
    uint32_t v;
    ReadBytes(&v, 4);
    return v;
}

uint64_t GGUFParser::ReadU64()
{
    uint64_t v;
    ReadBytes(&v, 8);
    return v;
}

std::string GGUFParser::ReadString() 
{
    uint64_t len = ReadU64();
    if (len > std::numeric_limits<size_t>::max() ||
        cursor > file_size || static_cast<size_t>(len) > file_size - cursor) {
        throw std::runtime_error("Invalid string length in GGUF");
    }
    const size_t string_size = static_cast<size_t>(len);
    std::string s(static_cast<const char*>(mmap_addr) + cursor, string_size);
    cursor += string_size;
    return s;
}

//跳过无关的元数据递归下降避免拷贝
void GGUFParser::SkipValue(uint32_t type, uint32_t depth)
{
    if (depth > 16U) throw std::runtime_error("GGUF metadata nesting exceeds the safety limit");
    switch(type) {
        case 0: SkipBytes(1); break; // UINT8
        case 1: SkipBytes(1); break; // INT8
        case 2: SkipBytes(2); break; // UINT16
        case 3: SkipBytes(2); break; // INT16
        case 4: SkipBytes(4); break; // UINT32
        case 5: SkipBytes(4); break; // INT32
        case 6: SkipBytes(4); break; // FLOAT32
        case 7: SkipBytes(1); break; // BOOL
        case 8: {
            const auto bytes = ReadU64();
            if (bytes > SIZE_MAX) throw std::runtime_error("GGUF string exceeds addressable space");
            SkipBytes(static_cast<size_t>(bytes));
            break;
        }
        case 9: { // ARRAY
            uint32_t arr_type = ReadU32();
            uint64_t arr_len = ReadU64();
            if (arr_len > file_size) throw std::runtime_error("GGUF array length is implausible");
            for(uint64_t i = 0; i < arr_len; ++i) SkipValue(arr_type, depth + 1U);
            break;
        }
        case 10: SkipBytes(8); break; // UINT64
        case 11: SkipBytes(8); break; // INT64
        case 12: SkipBytes(8); break; // FLOAT64
        default: throw std::runtime_error("Unknown GGUF value type");
    }
}

bool GGUFParser::ParseHeaders() 
{
    cursor = 0;
    data_offset = 0;
    parsed = false;
    kv_strings.clear();
    kv_uint32s.clear();
    kv_uint64s.clear();
    kv_floats.clear();
    tensors.clear();
    vocab_strings.clear();
    merges_record = types_record = {};
    tokenizer_flags.clear();
    if (ReadU32() != GGUF_MAGIC) return false;
    
    uint32_t version = ReadU32();
    if (version < 2 || version > 3) return false;

    uint64_t tensor_count = ReadU64();
    uint64_t kv_count = ReadU64();
    if (tensor_count > file_size / 16U || kv_count > file_size / 8U) {
        throw std::runtime_error("GGUF entry counts exceed conservative file bounds");
    }

    metadata_names.clear();
    metadata_names.reserve(static_cast<size_t>(kv_count));

    //抓取字典KV Pairs
    for (uint64_t i = 0; i < kv_count; ++i) {
        std::string key = ReadString();
        if (key.empty() || key.find('\0') != std::string::npos || !metadata_names.insert(key).second) {
            throw std::runtime_error("GGUF contains a duplicate metadata key");
        }
        uint32_t type = ReadU32();

        if (type == 4) { // UINT32
            kv_uint32s[key] = ReadU32();
        } else if (type == 10) { // UINT64
            kv_uint64s[key] = ReadU64();
        } else if (type == 6) { // FLOAT32
            float value = 0.0f;
            ReadBytes(&value, sizeof(value));
            if (!std::isfinite(value)) throw std::runtime_error("GGUF float metadata is not finite");
            kv_floats[key] = value;
        } else if (type == 12) { // FLOAT64
            double value = 0.0;
            ReadBytes(&value, sizeof(value));
            if (!std::isfinite(value)) throw std::runtime_error("GGUF float metadata is not finite");
            kv_floats[key] = value;
        } else if (type == 8) { // STRING
            kv_strings[key] = ReadString();
            if (kv_strings[key].find('\0') != std::string::npos)
                throw std::runtime_error("Embedded NUL in GGUF metadata string");
        } else if (type == 9 && key == "tokenizer.ggml.tokens") { // ARRAY
            //获取Token字符串
            uint32_t arr_type = ReadU32();
            uint64_t arr_len = ReadU64();
            if (arr_len > std::numeric_limits<uint32_t>::max() ||
                arr_len > std::numeric_limits<size_t>::max()) {
                throw std::runtime_error("tokenizer vocabulary is too large");
            }
            if (arr_type != 8 || arr_len > (file_size - cursor) / 8)
                throw std::runtime_error("tokenizer vocabulary must be a bounded string array");
            
            vocab_strings.reserve(static_cast<size_t>(arr_len));
            for(uint64_t j = 0; j < arr_len; ++j) {
                vocab_strings.push_back(ReadString());
            }
        } else if (type == 7 && key.starts_with("tokenizer.ggml.")) {
            uint8_t flag;
            ReadBytes(&flag, sizeof(flag));
            tokenizer_flags.emplace(key, flag);
        } else if (type == 9 && (key == "tokenizer.ggml.merges" || key == "tokenizer.ggml.token_type")) {
            const size_t begin = cursor;
            SkipValue(type);
            auto& record = key == "tokenizer.ggml.merges" ? merges_record : types_record;
            record = {static_cast<const char*>(mmap_addr) + begin, cursor - begin};
        } else {
            SkipValue(type);
        }
    }

    std::unordered_set<std::string> tensor_names;
    tensor_names.reserve(static_cast<size_t>(tensor_count));

    //抓取Tensor张量偏移
    for (uint64_t i = 0; i < tensor_count; ++i) {
        GGUFTensorInfo info{};
        info.name = ReadString();
        if (info.name.empty() || info.name.find('\0') != std::string::npos || !tensor_names.insert(info.name).second) {
            throw std::runtime_error("GGUF tensor name is empty or duplicated");
        }
        info.n_dims = ReadU32();
        if (info.n_dims == 0 || info.n_dims > 4) {
            throw std::runtime_error("GGUF tensor rank is outside the supported range 1..4");
        }
        for (uint32_t d = 0; d < info.n_dims; ++d) info.dimensions[d] = ReadU64();
        info.type = static_cast<GGMLType>(ReadU32());
        info.offset = ReadU64(); //相对数据区块首地址的偏移

        uint64_t element_count = 1;
        for (uint32_t d = 0; d < info.n_dims; ++d) {
            if (info.dimensions[d] == 0 ||
                element_count > std::numeric_limits<uint64_t>::max() / info.dimensions[d]) {
                throw std::runtime_error("GGUF tensor element count overflows uint64_t");
            }
            element_count *= info.dimensions[d];
        }
        switch (info.type) {
            case GGMLType::F32:
                if (element_count > std::numeric_limits<uint64_t>::max() / 4U) {
                    throw std::runtime_error("F32 tensor byte size overflows uint64_t");
                }
                info.byte_size = element_count * 4U;
                break;
            case GGMLType::F16:
                if (element_count > std::numeric_limits<uint64_t>::max() / 2U) {
                    throw std::runtime_error("F16 tensor byte size overflows uint64_t");
                }
                info.byte_size = element_count * 2U;
                break;
            case GGMLType::Q8_0:
                if (element_count % 32U != 0 ||
                    element_count / 32U > std::numeric_limits<uint64_t>::max() / 34U) {
                    throw std::runtime_error("Q8_0 tensor is not block aligned or its byte size overflows");
                }
                info.byte_size = element_count / 32U * 34U;
                break;
            default:
                throw std::runtime_error("GGML tensor type is not supported by this checkpoint");
        }
        tensors.push_back(info);
    }
    
    //全局数据区块起始点对齐32字节界限
    uint64_t alignment_value = 32; //默认回退值
    auto it_align = kv_uint32s.find("general.alignment");
    if (it_align != kv_uint32s.end()) {
        alignment_value = it_align->second;
    } else {
        const auto it_align64 = kv_uint64s.find("general.alignment");
        if (it_align64 != kv_uint64s.end()) alignment_value = it_align64->second;
    }
    if (alignment_value > std::numeric_limits<size_t>::max()) {
        throw std::runtime_error("GGUF alignment exceeds the host address range");
    }
    const size_t alignment = static_cast<size_t>(alignment_value);
    if (alignment == 0 || (alignment & (alignment - 1U)) != 0) {
        throw std::runtime_error("GGUF alignment is not a non-zero power of two");
    }
    if (cursor > std::numeric_limits<size_t>::max() - (alignment - 1U)) {
        throw std::runtime_error("GGUF data offset alignment overflows size_t");
    }
    data_offset = (cursor + alignment - 1U) & ~(alignment - 1U);
    if (data_offset > file_size) {
        throw std::runtime_error("GGUF data region begins beyond the file boundary");
    }
    const uint64_t data_region_size = static_cast<uint64_t>(file_size - data_offset);
    std::vector<std::pair<uint64_t, uint64_t>> ranges;
    ranges.reserve(tensors.size());
    for (const auto& tensor : tensors) {
        if (tensor.offset % alignment != 0 || tensor.offset > data_region_size ||
            tensor.byte_size > data_region_size - tensor.offset) {
            throw std::runtime_error("GGUF tensor range is misaligned or outside the data region");
        }
        ranges.emplace_back(tensor.offset, tensor.offset + tensor.byte_size);
    }
    std::sort(ranges.begin(), ranges.end());
    for (size_t i = 1; i < ranges.size(); ++i) {
        if (ranges[i].first < ranges[i - 1].second) {
            throw std::runtime_error("GGUF tensor byte ranges overlap");
        }
    }
    cursor = data_offset;
    parsed = true;

    return true;
}

const std::string& GGUFParser::Architecture() const
{
    const auto it = kv_strings.find("general.architecture");
    if (!parsed || it == kv_strings.end() || it->second.empty())
        throw std::runtime_error("GGUF requires a string general.architecture");
    return it->second;
}

std::string_view GGUFParser::TextString(const char* key) const
{
    const auto it = kv_strings.find(key);
    if (it != kv_strings.end()) return it->second;
    if (metadata_names.contains(key)) throw std::runtime_error(std::string("Expected a GGUF string: ") + key);
    return {};
}

uint32_t GGUFParser::TextUint(const char* key) const
{
    const auto it = kv_uint32s.find(key);
    if (it == kv_uint32s.end()) throw std::runtime_error(std::string("Expected a GGUF uint32: ") + key);
    return it->second;
}

bool GGUFParser::TextFlag(const char* key) const
{
    const auto it = tokenizer_flags.find(key);
    if (it != tokenizer_flags.end()) {
        if (it->second > 1) throw std::runtime_error(std::string("Invalid tokenizer flag: ") + key);
        return it->second != 0;
    }
    if (metadata_names.contains(key)) throw std::runtime_error(std::string("Expected a GGUF boolean: ") + key);
    return false;
}

bool GGUFParser::Describe(const MiciusGGUFParserV1& module, MiciusGGUFPlanV1& plan) const
{
    if (!parsed || module.abi_version != MICIUS_GGUF_PARSER_ABI || module.struct_size < sizeof(module) ||
        !module.architecture || Architecture() != module.architecture || !module.describe || !module.map_tensor)
        return false;
    MiciusGGUFMetadataV1 view{};
    view.reader = this;
    view.vocab_size = static_cast<uint32_t>(vocab_strings.size());
    view.get_uint = [](const void* opaque, const char* k, uint64_t* out) {
        const auto& r = *static_cast<const GGUFParser*>(opaque);
        if (const auto it = r.kv_uint32s.find(k); it != r.kv_uint32s.end()) { *out = it->second; return 1; }
        if (const auto it = r.kv_uint64s.find(k); it != r.kv_uint64s.end()) { *out = it->second; return 1; }
        return r.metadata_names.contains(k) ? -1 : 0;
    };
    view.get_float = [](const void* opaque, const char* k, double* out) {
        const auto& r = *static_cast<const GGUFParser*>(opaque);
        if (const auto it = r.kv_floats.find(k); it != r.kv_floats.end()) { *out = it->second; return 1; }
        return r.metadata_names.contains(k) ? -1 : 0;
    };
    view.get_string = [](const void* opaque, const char* k, const char** out) {
        const auto& r = *static_cast<const GGUFParser*>(opaque);
        if (const auto it = r.kv_strings.find(k); it != r.kv_strings.end()) { *out = it->second.c_str(); return 1; }
        return r.metadata_names.contains(k) ? -1 : 0;
    };
    MiciusGGUFPlanV1 candidate{};
    if (module.describe(&view, &candidate) != 1 ||
        (candidate.features & ~(MICIUS_GGUF_QKV_BIAS | MICIUS_GGUF_SPLIT_HALF_ROPE))) return false;
    plan = candidate;
    return true;
}

void GGUFParser::MapTensors(const MiciusGGUFParserV1& module, const MiciusGGUFPlanV1& plan)
{
    std::unordered_set<std::string> canonical;
    for (auto& t : tensors) {
        const char* mapped = module.map_tensor(t.name.c_str(), &plan);
        if (!mapped || !*mapped || !canonical.emplace(mapped).second)
            throw std::runtime_error("Unsupported or duplicate tensor mapping: " + t.name);
        if (mapped != t.name.c_str()) t.name = mapped;
    }
}

const void* GGUFParser::GetTensorData(uint64_t offset) const noexcept
{
    if (!parsed || !mmap_addr || offset > file_size - data_offset) return nullptr;
    return static_cast<const char*>(mmap_addr) + data_offset + static_cast<size_t>(offset);
}

const void* GGUFParser::GetTensorDataBase() const 
{
    return parsed ? static_cast<const char*>(mmap_addr) + data_offset : nullptr;
}

void GGUFParser::PrintTopology() const 
{
    std::cout << "\n[Model Topology Discovered]\n";
    for (const auto& pair : kv_strings) {
        if (pair.first.find("architecture") != std::string::npos || pair.first.find("name") != std::string::npos) {
            std::cout << "  * " << pair.first << " : " << pair.second << "\n";
        }
    }
    std::cout << "  * Tensors Mapped   : " << tensors.size() << "\n";
    std::cout << "  * Memory Locked    : " << (file_size / 1024 / 1024) << " MB (Zero-Copy)\n\n";
}

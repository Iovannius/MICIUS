#pragma once

#include "micius_gguf.h"
#include <unordered_set>

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <stdexcept>
#include <string_view>

constexpr uint32_t GGUF_MAGIC = 0x46554747;

//GGUF支持的元数据
enum class GGUFValueType : uint32_t {
    UINT8 = 0, INT8 = 1, UINT16 = 2, INT16 = 3, 
    UINT32 = 4, INT32 = 5, FLOAT32 = 6, BOOL = 7, 
    STRING = 8, ARRAY = 9, UINT64 = 10, INT64 = 11, FLOAT64 = 12
};

//简化的量化类型，对应张量存储格式
enum class GGMLType : uint32_t {
    F32  = 0,
    F16  = 1,
    Q4_0 = 2,
    Q8_0 = 8, //这是引擎原生支持的INT8
};

//张量描述块
struct GGUFTensorInfo {
    std::string name;
    uint32_t n_dims;
    uint64_t dimensions[4]; //形状维度，如[4096, 4096]
    GGMLType type;
    uint64_t offset;        //该张量在文件中的偏移
    uint64_t byte_size;     //由type与shape严格计算的映射字节数
};

class GGUFParser
{

private:
    std::string file_path;
    
    //映射指针与大小
    void* mmap_addr;
    size_t file_size;
    size_t cursor; //遍历游标
    size_t data_offset;
    bool parsed;

#ifdef _WIN32
    void* hFile;
    void* hMapping;
#else
    int fd;
#endif

    std::unordered_map<std::string, std::string> kv_strings;
    std::unordered_map<std::string, uint32_t> kv_uint32s;
    std::unordered_map<std::string, uint64_t> kv_uint64s;
    std::unordered_map<std::string, double> kv_floats;

    std::unordered_set<std::string> metadata_names;
    std::vector<GGUFTensorInfo> tensors;

    //模型原生词表数组
    std::vector<std::string> vocab_strings;
    // Keep borrowed array records so numeric-only sessions need no merge table.
    std::string_view merges_record, types_record;
    std::unordered_map<std::string, uint8_t> tokenizer_flags;

    //解析二进制流的辅助探针
    void ReadBytes(void* dst, size_t len);
    void SkipBytes(size_t len);
    uint32_t ReadU32();
    uint64_t ReadU64();
    std::string ReadString();
    void SkipValue(uint32_t type, uint32_t depth = 0);

public:
    GGUFParser(const std::string& path);
    ~GGUFParser();
    GGUFParser(const GGUFParser&) = delete;
    GGUFParser& operator=(const GGUFParser&) = delete;

    //解析头部与KV字典
    bool ParseHeaders();

    //将字典转译为引擎所需的配置
    const std::string& Architecture() const;
    bool Describe(const MiciusGGUFParserV1& parser, MiciusGGUFPlanV1& plan) const;
    void MapTensors(const MiciusGGUFParserV1& parser, const MiciusGGUFPlanV1& plan);
    const void* FileData() const noexcept { return mmap_addr; }
    size_t FileSize() const noexcept { return file_size; }
    std::string_view TextString(const char* key) const;
    uint32_t TextUint(const char* key) const;
    bool TextFlag(const char* key) const;
    std::string_view MergesRecord() const { return merges_record; }
    std::string_view TypesRecord() const { return types_record; }

    //获取张量元数据表，供外壳构建C-ABI桥接数组
    const std::vector<GGUFTensorInfo>& GetTensors() const
    {
        return tensors;
    }

    //向外壳暴露提取的真实词表
    const std::vector<std::string>& GetVocab() const
    {
        return vocab_strings;
    }

    //偏移获取地址，由于mmap的特性通过首地址+游标+相对偏移
    const void* GetTensorData(uint64_t offset) const noexcept;

    //获取张量区域的基础指针供后续接驳Engine
    const void* GetTensorDataBase() const;
    
    //打印终端报告
    void PrintTopology() const;
};

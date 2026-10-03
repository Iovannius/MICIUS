#pragma once

#include "../micius_runtime.h"

#include <string>
#include <vector>
#include <unordered_map>
#include <cstdint>
#include <stdexcept>

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

    std::vector<GGUFTensorInfo> tensors;

    //模型原生词表数组
    std::vector<std::string> vocab_strings;

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

    //解析头部与KV字典
    bool ParseHeaders();

    //将字典转译为引擎所需的配置
    bool ExtractEngineConfig(MiciusConfig& out_config);

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

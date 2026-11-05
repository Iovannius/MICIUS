#include "micius_gguf.h"
#include "MiciusGGUFParser.h"
#include "MiciusTensorCodec.h"
#include "AI/FileTensorBinding.h"
#include "ContextDirectory.h"
#include "FileReader.h"
#include "app/I22LoaderLifetime.h"
#include "app/I25LoaderLifetime.h"
#include "app/I22Memory.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <immintrin.h>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

struct MiciusFileModel {
    explicit MiciusFileModel(const char* path) : reader(path) {}
    ~MiciusFileModel() {
        if (engine) Micius_DestroyEngine(engine);
        for (void* p : buffers) _mm_free(p);
    }
    GGUFParser reader;
    MiciusConfig config{};
    MiciusContext* engine{};
    std::vector<void*> buffers;
    std::string directory;
    std::string architecture;
};

const GGUFParser& micius::gguf::FileReader(const MiciusFileModel& model) { return model.reader; }

namespace {
#if defined(MICIUS_I6_BENCHMARK)
static void I22Ledger(const char* phase,const MiciusFileModel& model,
    const std::vector<MiciusTensorEntryV1>& entries,size_t released=0) {
    if(!micius::i22::Observing())return;
    std::ostringstream x;size_t bytes=0,scales=0,weights=0,f32=0,count=0;
    x<<",\"allocations\":[";
    for(const auto& e:entries) {
        if(std::find(model.buffers.begin(),model.buffers.end(),e.tensor_data)==model.buffers.end())continue;
        if(count++)x<<',';
        x<<"{\"name\":"<<micius::i22::Quote(e.tensor_name)<<",\"type\":"<<e.data_type<<",\"requested_bytes\":"<<e.byte_size<<",\"alignment\":64}";
        bytes+=e.byte_size;
        if(e.data_type==MICIUS_TENSOR_Q8_32_SCALE)scales+=e.byte_size;
        else if(e.data_type==MICIUS_TENSOR_Q8_32_WEIGHT)weights+=e.byte_size;
        else f32+=e.byte_size;
    }
    x<<"],\"shell_owned_buffers\":"<<model.buffers.size()<<",\"shell_requested_bytes\":"<<bytes
     <<",\"shell_public_scale_bytes\":"<<scales<<",\"shell_q8_weight_bytes\":"<<weights
     <<",\"shell_f32_bytes\":"<<f32<<",\"released_public_scale_bytes\":"<<released;
    micius::i22::Memory(phase,x.str());
}

#else
static void I22Ledger(const char*, const MiciusFileModel&, const std::vector<MiciusTensorEntryV1>&, size_t = 0) noexcept {}
#endif

using Scale = MiciusQ8Block32Scale;
size_t Product(uint64_t a, uint64_t b)
{
    if (a > SIZE_MAX || b > SIZE_MAX || (a && b > SIZE_MAX / a))
        throw std::runtime_error("Tensor allocation exceeds addressable space");
    return static_cast<size_t>(a * b);
}

template<class T> T* Allocate(MiciusFileModel& model, size_t count)
{
    void* ptr = _mm_malloc(Product(count, sizeof(T)), 64);
    if (!ptr) throw std::bad_alloc();
    try { model.buffers.push_back(ptr); }
    catch (...) { _mm_free(ptr); throw; }
    return static_cast<T*>(ptr);
}

MiciusTensorEntryV1 Entry(const char* name, const void* data, size_t bytes, uint32_t type,
                          uint64_t rows, uint64_t cols, size_t stride)
{
    MiciusTensorEntryV1 e{};
    e.tensor_name = name;
    e.tensor_data = data;
    e.byte_size = bytes;
    e.data_type = type;
    e.rank = cols ? 2U : 1U;
    e.dimensions[0] = rows;
    e.dimensions[1] = cols;
    e.row_stride_bytes = stride;
    return e;
}

float* FloatCopy(MiciusFileModel& model, const GGUFTensorInfo& t, const void* raw, size_t count)
{
    float* dst = Allocate<float>(model, count);
    if (t.type == GGMLType::F32) {
        std::memcpy(dst, raw, Product(count, sizeof(float)));
    } else if (t.type == GGMLType::F16) {
        const auto* source = static_cast<const unsigned char*>(raw);
        for (size_t i = 0; i < count; ++i) {
            uint16_t bits;
            std::memcpy(&bits, source + i * sizeof(bits), sizeof(bits));
            dst[i] = MiciusTensorCodec::DecodeFloat16(bits);
        }
    } else if (t.type != GGMLType::Q8_0 ||
               !MiciusTensorCodec::DequantizeGGUFQ8_0ToF32(raw, static_cast<size_t>(t.byte_size),
                    static_cast<size_t>(t.dimensions[1]), static_cast<size_t>(t.dimensions[0]), dst, count)) {
        throw std::runtime_error("Cannot convert " + t.name + " to F32");
    }
    return dst;
}

void TranslateAndBind(MiciusFileModel& model, uint32_t features)
{
    const auto& tensors = model.reader.GetTensors();
    if (tensors.size() > (UINT32_MAX-1U)/2U) throw std::runtime_error("Too many GGUF tensors");
    const bool tied = std::none_of(tensors.begin(), tensors.end(),
        [](const auto& t) { return t.name == "output.weight"; });
    std::vector<MiciusTensorEntryV1> entries;
    std::vector<std::string> scale_names;
    entries.reserve(tensors.size()*2+1);
    scale_names.reserve(tensors.size());
    model.buffers.reserve(tensors.size()*2+1);
    for (const auto& t : tensors) {
        const void* raw = model.reader.GetTensorData(t.offset);
        if (!raw || t.n_dims > 2) throw std::runtime_error("Unsupported tensor range/rank: " + t.name);
        const bool matrix = t.n_dims == 2;
        const uint64_t rows = matrix ? t.dimensions[1] : t.dimensions[0];
        const uint64_t cols = matrix ? t.dimensions[0] : 0;
        const size_t count = Product(rows, matrix ? cols : 1);
        if (t.name == "token_embd.weight") {
            if (!matrix || cols != model.config.hidden_dim || rows != model.config.vocab_size)
                throw std::runtime_error("Invalid token embedding dimensions");
            const auto* embedding = FloatCopy(model, t, raw, count);
            entries.push_back(Entry("token_embd.weight.f32", embedding, Product(count,sizeof(float)),
                                    MICIUS_TENSOR_F32, rows, cols, Product(cols,sizeof(float))));
            if (!tied || t.type != GGMLType::Q8_0) continue;
        }
        if (t.type == GGMLType::Q8_0 && matrix && cols % 32 == 0 && t.name.ends_with(".weight")) {
            const size_t blocks = count/32;
            auto* weights = Allocate<int8_t>(model, count);
            auto* scales = Allocate<Scale>(model, blocks);
            if (!MiciusTensorCodec::ConvertGGUFQ8_0ToQ8_32(raw, static_cast<size_t>(t.byte_size),
                    static_cast<size_t>(rows), static_cast<size_t>(cols), weights, count, scales, blocks))
                throw std::runtime_error("Cannot convert Q8_0 matrix: " + t.name);
            entries.push_back(Entry(t.name.c_str(), weights, count, MICIUS_TENSOR_Q8_32_WEIGHT,
                                    rows, cols, static_cast<size_t>(cols)));
            scale_names.push_back(t.name.substr(0,t.name.size()-7) + ".scales");
            entries.push_back(Entry(scale_names.back().c_str(), scales, Product(blocks,sizeof(Scale)),
                                    MICIUS_TENSOR_Q8_32_SCALE, rows, cols/32, Product(cols/32,sizeof(Scale))));
        } else if (t.type == GGMLType::F32 || (t.type == GGMLType::F16 && !matrix)) {
            const void* values = t.type == GGMLType::F32 ? raw : FloatCopy(model,t,raw,count);
            entries.push_back(Entry(t.name.c_str(),values,Product(count,sizeof(float)),MICIUS_TENSOR_F32,
                                    rows,cols,Product(cols,sizeof(float))));
        } else {
            throw std::runtime_error("Unsupported type/shape for " + t.name);
        }
    }
    micius::internal::ProjectionSemantics semantics{
        bool(features & MICIUS_GGUF_SPLIT_HALF_ROPE), bool(features & MICIUS_GGUF_QKV_BIAS)};
    I22Ledger("translated_before_bind",model,entries);
    if (!micius::internal::BindTypedTensors(model.engine, entries.data(), static_cast<uint32_t>(entries.size()),
            sizeof(MiciusTensorEntryV1), MICIUS_ABI_VERSION, semantics))
        throw std::runtime_error(std::string("Tensor binding rejected: ") + Micius_GetLastError());
    I22Ledger("bound_before_cleanup",model,entries);
    size_t released_bytes = 0;
#if defined(MICIUS_I22_RELEASE_LOADER_SCALES)
    micius::i22::ReleaseStats released;
    if (!micius::i22::ReleasePublicScales(model.buffers,entries,released))
        throw std::runtime_error("Loader scale ownership contract failed");
    released_bytes = released.bytes;
#endif
    size_t retired = 0;
    if (!micius::i25::ReleaseWeights(model.buffers,entries,micius::i25::RetirableWeights(model.engine),retired))
        throw std::runtime_error("Loader weight ownership contract failed");
    I22Ledger("bound_after_cleanup",model,entries,released_bytes);
#if defined(MICIUS_I6_BENCHMARK)
    for (const auto& old : micius::i25::RetirableWeights(model.engine))
        if (std::find(model.buffers.begin(),model.buffers.end(),reinterpret_cast<void*>(old.address)) != model.buffers.end())
            throw std::runtime_error("Retired weight is still owned");
#endif
}
}

extern "C" MiciusFileModel* Micius_OpenGGUFV1(const MiciusGGUFOpenOptionsV1* o, char* error, size_t capacity)
{
    if (error && capacity) error[0] = 0;
    try {
        if (!o || o->abi_version != MICIUS_GGUF_FILE_ABI || o->struct_size < sizeof(*o) ||
            !o->path || !*o->path || !o->parsers || !o->parser_count || (o->cache_root && !*o->cache_root))
            throw std::invalid_argument("Invalid file options, parser list or ABI version");
        auto model = std::make_unique<MiciusFileModel>(o->path);
        if (!model->reader.ParseHeaders()) throw std::runtime_error("Expected a GGUF v2/v3 file");
        model->architecture = model->reader.Architecture();
        const auto& arch = model->architecture;
        const MiciusGGUFParserV1* selected = nullptr;
        std::unordered_set<std::string> registered;
        for (uint32_t i = 0; i < o->parser_count; ++i) {
            const auto* p = o->parsers[i];
            if (!p || p->abi_version != MICIUS_GGUF_PARSER_ABI || p->struct_size < sizeof(*p) ||
                !p->architecture || !*p->architecture || !p->describe || !p->map_tensor ||
                !registered.emplace(p->architecture).second)
                throw std::invalid_argument("Invalid or duplicate architecture parser");
            if (arch == p->architecture) selected = p;
        }
        if (!selected) throw std::runtime_error("No linked parser for architecture: " + arch);
        MiciusGGUFPlanV1 plan{};
        if (!model->reader.Describe(*selected,plan)) throw std::runtime_error("Unsupported " + arch + " model metadata");
        micius::i22::Memory("headers_mapped");
        model->reader.MapTensors(*selected,plan);
        model->config = plan.config;
        if (o->context_length > model->config.max_seq_len) throw std::runtime_error("Requested context exceeds model limit");
        if (o->context_length) model->config.max_seq_len = o->context_length;
        model->engine = Micius_CreateEngineV1(&model->config,sizeof(model->config),MICIUS_ABI_VERSION);
        if (!model->engine) throw std::runtime_error(Micius_GetLastError());
        micius::i22::Memory("engine_allocated");
        TranslateAndBind(*model, plan.features);
        if (o->cache_root) model->directory = micius::gguf::CreateContextDirectory(o->cache_root,
            model->reader.FileData(), model->reader.FileSize());
        return model.release();
    } catch (const std::exception& e) {
        if (error && capacity) std::snprintf(error,capacity,"%s",e.what());
    } catch (...) {
        if (error && capacity) std::snprintf(error,capacity,"Unexpected model loading exception");
    }
    return nullptr;
}

extern "C" void Micius_CloseGGUFV1(MiciusFileModel* model) { delete model; }
extern "C" MiciusContext* Micius_FileEngineV1(const MiciusFileModel* model) { return model ? model->engine : nullptr; }
extern "C" const MiciusConfig* Micius_FileConfigV1(const MiciusFileModel* model) { return model ? &model->config : nullptr; }
extern "C" const char* Micius_FileArchitectureV1(const MiciusFileModel* model) { return model ? model->architecture.c_str() : nullptr; }
extern "C" const char* Micius_FileContextDirectoryV1(const MiciusFileModel* model) { return model ? model->directory.c_str() : nullptr; }

#include "micius_text.h"
#include "Tokenizer.h"
#include "app/MiciusGGUFParser.h"
#include <algorithm>
#include <cstdio>
#include <stdexcept>

struct MiciusTokenizer {
    explicit MiciusTokenizer(const GGUFParser& reader) : value(reader) {}
    micius::text::Tokenizer value;
};

extern "C" MiciusTokenizer* Micius_OpenTokenizerV1(const char* path, char* error, size_t cap)
{
    if (error && cap) *error = 0;
    try {
        if (!path || !*path) throw std::invalid_argument("Expected a UTF-8 GGUF path");
        GGUFParser reader(path);
        if (!reader.ParseHeaders()) throw std::runtime_error("Expected GGUF v2/v3");
        return new MiciusTokenizer(reader);
    } catch (const std::exception& e) {
        if (error && cap) std::snprintf(error,cap,"%s",e.what());
    } catch (...) {
        if (error && cap) std::snprintf(error,cap,"Tokenizer initialization failed");
    }
    return nullptr;
}
extern "C" void Micius_CloseTokenizerV1(MiciusTokenizer* t) { delete t; }
extern "C" int Micius_EncodeTextV1(const MiciusTokenizer* t, const char* text, size_t n,
                                    uint32_t* out, size_t cap, size_t* required)
{
    if (!required) return -1;
    *required = 0;
    if (!t || (!text && n) || (!out && cap)) return -1;
    try {
        const auto ids = t->value.Encode({text ? text : "",n});
        *required = ids.size();
        if (ids.size() > cap) return 0;
        if (!ids.empty()) std::copy(ids.begin(),ids.end(),out);
        return 1;
    } catch (...) { return -1; }
}
extern "C" int Micius_TokenBytesV1(const MiciusTokenizer* t, uint32_t id, const char** bytes, size_t* length)
{
    if (!bytes || !length) return 0;
    *bytes = nullptr; *length = 0;
    if (!t) return 0;
    try {
        auto piece = t->value.Piece(id);
        *bytes = piece.data(); *length = piece.size();
        return 1;
    } catch (...) { return 0; }
}

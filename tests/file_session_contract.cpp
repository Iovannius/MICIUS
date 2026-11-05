#include "micius_gguf.h"
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>

// Two live sessions, registration errors and an in-memory open through public headers.
int main(int argc, char** argv)
{
    if (argc != 3) return 2;
    const MiciusGGUFParserV1* parser = Micius_Qwen2ParserV1();
    MiciusGGUFOpenOptionsV1 o{MICIUS_GGUF_FILE_ABI,sizeof(o),argv[1],argv[2],&parser,1,8};
    using File = std::unique_ptr<MiciusFileModel, decltype(&Micius_CloseGGUFV1)>;
    char error[256]{};
    auto open = [&] { return File(Micius_OpenGGUFV1(&o,error,sizeof(error)), Micius_CloseGGUFV1); };
    auto a = open(), b = open();
    if (!a || !b) { std::fprintf(stderr,"%s\n",error); return 1; }
    auto* left = Micius_FileEngineV1(a.get());
    auto* right = Micius_FileEngineV1(b.get());
    const std::string directory = Micius_FileContextDirectoryV1(a.get());
    if (directory.empty() || directory == Micius_FileContextDirectoryV1(b.get()) ||
        Micius_PrefillTokenV1(left,3) != MICIUS_STATE_TOKEN_ACCEPTED ||
        Micius_GetSequenceLengthV1(right) != 0) return 3;
    auto rejected = [&](const char* diagnostic) {
        return !open() && std::strstr(error,diagnostic) && Micius_GetSequenceLengthV1(left)==1;
    };
    o.abi_version++;
    if (!rejected("ABI")) return 4;
    o.abi_version = MICIUS_GGUF_FILE_ABI;
    o.struct_size--;
    if (!rejected("ABI")) return 5;
    o.struct_size = sizeof(o);
    const MiciusGGUFParserV1* duplicate[] = {parser,parser};
    o.parsers=duplicate; o.parser_count=2;
    if (!rejected("duplicate")) return 6;
    MiciusGGUFParserV1 invalid=*parser;
    parser=&invalid; o.parsers=&parser; o.parser_count=1;
    invalid.struct_size--;
    if (!rejected("Invalid")) return 7;
    invalid=*Micius_Qwen2ParserV1(); invalid.abi_version++;
    if (!rejected("Invalid")) return 8;
    invalid=*Micius_Qwen2ParserV1(); invalid.architecture="unrelated";
    if (!rejected("No linked parser")) return 9;
    parser=Micius_Qwen2ParserV1();
    o.context_length=UINT32_MAX;
    if (!rejected("exceeds")) return 10;
    o.context_length=8;
    o.cache_root=nullptr;
    auto memory=open();
    if (!memory || std::strcmp(Micius_FileContextDirectoryV1(memory.get()),"") ||
        !Micius_ResetEngineV1(left) || Micius_GetSequenceLengthV1(left)!=0 ||
        directory!=Micius_FileContextDirectoryV1(a.get()) || Micius_GetSequenceLengthV1(right)!=0) return 11;
    a.reset();
    return Micius_PrefillTokenV1(right,7)==MICIUS_STATE_TOKEN_ACCEPTED ? 0 : 12;
}

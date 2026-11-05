#include "micius_gguf.h"
#include "app/MiciusGGUFParser.h"
#include "gguf/FileReader.h"
#include "text/Conversation.h"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <iterator>
#include <memory>
#include <stdexcept>

using namespace micius::text;
namespace {
void Check(bool value, const char* name) { if (!value) throw std::runtime_error(name); }
int PromptOnly(char** argv)
{
    GGUFParser reader(argv[2]); Check(reader.ParseHeaders(),"GGUF header");
    Tokenizer t(reader);
    std::ifstream f(argv[3],std::ios::binary);
    Check(bool(f),"input");
    std::string text{std::istreambuf_iterator<char>(f),{}};
    auto ids=t.Prompt(std::string_view(argv[4])=="first",text);
    std::ofstream out(argv[5],std::ios::binary);
    out.write(reinterpret_cast<const char*>(ids.data()),static_cast<std::streamsize>(ids.size()*4));
    out.close(); Check(bool(out),"output"); return 0;
}
void State(const char* path)
{
    const MiciusGGUFParserV1* modules[]={Micius_LlamaParserV1(),Micius_Qwen2ParserV1()};
    MiciusGGUFOpenOptionsV1 opts{MICIUS_GGUF_FILE_ABI,sizeof(opts),path,nullptr,modules,2,256};
    auto open=[&]() {
        char error[512]{};
        std::unique_ptr<MiciusFileModel,decltype(&Micius_CloseGGUFV1)> model(Micius_OpenGGUFV1(&opts,error,sizeof(error)),Micius_CloseGGUFV1);
        if (!model) throw std::runtime_error(error);
        return model;
    };
    auto active=open(), reference=open();
    auto* a=Micius_FileEngineV1(active.get());
    auto* b=Micius_FileEngineV1(reference.get());
    Tokenizer t(micius::gguf::FileReader(*active));
    Conversation chat(a,256,t);
    std::atomic<bool> cancel{false};
    std::string bytes;
    auto emit=[&](std::string_view p){ bytes.append(p); };
    auto closed=[&](bool first, std::string_view user, const std::vector<uint32_t>& answer) {
        auto ids=t.Prompt(first,user);
        ids.insert(ids.end(),answer.begin(),answer.end());
        auto end=t.EndTurn(); ids.insert(ids.end(),end.begin(),end.end());
        for (uint32_t id:ids) Check(Micius_PrefillTokenV1(b,id)==MICIUS_STATE_TOKEN_ACCEPTED,"reference replay");
        auto size=Micius_GetSequenceLengthV1(b);
        Check(size==Micius_GetSequenceLengthV1(a),"history length");
        auto* config=Micius_FileConfigV1(active.get());
        const size_t width=size_t(config->num_kv_heads)*config->head_dim;
        std::vector<float> ak(width),av(width),bk(width),bv(width);
        for (uint32_t layer=0;layer<config->num_layers;++layer)
            for (uint32_t pos=0;pos<size;++pos)
                Check(Micius_CopyReferenceKVV1(a,layer,pos,ak.data(),av.data(),width) &&
                      Micius_CopyReferenceKVV1(b,layer,pos,bk.data(),bv.data(),width) &&
                      ak==bk && av==bv,"KV differs from explicit full-history replay");
    };
    auto first=chat.Reply("first",12,cancel,emit);
    Check(first.reason==Finish::End && first.tokens==3 && bytes=="\xe4\xbd\xa0","EOS and byte pieces");
    closed(true,"first",{0xe4,0xbd,0xa0});
    auto interrupted=chat.Reply("discard this",8,cancel,[&](std::string_view){ cancel=true; });
    Check(interrupted.reason==Finish::Interrupted,"cancellation");
    cancel=false;
    bool rejected=false;
    try { chat.Reply(std::string(500,'z'),4,cancel,emit); } catch(const std::length_error&) { rejected=true; }
    Check(rejected,"oversized turn");
    bytes.clear();
    auto second=chat.Reply("next",1,cancel,emit);
    Check(second.reason==Finish::ReplyLimit && second.tokens==1 && bytes=="\xe4","bounded reply");
    closed(false,"next",{0xe4});
    chat.Reset(); Check(Micius_ResetEngineV1(b),"reset reference"); bytes.clear();
    Check(chat.Reply("first",12,cancel,emit).reason==Finish::End,"restart");
    closed(true,"first",{0xe4,0xbd,0xa0});
    const auto cap=t.Prompt(true,"x").size()+t.EndTurn().size()+1;
    Conversation small(a,cap,t);
    Check(small.Reply("x",8,cancel,emit).reason==Finish::ContextLimit && Micius_GetSequenceLengthV1(a)==cap,"context limit");
}
}
int main(int argc,char** argv)
{
    try {
        if (argc==6 && std::string_view(argv[1])=="--prompt-only") return PromptOnly(argv);
        if (argc!=2) return 2;
        State(argv[1]); std::cout<<"chat state, KV and rollback: PASS\n"; return 0;
    } catch (const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}

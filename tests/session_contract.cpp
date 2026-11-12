#include "micius_session.h"
#include "micius_state.h"
#include "session/Storage.h"
#include <cstring>
#include <filesystem>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>
namespace fs=std::filesystem;
using namespace micius::session;
static void Check(bool b,const char* message) { if(!b)throw std::runtime_error(message); }
static std::vector<uint8_t> State(MiciusContext* e) {
    size_t n=0;Check(Micius_ExportStateV1(e,nullptr,0,&n)==0,"size");std::vector<uint8_t> b(n);
    Check(Micius_ExportStateV1(e,b.data(),b.size(),&n)==1,"export");return b;
}
static int Emit(const char* s,size_t n,void* p) { static_cast<std::string*>(p)->append(s,n);return 1; }
static int Cancel(const char*,size_t,void*) { return 0; }
int main(int argc,char** argv) { try {
    Check(argc==3,"MODEL CACHE_ROOT");char error[1024]={};
    const MiciusGGUFParserV1* parsers[]={Micius_LlamaParserV1(),Micius_Qwen2ParserV1()};
    MiciusGGUFOpenOptionsV1 o={MICIUS_GGUF_FILE_ABI,sizeof(o),argv[1],argv[2],parsers,2,256};
    auto open=[&]{auto* m=Micius_OpenGGUFV1(&o,error,sizeof(error));Check(m,error);return std::unique_ptr<MiciusFileModel,decltype(&Micius_CloseGGUFV1)>(m,Micius_CloseGGUFV1);};
    auto a=open(),b=open();auto* engine=Micius_FileEngineV1(a.get());
    auto empty=State(engine);Check(Micius_ImportStateV1(engine,empty.data(),empty.size(),0),"empty import");
    for(uint32_t i=0;i<19;++i)Check(Micius_PrefillTokenV1(engine,i)==MICIUS_STATE_TOKEN_ACCEPTED,"prefix");
    auto original=State(engine);
    auto reject=[&](std::vector<uint8_t> bad,uint32_t n=19) {
        Check(!Micius_ImportStateV1(engine,bad.data(),bad.size(),n),"invalid raw state accepted");
        Check(State(engine)==original,"failed import changed live state");
    };
    auto bad=original;bad.pop_back();reject(bad);
    bad=original;bad.push_back(0);reject(bad);
    bad=original;bad[0]^=1;reject(bad);
    bad=original;bad[12]^=1;reject(bad); // numerical fingerprint
    bad=original;bad[76]^=1;reject(bad); // shape after 64-byte identity
    bad=original;bad[bad.size()-1]=0x7f;bad[bad.size()-2]=0xc0;reject(bad); // NaN
    reject(original,18);
    unsigned csr=_mm_getcsr();_mm_setcsr(csr^0x8000u);
    Check(!Micius_ImportStateV1(engine,original.data(),original.size(),19),"FP controls mismatch");_mm_setcsr(csr);
    Check(State(engine)==original,"FP failed import mutated");
    auto* other=Micius_FileEngineV1(b.get());Check(Micius_ImportStateV1(other,original.data(),original.size(),19),"restore");
    uint32_t x=0,y=0;Check(Micius_DecodeOneV1(engine,22,&x)==MICIUS_STATE_YIELD_TOKEN &&
        Micius_DecodeOneV1(other,22,&y)==MICIUS_STATE_YIELD_TOKEN && x==y && State(engine)==State(other),"resume next state");
    // Reach full capacity and restore it; a subsequent token must remain rejected.
    for(uint32_t i=20;i<256;++i)Check(Micius_PrefillTokenV1(engine,10)==MICIUS_STATE_TOKEN_ACCEPTED,"fill");
    auto full=State(engine);Check(Micius_ImportStateV1(other,full.data(),full.size(),256),"full import");
    Check(Micius_PrefillTokenV1(other,10)==MICIUS_STATE_CONTEXT_FULL,"full guard");

    MiciusChatSession* s=Micius_OpenChatV1(a.get(),nullptr,error,sizeof(error));Check(s,error);
    auto dir=fs::u8path(Micius_ChatDirectoryV1(s));auto id=Utf8(dir.filename());
    Check(!Micius_OpenChatV1(a.get(),nullptr,error,sizeof(error)),"duplicate engine chat");
    Check(!Micius_OpenChatV1(b.get(),id.c_str(),error,sizeof(error)),"same directory lock");
    std::string answer;MiciusChatResultV1 r={MICIUS_SESSION_ABI_V1,sizeof(r),0,0,0};
    Check(Micius_ChatReplyV1(s,"first",5,8,Emit,&answer,&r,error,sizeof(error)),error);
    Check(Micius_SaveChatV1(s,error,sizeof(error)),error);
    auto live=State(engine);auto saved=ReadFile(dir/"kv/chat.mcs",1<<24);
    auto conf=ReadFile(dir/"sampling.conf",16384);
    AtomicText(dir/"sampling.conf","version=1\ntemperature=nan\n");
    Check(!Micius_ChatReplyV1(s,"bad",3,8,Emit,&answer,&r,error,sizeof(error)) && State(engine)==live,"bad config changed state");
    AtomicWrite(dir/"sampling.conf",conf);
    Check(Micius_ChatReplyV1(s,"discard",7,8,Cancel,nullptr,&r,error,sizeof(error)) && r.finish==3,"cancel");
    Check(Micius_SaveChatV1(s,error,sizeof(error)),error); // rollback before saving
    auto rollback=State(engine);
    // Only diagnostic generated-token count can change after replay, never KV/logits.
    const size_t counter=8+4+64+56+4+4+4;
    Check(live.size()==rollback.size(),"rollback size");
    for(size_t i=0;i<live.size();++i)if(i<counter || i>=counter+4)Check(live[i]==rollback[i],"cancel changed KV/logits");
    Micius_CloseChatV1(s);s=nullptr;
    saved=ReadFile(dir/"kv/chat.mcs",1<<24);bad=saved;bad.back()^=1;AtomicWrite(dir/"kv/chat.mcs",bad);
    Check(!Micius_OpenChatV1(a.get(),id.c_str(),error,sizeof(error)) && State(engine)==rollback,"corrupt chat restore changed state");
    AtomicWrite(dir/"kv/chat.mcs",saved);
    s=Micius_OpenChatV1(a.get(),id.c_str(),error,sizeof(error));Check(s,error);
    Check(State(engine)==rollback,"chat snapshot restores raw state");
    // An exclusive destination lock simulates replacement failure. Preserve old bytes.
#ifdef _WIN32
    { Lock blocked(dir/"kv/chat.mcs");Check(!Micius_SaveChatV1(s,error,sizeof(error)),"locked save accepted"); }
    Check(ReadFile(dir/"kv/chat.mcs",1<<24)==saved,"failed replacement destroyed old snapshot");
#endif
    Check(Micius_ResetChatV1(s,error,sizeof(error)) && Micius_SaveChatV1(s,error,sizeof(error)),"reset/save");
    Check(Micius_GetSequenceLengthV1(engine)==0,"reset length");Micius_CloseChatV1(s);
    s=Micius_OpenChatV1(a.get(),id.c_str(),error,sizeof(error));Check(s && Micius_GetSequenceLengthV1(engine)==0,"empty reset restore");Micius_CloseChatV1(s);
    std::cout<<"State/session corruption, atomic save, config, locks, cancellation and capacity: PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;} }

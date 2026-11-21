#include "Session.h"
#include "Binary.h"
#include "gguf/ContextDirectory.h"
#include "gguf/FileReader.h"
#include "gguf/NativePath.h"
#include "micius_state.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <set>
#include <sstream>

namespace micius::session {
namespace fs=std::filesystem;
using gguf::NativePath;
using state::Product;
static void Require(bool b,const char* why) { if(!b)throw std::runtime_error(why); }
static bool ValidId(std::string_view id) {
    return id.size()==32 && std::all_of(id.begin(),id.end(),[](char c){return (c>='0'&&c<='9')||(c>='a'&&c<='f');});
}
static fs::path Directory(MiciusFileModel* f,const char* id) {
    Require(f!=nullptr,"A file model is required");
    const char* current=Micius_FileContextDirectoryV1(f);
    Require(current && *current,"A cache root is required for persistent chat");
    fs::path dir=fs::u8path(current);
    if(id) { Require(ValidId(id),"Expected a 32-digit lowercase context ID");dir=dir.parent_path()/id; }
    PlainDirectory(dir);PlainDirectory(dir/"kv");return dir;
}
EngineLease::EngineLease(MiciusFileModel* f):file(f) {
    Require(gguf::AcquireChatEngine(f),"This file engine already has an active chat session");
}
EngineLease::~EngineLease() { gguf::ReleaseChatEngine(file); }
Session::Session(MiciusFileModel* f,const char* id)
    :lease(f),file(f),directory(Utf8(session::Directory(f,id))),path(fs::u8path(directory)),
     lock(std::make_unique<Lock>(path/".lock")),tokenizer(gguf::FileReader(*f)),
     chat(Micius_FileEngineV1(f),Micius_FileConfigV1(f)->max_seq_len,tokenizer,false)
{
    const bool existing=fs::exists(NativePath(path/"kv/chat.mcs"));
    if(id)Require(existing,"This context has no saved snapshot; use a new conversation");
    if(!fs::exists(NativePath(path/"sampling.conf"))) {
        Require(!existing,"Saved context is missing sampling.conf");
        AtomicText(path/"sampling.conf",WriteConfig(config));
    }
    // Validate the editable file even when a saved snapshot supplies the last
    // effective parameters. It will be re-read before the next user turn.
    Reload();
    if(existing)Restore();
    else {
        // Human-readable identity is model-scoped, each context remains private.
        AtomicText(path.parent_path().parent_path()/"model.txt",
            std::string("sha256=")+Micius_FileIdentityV1(file)+"\narchitecture="+
            Micius_FileArchitectureV1(file)+"\n");
        chat.Reset();
    }
}
void Session::Reload() {
    auto raw=ReadFile(path/"sampling.conf",16384);
    auto next=ParseConfig({reinterpret_cast<const char*>(raw.data()),raw.size()});
    chat.Configure(next.sample,next.seed);config=next;
}
void Session::Reset() { chat.Reset(); turns=0; }
text::Answer Session::Reply(std::string_view text,uint32_t limit,const std::atomic<bool>& cancel,
                           const std::function<void(std::string_view)>& emit) {
    Reload();auto answer=chat.Reply(text,limit?limit:config.max_new_tokens,cancel,emit);
    if(answer.reason!=text::Finish::Interrupted)++turns;
    return answer;
}
void Session::Save() {
    chat.EnsureClean();
    auto* engine=Micius_FileEngineV1(file);
    Require(chat.Tokens().size()==Micius_GetSequenceLengthV1(engine),"Snapshot history/KV length differs");
    size_t needed=0;
    Require(Micius_ExportStateV1(engine,nullptr,0,&needed)==0,"Cannot size engine snapshot");
    std::vector<uint8_t> state(needed);
    Require(Micius_ExportStateV1(engine,state.data(),state.size(),&needed)==1,"Cannot export engine snapshot");
    state::Writer w;
    w.Raw("MCSCHAT1",8);w.Text(Micius_FileIdentityV1(file));w.Text(Micius_FileArchitectureV1(file));
    w.Text(WriteConfig(config));w.U64(chat.RandomState());w.U64(turns);
    w.U32(static_cast<uint32_t>(chat.Tokens().size()));
    for(uint32_t id:chat.Tokens())w.U32(id);
    w.U64(state.size());w.Raw(state.data(),state.size());
    const auto hash=gguf::Sha256(w.bytes.data(),w.bytes.size());w.Raw(hash.data(),hash.size());
    AtomicWrite(path/"kv/chat.mcs",w.bytes);
}
void Session::Restore() {
    const auto* cfg=Micius_FileConfigV1(file);
    // Bound allocation from the already validated model, before reading a file.
    size_t kv=Product(Product(Product(cfg->max_seq_len,cfg->num_layers),cfg->num_kv_heads),cfg->head_dim);
    size_t limit=Product(kv,8);
    size_t rest=Product(size_t(cfg->vocab_size)+cfg->max_seq_len,4)+32768;
    Require(limit<=SIZE_MAX-rest,"Snapshot size overflow");
    auto bytes=ReadFile(path/"kv/chat.mcs",limit+rest);
    Require(bytes.size()>=64,"Truncated snapshot checksum");
    auto payload=std::span<const uint8_t>(bytes).first(bytes.size()-64);
    auto hash=gguf::Sha256(payload.data(),payload.size());
    Require(std::memcmp(hash.data(),bytes.data()+payload.size(),64)==0,"Snapshot checksum mismatch");
    state::Reader r{payload};auto magic=r.Raw(8);
    Require(std::memcmp(magic.data(),"MCSCHAT1",8)==0,"Unknown chat snapshot format");
    Require(r.Text(64)==Micius_FileIdentityV1(file),"Snapshot belongs to a different model");
    Require(r.Text(128)==Micius_FileArchitectureV1(file),"Snapshot architecture differs");
    Config saved=ParseConfig(r.Text(16384));uint64_t rng=r.U64(),saved_turns=r.U64();
    uint32_t n=r.U32();Require(n<=cfg->max_seq_len,"Snapshot history too long");
    Require((n==0)==(saved_turns==0) && saved_turns<=n,"Invalid saved turn count");
    std::vector<uint32_t> history;history.reserve(n);
    for(uint32_t i=0;i<n;++i) { auto id=r.U32();Require(id<cfg->vocab_size,"Snapshot token out of range");history.push_back(id); }
    auto length=r.U64();Require(length<=SIZE_MAX,"Snapshot payload too large");
    auto raw=r.Raw(static_cast<size_t>(length));r.End();
    // Configure only the not-yet-published chat object; engine commit is last.
    chat.Configure(saved.sample,saved.seed);
    Require(Micius_ImportStateV1(Micius_FileEngineV1(file),raw.data(),raw.size(),n)==1,
        "Snapshot engine contract/shape/FP state mismatch; active engine preserved");
    chat.AdoptRestoredHistory(history,rng);config=saved;turns=saved_turns;
}
std::string Session::List(MiciusFileModel* file) {
    auto root=session::Directory(file,nullptr).parent_path();std::set<std::string> rows;
    for(const auto& entry:fs::directory_iterator(NativePath(root))) {
        auto id=Utf8(entry.path().filename());
        if(!ValidId(id) || !entry.is_directory() || entry.is_symlink())continue;
        if(fs::is_regular_file(entry.path()/"sampling.conf"))
            rows.insert(id+(fs::is_regular_file(entry.path()/"kv"/"chat.mcs")?"  saved":"  unsaved"));
    }
    std::string result;for(const auto& row:rows)result+=row+'\n';return result.empty()?"No saved chat contexts.\n":result;
}
}

struct MiciusChatSession {
    micius::session::Session impl;
    MiciusFileModel* file;
    std::atomic<bool> cancel{false};
    MiciusChatSession(MiciusFileModel* f,const char* id):impl(f,id),file(f){}
};
namespace {
template<class F> int Guard(char* error,size_t cap,F&& f) noexcept {
    if(error&&cap)error[0]=0;
    try { f();return 1; }
    catch(const std::exception& e) { if(error&&cap)std::snprintf(error,cap,"%s",e.what()); }
    catch(...) { if(error&&cap)std::snprintf(error,cap,"Unexpected session error"); }
    return 0;
}
void CheckChat(MiciusChatSession* s) { if(!s)throw std::invalid_argument("Null chat session"); }
}
extern "C" MiciusChatSession* Micius_OpenChatV1(MiciusFileModel* f,const char* id,char* err,size_t cap) {
    MiciusChatSession* s=nullptr;Guard(err,cap,[&]{s=new MiciusChatSession(f,id);});return s;
}
extern "C" void Micius_CloseChatV1(MiciusChatSession* s) { delete s; }
extern "C" const char* Micius_ChatDirectoryV1(const MiciusChatSession* s) { return s?s->impl.Directory().c_str():nullptr; }
extern "C" void Micius_CancelChatV1(MiciusChatSession* s) { if(s)s->cancel.store(true,std::memory_order_relaxed); }
extern "C" int Micius_SaveChatV1(MiciusChatSession* s,char* e,size_t n) { return Guard(e,n,[&]{CheckChat(s);s->impl.Save();}); }
extern "C" int Micius_ResetChatV1(MiciusChatSession* s,char* e,size_t n) { return Guard(e,n,[&]{CheckChat(s);s->impl.Reset();}); }
extern "C" int Micius_ReloadChatConfigV1(MiciusChatSession* s,char* e,size_t n) { return Guard(e,n,[&]{CheckChat(s);s->impl.Reload();}); }
extern "C" int Micius_ChatReplyV1(MiciusChatSession* s,const char* text,size_t length,uint32_t limit,
    MiciusChatEmitV1 emit,void* user,MiciusChatResultV1* result,char* error,size_t cap) {
    return Guard(error,cap,[&]{
        CheckChat(s);
        if(!text || !length || !emit || !result || result->abi_version!=MICIUS_SESSION_ABI_V1 || result->struct_size<sizeof(*result))
            throw std::invalid_argument("Invalid chat request/result ABI");
        s->cancel=false;
        auto answer=s->impl.Reply({text,length},limit,s->cancel,[&](std::string_view p){if(!emit(p.data(),p.size(),user))s->cancel=true;});
        result->finish=static_cast<uint32_t>(answer.reason);result->tokens=static_cast<uint32_t>(answer.tokens);
        result->context_length=Micius_GetSequenceLengthV1(Micius_FileEngineV1(s->file));
    });
}

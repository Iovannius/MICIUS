#include "app/MiciusShell.h"
#include "I22Memory.h"
#include "AI/PreparedGEMVWeights.h"
#include "AI/I25Binding.h"
#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <immintrin.h>
#include <iostream>
#include <map>
#include <stdexcept>
#include <windows.h>
#include <psapi.h>

namespace {
using Clock=std::chrono::steady_clock;
void Need(bool ok,const char* message) { if(!ok) throw std::runtime_error(message); }
uint32_t ReadU32(std::istream& in) {
    uint32_t v=0;in.read(reinterpret_cast<char*>(&v),4);Need(bool(in),"truncated token file");return v;
}
std::vector<uint32_t> ReadSequence(const std::string& name,uint32_t vocab,uint32_t context) {
    static_assert(std::endian::native==std::endian::little && sizeof(float)==4);
    std::ifstream in(name,std::ios::binary);char magic[8]{};in.read(magic,8);
    Need(bool(in) && std::memcmp(magic,"MICI4T1\0",8)==0,"token magic mismatch");
    Need(ReadU32(in)==vocab && ReadU32(in)==1,"one segment with matching vocabulary required");
    const auto n=ReadU32(in);Need(n>=33 && n<=2049 && n-1<=context,"invalid trajectory length");
    std::vector<uint32_t> ids(n);
    for(auto& id:ids) {id=ReadU32(in);Need(id<vocab,"token outside vocabulary");}
    Need(in.peek()==std::char_traits<char>::eof(),"trailing token data");return ids;
}
uint32_t Number(const std::string& text) {
    uint32_t value=0;const auto r=std::from_chars(text.data(),text.data()+text.size(),value);
    Need(!text.empty() && r.ec==std::errc{} && r.ptr==text.data()+text.size(),"invalid CPU number");return value;
}
void Write(const std::filesystem::path& p,const void* data,size_t bytes) {
    std::ofstream f(p,std::ios::binary);f.write(static_cast<const char*>(data),static_cast<std::streamsize>(bytes));
    f.flush();Need(bool(f),"output write failed");f.close();Need(bool(f),"output close failed");
}
double Seconds(Clock::time_point a,Clock::time_point b) {return std::chrono::duration<double>(b-a).count();}
void Memory(std::ostream& out,const char* phase,int cycle) {
    PROCESS_MEMORY_COUNTERS_EX m{};m.cb=sizeof(m);DWORD handles=0;
    Need(GetProcessMemoryInfo(GetCurrentProcess(),reinterpret_cast<PROCESS_MEMORY_COUNTERS*>(&m),sizeof(m))!=0,
         "process memory counters unavailable");
    Need(GetProcessHandleCount(GetCurrentProcess(),&handles)!=0,"handle count unavailable");
    out<<"{\"phase\":\""<<phase<<"\",\"cycle\":"<<cycle<<",\"private_bytes\":"<<m.PrivateUsage
       <<",\"peak_commit_bytes\":"<<m.PeakPagefileUsage<<",\"working_set_bytes\":"<<m.WorkingSetSize
       <<",\"peak_working_set_bytes\":"<<m.PeakWorkingSetSize<<",\"handles\":"<<handles<<"}\n";
    out.flush();Need(bool(out),"resource write failed");
}
uint32_t AC() {SYSTEM_POWER_STATUS s{};return GetSystemPowerStatus(&s)?s.ACLineStatus:255;}
}

int MiciusShell::RunI6Benchmark(const std::vector<std::string>& args) {
    try {
        const std::array<std::string,5> keys={"--i6-mode","--model","--tokens","--output","--cpu"};
        std::map<std::string,std::string> options;
        Need(args.size()==10,"I6 requires mode, model, tokens, output and CPU pairs");
        for(size_t i=0;i<args.size();i+=2) {
            Need(std::find(keys.begin(),keys.end(),args[i])!=keys.end() && !args[i+1].empty(),"unknown or empty I6 option");
            Need(options.emplace(args[i],args[i+1]).second,"duplicate I6 option");
        }
        const auto mode=options.at("--i6-mode");
        Need(mode=="parity" || mode=="timing" || mode=="resource" || mode=="lifecycle","invalid I6 mode");
        const uint32_t cpu=Number(options.at("--cpu"));
        Need(cpu<sizeof(DWORD_PTR)*8 && GetActiveProcessorGroupCount()==1,"I6 requires a valid CPU in one processor group");
        const DWORD_PTR mask=DWORD_PTR(1)<<cpu;DWORD_PTR process_mask=0,system_mask=0;
        Need(GetProcessAffinityMask(GetCurrentProcess(),&process_mask,&system_mask) && (system_mask&mask),"CPU absent");
        Need(SetProcessAffinityMask(GetCurrentProcess(),mask) && SetThreadAffinityMask(GetCurrentThread(),mask),"affinity failed");
        Need(GetPriorityClass(GetCurrentProcess())==NORMAL_PRIORITY_CLASS,"normal process priority required");
        Need(GetCurrentProcessorNumber()==cpu && (_mm_getcsr()&0xffc0U)==0x1f80U,"CPU or FP control mismatch");
        const uint32_t start_cpu=GetCurrentProcessorNumber(),ac_before=AC();
        const std::filesystem::path out(options.at("--output"));
        Need(std::filesystem::create_directory(out),"fresh output directory required");
        std::ofstream resources;
        const bool observe=mode=="resource" || mode=="lifecycle";
        if(observe) {resources.open(out/"resources.jsonl",std::ios::binary);Need(bool(resources),"resource file failed");Memory(resources,"before_load",0);}
        auto release=[&]() {
            if(ctx.engine_ctx) {Micius_DestroyEngine(ctx.engine_ctx);ctx.engine_ctx=nullptr;}
            ClearManagedBuffers(&ctx);
            if(ctx.model_parser) {delete ctx.model_parser;ctx.model_parser=nullptr;}
            Need(!ctx.engine_ctx && !ctx.model_parser && ctx.managed_buffers.empty() && ctx.dynamic_tensor_names.empty(),"ownership remained after release");
        };
        auto load=[&]() {
            const auto t=Clock::now();CmdLoad(this,&ctx,{"load",options.at("--model")});
            const double seconds=Seconds(t,Clock::now());Need(ctx.engine_ctx!=nullptr,"load/bind failed");
            const auto stats=micius::internal::GetPreparedWeightStats(ctx.engine_ctx);
            Need(stats.selection==0 && stats.extra_bytes==0 && stats.matrices==0,"unexpected weight shadow");
            Need(Micius_GetWorkerCount(ctx.engine_ctx)==0,"reference workload must have zero workers");
            for(const auto& old:micius::i25::RetirableWeights(ctx.engine_ctx))
                Need(std::find(ctx.managed_buffers.begin(),ctx.managed_buffers.end(),reinterpret_cast<void*>(old.address))==ctx.managed_buffers.end(),"old Q8 owner survived panel commit");
            return seconds;
        };
        micius::i22::Memory("before_load");
        const double load_seconds=load();
        const auto panels=micius::i25::BindingStats(ctx.engine_ctx);
        {
            std::ofstream f(out/"i25-binding.json",std::ios::binary);
            f<<"{\"active\":"<<(panels.active?"true":"false")<<",\"matrices\":"<<panels.matrices
             <<",\"payload_bytes\":"<<panels.payload_bytes<<",\"capacity_bytes\":"<<panels.capacity_bytes
             <<",\"scratch_bytes\":"<<panels.scratch_bytes<<",\"retired_weight_bytes\":"<<panels.retired_weight_bytes
             <<",\"retired_weight_buffers\":"<<panels.retired_weight_buffers
             <<",\"legacy_consumer_pointers\":"<<panels.legacy_consumer_pointers
             <<",\"legacy_scale_buffers\":"<<panels.legacy_scale_buffers<<",\"old_shell_owners_remaining\":0}\n";
            f.flush();Need(bool(f),"binding evidence write failed");f.close();Need(bool(f),"binding evidence close failed");
        }
        micius::i22::Memory("after_load");
        const auto ids=ReadSequence(options.at("--tokens"),ctx.engine_config.vocab_size,ctx.engine_config.max_seq_len);
        const auto vocab=ctx.engine_config.vocab_size;const size_t steps=ids.size()-1;
        std::vector<float> logits(vocab),reference(vocab);
        std::vector<uint32_t> predictions(steps);
        auto reset=[&]() {Need(Micius_ResetEngineV1(ctx.engine_ctx) && Micius_GetSequenceLengthV1(ctx.engine_ctx)==0,"reset failed");};
        auto copy=[&]() {Need(Micius_CopyLastLogitsV1(ctx.engine_ctx,logits.data(),logits.size()),"logits copy failed");};
        auto step=[&](size_t i) {Need(Micius_DecodeOneV1(ctx.engine_ctx,ids[i],&predictions[i])==MICIUS_STATE_YIELD_TOKEN,"decode failed");};
        double first_decode_seconds=0;
        std::vector<double> band_seconds;
        std::vector<size_t> endpoints;
        const int cycles=mode=="lifecycle"?8:1;
        if(mode=="lifecycle") {
            std::ofstream checks(out/"lifecycle.jsonl",std::ios::binary);checks<<std::setprecision(17);
            std::vector<float> first_cycle;
            for(int cycle=1;cycle<=cycles;++cycle) {
                const double loaded=cycle==1?load_seconds:load();
                Memory(resources,"after_load",cycle);reset();
                for(size_t i=0;i<32;++i)step(i);copy();reference=logits;
                if(cycle==1) first_cycle=reference;
                Need(std::memcmp(reference.data(),first_cycle.data(),vocab*4)==0,"reload logits changed");
                reset();uint32_t sentinel=0xfeedbeefU;
                Need(Micius_DecodeOneV1(ctx.engine_ctx,vocab,&sentinel)==MICIUS_STATE_INVALID_ARGUMENT && sentinel==0xfeedbeefU,"invalid token did not fail closed");
                Need(Micius_DecodeOneV1(ctx.engine_ctx,ids[0],nullptr)==MICIUS_STATE_INVALID_ARGUMENT,"null output accepted");
                std::fill(logits.begin(),logits.end(),123.0f);
                Need(!Micius_CopyLastLogitsV1(ctx.engine_ctx,logits.data(),vocab-1) &&
                     std::all_of(logits.begin(),logits.end(),[](float x){return x==123.0f;}),"short copy did not fail closed");
                Need(Micius_GetSequenceLengthV1(ctx.engine_ctx)==0,"failed call advanced context");
                Need(!Micius_CopyLastLogitsV1(ctx.engine_ctx,logits.data(),vocab),"reset exposed stale logits");
                for(size_t i=0;i<32;++i)step(i);copy();
                Need(Micius_GetSequenceLengthV1(ctx.engine_ctx)==32 && std::memcmp(logits.data(),reference.data(),vocab*4)==0,"reset replay changed logits");
                Memory(resources,"after_replay",cycle);release();Memory(resources,"after_release",cycle);
                checks<<"{\"cycle\":"<<cycle<<",\"load_seconds\":"<<loaded<<",\"checks_passed\":true}\n";checks.flush();Need(bool(checks),"lifecycle record failed");
            }
            predictions.resize(32);
        } else {
            if(observe)Memory(resources,"after_load",1);
            reset();
            if(mode!="resource") {for(size_t i=0;i<32;++i)step(i);reset();}
            if(mode=="parity")micius::i22::Memory("after_warmup_reset");
            if(mode=="timing") {
                // Only decode, status checks, output stores and five clock reads in the ROI.
                std::array<Clock::time_point,5> stamps{};size_t pos=0,index=0;
                stamps[0]=Clock::now();
                for(const size_t end: {size_t(256),size_t(512),size_t(1024),size_t(2048)}) {
                    const size_t limit=std::min(steps,end);
                    for(;pos<limit;++pos)step(pos);
                    stamps[++index]=Clock::now();
                    if(pos==steps)break;
                }
                for(size_t i=0;i<index;++i)band_seconds.push_back(Seconds(stamps[i],stamps[i+1]));
                for(const size_t end: {size_t(256),size_t(512),size_t(1024),size_t(2048)}) {
                    endpoints.push_back(std::min(steps,end));if(end>=steps)break;
                }
            } else {
                std::ofstream raw;
                if(mode=="parity") {raw.open(out/"logits.f32",std::ios::binary);Need(bool(raw),"raw logits open failed");}
                for(size_t i=0;i<steps;++i) {
                    if(mode=="resource" && i==0) {
                        const auto begin=Clock::now();step(i);first_decode_seconds=Seconds(begin,Clock::now());
                        Memory(resources,"after_first_decode",1);
                    } else step(i);
                    if(mode=="parity") {
                        if(i+1==256 || i+1==512 || i+1==1024 || i+1==2048)
                            micius::i22::Memory(("after_decode_"+std::to_string(i+1)).c_str());
                        Need(Micius_GetSequenceLengthV1(ctx.engine_ctx)==i+1,"prediction position mismatch");copy();
                        raw.write(reinterpret_cast<const char*>(logits.data()),vocab*4);Need(bool(raw),"raw logits write failed");
                    }
                }
                if(mode=="parity") {raw.flush();Need(bool(raw),"raw flush failed");raw.close();Need(bool(raw),"raw close failed");}
            }
            Need(Micius_GetSequenceLengthV1(ctx.engine_ctx)==steps,"final sequence length mismatch");copy();
            if(observe) {Memory(resources,"after_decode",1);release();Memory(resources,"after_release",1);}
        }
        Need((_mm_getcsr()&0xffc0U)==0x1f80U && GetCurrentProcessorNumber()==cpu,"final CPU/FP mismatch");
        Need(GetProcessAffinityMask(GetCurrentProcess(),&process_mask,&system_mask) && process_mask==mask,"process affinity changed");
        const auto ac_after=AC();
        Write(out/"predictions.u32",predictions.data(),predictions.size()*4);
        Write(out/"last-logits.f32",logits.data(),logits.size()*4);
        std::ofstream result(out/"result.json",std::ios::binary);result<<std::setprecision(17);
        result<<"{\"status\":\"COMPLETE\",\"mode\":\""<<mode<<"\",\"i1_mode\":"<<MICIUS_I1_MODE
          <<",\"abi\":"<<MICIUS_ABI_VERSION<<",\"vocab\":"<<vocab<<",\"steps\":"<<steps
          <<",\"warmup_steps\":"<<((mode=="parity" || mode=="timing")?32:0)<<",\"cycles\":"<<cycles
          <<",\"requested_cpu\":"<<cpu<<",\"start_cpu\":"<<start_cpu<<",\"end_cpu\":"<<GetCurrentProcessorNumber()
          <<",\"affinity_mask\":"<<process_mask<<",\"ac_before\":"<<ac_before<<",\"ac_after\":"<<ac_after
          <<",\"mxcsr\":"<<_mm_getcsr()<<",\"pw2_extra_bytes\":0,\"load_seconds\":"<<load_seconds
          <<",\"first_decode_seconds\":"<<first_decode_seconds<<",\"band_endpoints\":[";
        for(size_t i=0;i<endpoints.size();++i)result<<(i?",":"")<<endpoints[i];result<<"],\"band_seconds\":[";
        for(size_t i=0;i<band_seconds.size();++i)result<<(i?",":"")<<band_seconds[i];result<<"]}\n";
        result.flush();Need(bool(result),"result flush failed");result.close();Need(bool(result),"result close failed");
        if(observe) {resources.close();Need(bool(resources),"resource close failed");}
        if(micius::i22::Observing()) {release();micius::i22::Memory("after_release");}
        std::cout<<"mode=i6_"<<mode<<"\nstatus=PASS\n";return 0;
    } catch(const std::exception& error) {std::cout<<"status=FAIL\nerror="<<error.what()<<'\n';return 1;}
}

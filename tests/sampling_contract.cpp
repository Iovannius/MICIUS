#include "micius_sampling.h"
#include "session/SamplingConfig.h"
#include <cmath>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>
static void Check(bool b) { if(!b)throw std::runtime_error("sampling contract"); }
int main() { try {
    MiciusSamplingOptionsV1 p{};Check(Micius_InitSamplingV1(&p,sizeof(p)));
    float v[]={1,3,3,-1};uint32_t out=99;uint64_t rng=0;
    auto choose=[&](const uint32_t* h=nullptr,size_t n=0,const uint8_t* mask=nullptr) {
        return Micius_SampleTokenV1(&p,v,4,mask,h,n,&rng,&out);
    };
    Check(choose() && out==1 && rng==0); // tie is lowest ID, no random draw
    p.repeat_penalty=2;p.repeat_last_n=8;uint32_t history[]={1,1};
    Check(choose(history,2) && out==2 && v[1]==3); // applied once, no logits mutation
    p.temperature=1;p.top_k=1;
    Check(choose(history,2) && out==2 && rng==UINT64_C(0x9e3779b97f4a7c15));
    p.repeat_penalty=1;p.top_k=0;p.top_p=0.01;
    for(int i=0;i<50;++i)Check(choose() && out==1); // nucleus must retain at least one
    p.top_p=1;p.top_k=2;rng=1234;
    std::vector<uint32_t> ids;
    for(int i=0;i<32;++i) { Check(choose());Check(out==1 || out==2);ids.push_back(out); }
    rng=1234;for(auto id:ids)Check(choose() && out==id);
    // A known SplitMix64(0) draw is ~0.88331: equal two-token CDF selects ID 2.
    rng=0;Check(choose() && out==2);
    uint8_t mask[]={0,0,0,1};Check(choose(nullptr,0,mask) && out==3);
    uint8_t none[4]={};auto old=rng;out=99;
    Check(!choose(nullptr,0,none) && out==99 && rng==old);
    p.temperature=-1;Check(!choose() && rng==old && out==99);
    p.temperature=1;p.top_p=std::numeric_limits<double>::quiet_NaN();Check(!choose());
    p.top_p=1;v[2]=std::numeric_limits<float>::infinity();Check(!choose() && rng==old);
    using namespace micius::session;
    auto cfg=ParseConfig("version=1\r\ntemperature=0.7\ntop_k=40\ntop_p=0.9\nseed=18446744073709551615\n");
    auto again=ParseConfig(WriteConfig(cfg));Check(again.seed==UINT64_MAX && again.sample.temperature==0.7);
    for(auto bad:{"", "version=2", "version=1\nseed=-1", "version=1\ntemperature=nan", "version=1\ntop_p=0",
        "version=1\nfoo=2", "version=1\nversion=1", "version=1\nmax_new_tokens=0", "version=1\ntop_k=1junk"}) {
        bool rejected=false;try { ParseConfig(bad); }catch(...){rejected=true;}Check(rejected);
    }
    std::cout<<"Sampling ordering, seeded replay, filtering, rejection and config contract: PASS\n";return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<'\n';return 1;} }

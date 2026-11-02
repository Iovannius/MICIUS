#include "AI/I4MixedPrecision.h"
#include "app/I4Evaluation.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>
#ifdef NDEBUG
#error "I4 tests require active assertions"
#endif
namespace r8_frozen {
Q8_32_GEMVStatus RowPair(const int8_t*,const QuantDescriptor*,const float*,float*,uint32_t,uint32_t) noexcept;
}
using namespace micius::i4;
template<class F> void rejects(F f) {bool caught=false;try{f();}catch(const std::exception&){caught=true;}assert(caught);}
void U32(std::ostream& f,uint32_t v){f.write(reinterpret_cast<const char*>(&v),4);}
void Tokens(const std::filesystem::path& p,const std::vector<std::vector<uint32_t>>& seqs) {
    std::ofstream f(p,std::ios::binary);f.write("MICI4T1\0",8);U32(f,128);U32(f,static_cast<uint32_t>(seqs.size()));
    for(const auto& seq:seqs){U32(f,static_cast<uint32_t>(seq.size()));for(auto t:seq)U32(f,t);}
}
struct Model {
    MiciusConfig cfg{};MiciusContext* ctx=nullptr;
    std::vector<float> embeddings,norm;
    std::vector<int8_t> weights,output;
    std::vector<MiciusQ8Block32Scale> scales;
    std::vector<std::string> names;
    std::vector<MiciusTensorEntryV1> entries;
    Model() {
        assert(Micius_InitConfig(&cfg,sizeof(cfg)));
        cfg.hidden_dim=64;cfg.vocab_size=128;cfg.max_seq_len=8;cfg.kv_pool_size_mb=1;
        cfg.worker_threads=0;cfg.num_layers=32;cfg.num_heads=1;cfg.num_kv_heads=1;cfg.head_dim=64;
        cfg.intermediate_dim=64;cfg.execution_mode=MICIUS_EXECUTION_REFERENCE;cfg.rope_dimension_count=64;
        embeddings.resize(128*64);norm.assign(64,1);weights.resize(64*64);output.resize(128*64);
        scales.assign(128*2,MiciusQ8Block32Scale{1.0f/128,0,{0,0}});
        std::mt19937 rng(1104);
        for(auto& v:embeddings)v=static_cast<float>(static_cast<int>(rng()%2001)-1000)/1000.0f;
        for(auto& v:weights)v=static_cast<int8_t>(static_cast<int>(rng()%7)-3);
        for(auto& v:output)v=static_cast<int8_t>(static_cast<int>(rng()%17)-8);
        names.reserve(1024);entries.reserve(1024);
        auto add=[&](std::string name,const void* data,uint32_t type,uint64_t rows,uint64_t cols,size_t bytes,size_t stride) {
            names.push_back(std::move(name));MiciusTensorEntryV1 e{};e.tensor_name=names.back().c_str();e.tensor_data=data;
            e.byte_size=bytes;e.data_type=type;e.rank=cols?2:1;e.dimensions[0]=rows;e.dimensions[1]=cols;e.row_stride_bytes=stride;entries.push_back(e);
        };
        auto f32=[&](std::string name,const float* data,uint64_t rows,uint64_t cols) {
            add(name,data,MICIUS_TENSOR_F32,rows,cols,static_cast<size_t>(rows*(cols?cols:1))*4,static_cast<size_t>(cols)*4);
        };
        auto q8=[&](std::string name,const int8_t* data,uint64_t rows) {
            add(name+".weight",data,MICIUS_TENSOR_Q8_32_WEIGHT,rows,64,static_cast<size_t>(rows)*64,64);
            add(name+".scales",scales.data(),MICIUS_TENSOR_Q8_32_SCALE,rows,2,static_cast<size_t>(rows)*2*sizeof(MiciusQ8Block32Scale),2*sizeof(MiciusQ8Block32Scale));
        };
        f32("token_embd.weight.f32",embeddings.data(),128,64);f32("output_norm.weight",norm.data(),64,0);
        q8("output",output.data(),128);
        for(unsigned layer=0;layer<32;++layer) {
            const auto prefix="blk."+std::to_string(layer)+".";
            f32(prefix+"attn_norm.weight",norm.data(),64,0);f32(prefix+"ffn_norm.weight",norm.data(),64,0);
            for(const char* op:{"attn_q","attn_k","attn_v","attn_output","ffn_gate","ffn_up","ffn_down"})q8(prefix+op,weights.data(),64);
        }
        ctx=Micius_CreateEngineV1(&cfg,sizeof(cfg),MICIUS_ABI_VERSION);assert(ctx);
        assert(Micius_BindModelTensorsV1(ctx,entries.data(),static_cast<uint32_t>(entries.size()),sizeof(MiciusTensorEntryV1),MICIUS_ABI_VERSION));
    }
    ~Model(){if(ctx)Micius_DestroyEngine(ctx);}
};
int main() {
    const auto saved=_mm_getcsr();_mm_setcsr(0x1f80U);
    Score score;
    std::vector<float> v(49152,0);
    assert(ScoreLogits(v,123,score) && std::abs(score.nll-std::log(49152.0))<1e-13 && score.top1==0);
    v={-4,-1,2,5};
    for(uint32_t target=0;target<4;++target) {
        assert(ScoreLogits(v,target,score));long double sum=0;
        for(auto x:v)sum+=std::exp(static_cast<long double>(x));
        assert(std::abs(static_cast<long double>(score.nll)-(std::log(sum)-v[target]))<1e-12L);
        auto shifted=v;for(auto& x:shifted)x+=10000;Score other;assert(ScoreLogits(shifted,target,other));
        assert(score.nll==other.nll);
    }
    v={std::numeric_limits<float>::max(),-std::numeric_limits<float>::max(),std::numeric_limits<float>::max()};
    assert(ScoreLogits(v,0,score) && score.nll==std::log(2.0));
    v[0]=std::numeric_limits<float>::infinity();assert(!ScoreLogits(v,0,score));assert(!ScoreLogits({},0,score));
    std::mt19937 rng(1104);size_t cases=0;
    for(uint32_t cols:{1U,15U,31U,32U,33U,63U,65U,960U,2560U})for(uint32_t rows:{1U,2U,3U}) {
        std::vector<float> x(cols),expected(rows),actual(rows+2,-99);
        std::vector<int8_t> w(static_cast<size_t>(cols)*rows);
        std::vector<QuantDescriptor> d(static_cast<size_t>((cols+31)/32)*rows,QuantDescriptor(0.015625f,0));
        for(auto& a:x)a=static_cast<float>(static_cast<int>(rng()%2001)-1000)/321.0f;
        for(auto& a:w)a=static_cast<int8_t>(static_cast<int>(rng()%256)-128);
        const auto before=x;
        assert(r8_frozen::RowPair(w.data(),d.data(),x.data(),expected.data(),cols,rows)==Q8_32_GEMVStatus::OK);
        for(bool validated:{false,true}) {
            assert(F64(w.data(),d.data(),x.data(),actual.data()+1,cols,rows,validated)==Q8_32_GEMVStatus::OK);
            assert(std::memcmp(expected.data(),actual.data()+1,rows*4U)==0 && x==before && actual.front()==-99 && actual.back()==-99);
        }
        ++cases;
    }
    for(unsigned p=0;p<4;++p) {
        unsigned count=0;
        for(uint32_t l=0;l<32;++l)for(unsigned op=0;op<5;++op)if(Select(static_cast<Policy>(p),l,static_cast<Op>(op))) {
            ++count;
            if(p==1)assert(l==31 && (op==2 || op==3));
            if(p==2)assert((l==3 || l==31) && op>=2);
        }
        if(Select(static_cast<Policy>(p),32,Op::LM))++count;
        assert(count==(p==0?0:p==1?2:p==2?6:161));
    }
    const auto root=std::filesystem::current_path()/("i4-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(root);
    const std::vector<std::vector<uint32_t>> sequences={{17,3,99,5,45},{4,2,3,1},{17,3,99,5,45}};
    Tokens(root/"tokens.bin",sequences);assert(ReadTokens((root/"tokens.bin").string(),128,8)==sequences);
    rejects([&]{ReadTokens((root/"tokens.bin").string(),127,8);});
    rejects([&]{ReadTokens((root/"tokens.bin").string(),128,2);});
    Tokens(root/"bad.bin",{{128,2}});rejects([&]{ReadTokens((root/"bad.bin").string(),128,8);});
    Tokens(root/"short.bin",{{1}});rejects([&]{ReadTokens((root/"short.bin").string(),128,8);});
    {std::ofstream f(root/"tokens.bin",std::ios::binary|std::ios::app);f.put('X');}
    rejects([&]{ReadTokens((root/"tokens.bin").string(),128,8);});Tokens(root/"tokens.bin",sequences);
    {
        Model model;
        assert(EvaluateTokens(model.ctx,model.cfg,(root/"tokens.bin").string(),(root/"result").string())==0);
        assert(EvaluateTokens(model.ctx,model.cfg,(root/"tokens.bin").string(),(root/"result").string())!=0);
        std::ifstream observed(root/"result/logits.f32",std::ios::binary);
        std::vector<float> expected(128),actual(128);unsigned nongreedy_targets=0;
        for(const auto& seq:sequences) {
            assert(Micius_ResetEngineV1(model.ctx));
            for(size_t i=0;i+1<seq.size();++i) {
                uint32_t prediction=0;
                assert(Micius_DecodeOneV1(model.ctx,seq[i],&prediction)==MICIUS_STATE_YIELD_TOKEN);
                assert(Micius_CopyLastLogitsV1(model.ctx,expected.data(),expected.size()));
                observed.read(reinterpret_cast<char*>(actual.data()),128*4);
                assert(observed && std::memcmp(expected.data(),actual.data(),128*4)==0);
                nongreedy_targets+=prediction!=seq[i+1];
            }
        }
        assert(nongreedy_targets>0 && observed.peek()==std::char_traits<char>::eof());
    }
    {RouteAudit audit(32);assert(audit.good());assert(RecordSelected(0,Op::LM));assert(!audit.finish());}
    {RouteAudit first(32);RouteAudit nested(32);assert(first.good() && !nested.good());}
    _mm_setcsr(saved);
    std::cout<<"I4 PASS: NLL analytic/long-double checks; "<<cases<<" shapes x 2 frozen F64 routes; policies; token-file rejection; teacher forcing vs independent decode client and A/B/A reset\n";
}

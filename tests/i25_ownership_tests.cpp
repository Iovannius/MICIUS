#include "micius_runtime.h"
#include "AI/I25Binding.h"
#include "app/I25LoaderLifetime.h"
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <immintrin.h>
#include <iostream>
#include <new>
#include <string>
#include <vector>

// Exhaust each C++ allocation site of a successful rebind. No injection hook
// is compiled into the production engine or selected by its environment.
static thread_local long fail_after=-1;
void* operator new(size_t n) {
    if(fail_after==0)throw std::bad_alloc();
    if(fail_after>0)--fail_after;
    if(void* p=std::malloc(n?n:1))return p;
    throw std::bad_alloc();
}
void operator delete(void* p) noexcept {std::free(p);}
void* operator new[](size_t n) {return ::operator new(n);}
void operator delete[](void* p) noexcept {::operator delete(p);}

struct Fixture {
    MiciusConfig cfg{};
    std::vector<std::string> names;
    std::vector<MiciusTensorEntryV1> entries;
    std::vector<void*> owners;
    std::vector<float> embedding,norm;
    Fixture(unsigned vocab=128) {
        assert(Micius_InitConfig(&cfg,sizeof(cfg)));
        cfg.hidden_dim=64;cfg.intermediate_dim=64;cfg.vocab_size=vocab;
        cfg.max_seq_len=64;cfg.kv_pool_size_mb=1;cfg.worker_threads=0;
        cfg.num_layers=1;cfg.num_heads=1;cfg.num_kv_heads=1;cfg.head_dim=64;
        cfg.rope_dimension_count=64;cfg.execution_mode=MICIUS_EXECUTION_REFERENCE;
        names.reserve(32);entries.reserve(32);owners.reserve(32);
        embedding.resize(vocab*64);norm.assign(64,1.f);
        for(size_t i=0;i<embedding.size();++i)embedding[i]=float(int(i%97)-48)*.0078125f;
        Add("token_embd.weight.f32",embedding.data(),MICIUS_TENSOR_F32,vocab,64,4);
        Add("output_norm.weight",norm.data(),MICIUS_TENSOR_F32,64,0,4);
        Add("blk.0.attn_norm.weight",norm.data(),MICIUS_TENSOR_F32,64,0,4);
        Add("blk.0.ffn_norm.weight",norm.data(),MICIUS_TENSOR_F32,64,0,4);
        for(auto name:{"attn_q","attn_k","attn_v","attn_output","ffn_gate","ffn_up","ffn_down"})Q8(std::string("blk.0.")+name,64);
        Q8("output",vocab);
    }
    ~Fixture(){for(auto p:owners)_mm_free(p);}
    void Add(std::string n,const void* p,uint32_t type,size_t rows,size_t cols,size_t element) {
        names.push_back(std::move(n));MiciusTensorEntryV1 e{};
        e.tensor_name=names.back().c_str();e.tensor_data=p;e.data_type=type;
        e.rank=cols?2:1;e.dimensions[0]=rows;e.dimensions[1]=cols;
        e.byte_size=rows*(cols?cols:1)*element;e.row_stride_bytes=cols*element;
        entries.push_back(e);
    }
    void Q8(const std::string& n,size_t rows) {
        auto* w=static_cast<int8_t*>(_mm_malloc(rows*64,64));assert(w);owners.push_back(w);
        for(size_t i=0;i<rows*64;++i)w[i]=int8_t(int((i*13+rows)%255)-127);
        auto* s=static_cast<MiciusQ8Block32Scale*>(_mm_malloc(rows*2*sizeof(MiciusQ8Block32Scale),64));assert(s);owners.push_back(s);
        for(size_t i=0;i<rows*2;++i)s[i]={.000244140625f,0,{0,0}};
        Add(n+".weight",w,MICIUS_TENSOR_Q8_32_WEIGHT,rows,64,1);
        Add(n+".scales",s,MICIUS_TENSOR_Q8_32_SCALE,rows,2,sizeof(*s));
    }
    bool Bind(MiciusContext* c){bool ok=Micius_BindModelTensorsV1(c,entries.data(),uint32_t(entries.size()),sizeof(entries[0]),MICIUS_ABI_VERSION);if(!ok&&fail_after<0)std::cerr<<Micius_GetLastError()<<'\n';return ok;}
};
static std::vector<float> Step(MiciusContext* c,unsigned vocab,unsigned t=1) {
    uint32_t token=0;assert(Micius_DecodeOneV1(c,t,&token)==MICIUS_STATE_YIELD_TOKEN);
    std::vector<float> x(vocab);assert(Micius_CopyLastLogitsV1(c,x.data(),x.size()));return x;
}
static void Transactions() {
    Fixture f;auto* c=Micius_CreateEngineV1(&f.cfg,sizeof(f.cfg),MICIUS_ABI_VERSION);assert(c&&f.Bind(c));
    assert(micius::i25::BindingStats(c).active);
    auto expected=Step(c,128);Step(c,128,2);std::vector<float> last(128);
    assert(Micius_CopyLastLogitsV1(c,last.data(),128));
    size_t failures=0;
    for(long n=0;n<2000;++n) {
        fail_after=n;bool ok=f.Bind(c);fail_after=-1;
        if(ok)break;
        ++failures;assert(Micius_GetSequenceLengthV1(c)==2);
        std::vector<float> x(128);assert(Micius_CopyLastLogitsV1(c,x.data(),128));assert(std::memcmp(last.data(),x.data(),512)==0);
    }
    assert(failures>20&&failures<2000);assert(Micius_GetSequenceLengthV1(c)==0);
    assert(Step(c,128)==expected);assert(Micius_ResetEngineV1(c));
    // After actual retirement, overwrite every surviving public descriptor.
    // Neither the packed weights nor scales may retain the old data pointers.
    size_t released=0;assert(micius::i25::ReleaseWeights(f.owners,f.entries,micius::i25::RetirableWeights(c),released));
    assert(released==7*64*64+128*64);
    for(const auto& e:f.entries)if(e.data_type==MICIUS_TENSOR_Q8_32_SCALE)std::memset(const_cast<void*>(e.tensor_data),0xff,e.byte_size);
    assert(Step(c,128)==expected);
    uint32_t token=123;assert(Micius_DecodeOneV1(c,128,&token)==MICIUS_STATE_INVALID_ARGUMENT&&token==123);
    assert(Micius_ResetEngineV1(c));assert(Step(c,128)==expected);
    Micius_DestroyEngine(c);
    std::cout<<"transaction_allocation_failures="<<failures<<"; retired data no longer consumed\n";
}
static void RetirementValidation() {
    Fixture f;auto* c=Micius_CreateEngineV1(&f.cfg,sizeof(f.cfg),MICIUS_ABI_VERSION);assert(c&&f.Bind(c));
    auto set=micius::i25::RetirableWeights(c);size_t released=0;const auto before=f.owners;
    auto bad=f.entries;auto e=bad[4];e.tensor_data=reinterpret_cast<const void*>(set[0].address+1);e.byte_size=4;e.data_type=MICIUS_TENSOR_F32;bad.push_back(e);
    assert(!micius::i25::ReleaseWeights(f.owners,bad,set,released));assert(f.owners==before);
    f.owners.push_back(f.owners.front());assert(!micius::i25::ReleaseWeights(f.owners,f.entries,set,released));f.owners.pop_back();assert(f.owners==before);
    // An exact Q8 alias in the Shell metadata is retired once, not twice.
    for(const auto& x:f.entries)if(reinterpret_cast<uintptr_t>(x.tensor_data)==set[0].address){bad=f.entries;bad.push_back(x);break;}
    assert(micius::i25::ReleaseWeights(f.owners,bad,set,released));assert(!micius::i25::ReleaseWeights(f.owners,bad,set,released));
    Step(c,128);Micius_DestroyEngine(c);
    Fixture tail(127);c=Micius_CreateEngineV1(&tail.cfg,sizeof(tail.cfg),MICIUS_ABI_VERSION);assert(c&&tail.Bind(c));
    assert(!micius::i25::BindingStats(c).active&&micius::i25::RetirableWeights(c).empty());
    auto owners=tail.owners;assert(micius::i25::ReleaseWeights(tail.owners,tail.entries,{},released)&&released==0&&owners==tail.owners);
    Step(c,127);Micius_DestroyEngine(c);
}
int main(){_mm_setcsr(0x1f80);Transactions();RetirementValidation();std::cout<<"I25 ownership and rollback PASS\n";}

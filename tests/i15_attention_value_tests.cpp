#include "../src/AI/I15AttentionValue.h"
#include <immintrin.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

namespace {
size_t cases = 0;
void Need(bool x, const char* why) { if (!x) throw std::runtime_error(why); }
bool Same(const std::vector<float>& a, const std::vector<float>& b) {
    return a.size()==b.size() && std::memcmp(a.data(),b.data(),a.size()*4)==0;
}
void Compare(const std::vector<float>& p, double sum, const std::vector<float>& v,
             size_t stride, uint32_t dim, unsigned control=0x1f80U) {
    const auto ps=p, vs=v;
    std::vector<float> a(dim+2,-123.0f),b=a;
    _mm_setcsr(control);
    auto sa=AccumulateAttentionValuesF32_AVX2(p.data(),sum,v.data(),stride,a.data()+1,
                                            static_cast<uint32_t>(p.size()),dim);
    unsigned fa=_mm_getcsr(); _mm_setcsr(control);
    auto sb=AccumulateAttentionValuesI15(p.data(),sum,v.data(),stride,b.data()+1,
                                        static_cast<uint32_t>(p.size()),dim);
    unsigned fb=_mm_getcsr();
    Need(sa==sb,"status differs");Need(Same(a,b),"output / error partial output differs");
    Need(fa==fb,"MXCSR flags or control differs");
    Need(Same(ps,p)&&Same(vs,v),"input modified");
    Need(a.front()==-123.0f&&a.back()==-123.0f,"output canary changed");
    if (sa==AttentionValueStatus::OK) {
        _mm_setcsr(control);
        std::vector<float> oracle(dim,0.0f);
        for (size_t t=0;t<p.size();++t) {
            volatile double prob=static_cast<double>(p[t])/sum;
            for (size_t d=0;d<dim;++d) {
                volatile double x=std::fma(prob,static_cast<double>(v[t*stride+d]),static_cast<double>(oracle[d]));
                volatile float rounded=static_cast<float>(x);oracle[d]=rounded;
            }
        }
        Need(std::memcmp(oracle.data(),b.data()+1,dim*4)==0,"independent ordered F64/F32 oracle differs");
    }
    _mm_setcsr(0x1f80U);++cases;
}
void RandomShapes() {
    std::mt19937 rng(101064);
    for (uint32_t n:{1U,2U,3U,31U,32U,33U,63U,64U,65U,127U,128U,129U,255U,257U,2048U,8192U})
        for (uint32_t dim:{1U,3U,4U,31U,32U,63U,64U,65U,128U}) {
            const size_t stride=dim+(n%3)*32;
            std::vector<float> p(n),v((n-1)*stride+dim);double sum=0;
            for (auto& x:p) {x=static_cast<float>(rng()%1024+1)/1024.f;sum+=x;}
            for (auto& x:v) x=static_cast<float>(static_cast<int>(rng()%32769)-16384);
            Compare(p,sum,v,stride,dim);
        }
    for (unsigned round=0;round<4;++round) for (unsigned flush=0;flush<4;++flush) {
        std::vector<float> p(129),v(128*320+64);
        for (auto& x:p) x=static_cast<float>(int(rng()%33)-16)/16.f;
        for (auto& x:v) x=std::ldexp(float(int(rng()%32769)-16384),int(rng()%170)-149);
        Compare(p,13.75,v,320,64,0x1f80U+(round<<13)+(flush&1?0x40:0)+(flush&2?0x8000:0));
    }
    std::vector<float> p={1,1,1},v(192);
    std::fill(v.begin(),v.begin()+64,16777216.f);
    std::fill(v.begin()+64,v.begin()+128,1.f);
    std::fill(v.begin()+128,v.end(),-16777216.f);
    Compare(p,1,v,64,64); // F64 across all time would incorrectly retain 1.
    // Exact arithmetic with exceptions unmasked exercises the original-order fallback.
    Compare({1},1,std::vector<float>(64,1),64,64,0);
}
void RangeContract() {
    std::mt19937 rng(150032);
    for(uint32_t n:{128U,129U,8192U,8193U}) {
        for(float mag:{0x1p100f,std::nextafter(0x1p100f,INFINITY),std::numeric_limits<float>::max()}) {
            std::vector<float> p(n),v(static_cast<size_t>(n)*64);
            for(size_t i=0;i<n;++i)p[i]=(i&1)?-1.f:1.f;
            for(auto& x:v)x=(rng()&1)?mag:-mag;
            for(unsigned mode=0;mode<4;++mode)Compare(p,1.0,v,64,64,0x1f80U+(mode<<13));
            p[32]=1.25f;Compare(p,1.0,v,64,64); // Probability bound fallback.
        }
    }
    for(unsigned i=0;i<128;++i) {
        std::vector<float> p(129),v(129*64);
        for(auto& x:p)x=std::bit_cast<float>(rng());
        for(auto& x:v)x=std::bit_cast<float>(rng());
        Compare(p,13.0,v,64,64); // Arbitrary payloads, including SNaN and infinities.
    }
}
void TileBoundaries() {
    for (uint32_t n:{128U,129U,159U,160U,161U,2048U}) {
        std::vector<float> p(n,1.f),v(static_cast<size_t>(n)*64,0.f);
        // A subnormal at the final step must survive with DAZ-only enabled.
        // At a tile boundary the next update must flush it exactly as the old path.
        for (unsigned flush=0;flush<4;++flush) for(unsigned round=0;round<4;++round) {
            unsigned control=0x1f80U+(round<<13)+(flush&1?0x40:0)+(flush&2?0x8000:0);
            for(size_t when:{size_t{31},size_t{32},size_t(n-1)}) {
                std::fill(v.begin(),v.end(),0.f);
                for(size_t d=0;d<64;++d)v[when*64+d]=std::numeric_limits<float>::min();
                Compare(p,2.0,v,64,64,control);
            }
        }
        std::fill(v.begin(),v.end(),std::numeric_limits<float>::max());
        Compare(p,1.0,v,64,64); // Range overflow requires original error replay.
        p[32]=std::numeric_limits<float>::infinity();Compare(p,1.0,v,64,64);
        p[32]=std::bit_cast<float>(uint32_t{0x7f800001});Compare(p,1.0,v,64,64);
    }
    // Per-time rounding witness on the optimized (not short fallback) path.
    std::vector<float> p(128,1.f),v(128*64,0.f);
    std::fill(v.begin(),v.begin()+64,16777216.f);
    std::fill(v.begin()+64,v.begin()+128,1.f);
    std::fill(v.begin()+128,v.begin()+192,-16777216.f);
    Compare(p,1,v,64,64);
}
void Failures() {
    const float mx=std::numeric_limits<float>::max(), inf=std::numeric_limits<float>::infinity();
    for (float bad:{inf,-inf,std::numeric_limits<float>::quiet_NaN()})
        for (size_t t:{0U,31U,32U,63U,64U,127U,128U}) for (size_t d=0;d<64;++d) {
            std::vector<float> p(129,1.f),v(129*64,0.5f);v[t*64+d]=bad;
            Compare(p,129,v,64,64);
        }
    for (double sum:{0.0,-1.0,std::numeric_limits<double>::infinity(),std::numeric_limits<double>::quiet_NaN()})
        Compare(std::vector<float>(3,1),sum,std::vector<float>(192,1),64,64);
    Compare({1,1,1},1,std::vector<float>(192,mx),64,64);
    Compare({1,std::ldexp(1.f,-52)},1,std::vector<float>(128,mx),64,64);
    Compare({1,inf,1},3,std::vector<float>(192,1),64,64);
    Compare({1,1},std::numeric_limits<double>::denorm_min(),std::vector<float>(128,1),64,64);
    // Invalid dimensions/stride/pointers must retain the original output and status.
    float p[2]={1,1},v[64]={},a[66],b[66];
    for (int which=0;which<7;++which) {
        std::fill_n(a,66,17.f);std::fill_n(b,66,17.f);
        size_t stride=which==4?63:which==5?SIZE_MAX:64;
        auto call=[&](auto fn,float* out) {return fn(which==0?nullptr:p,2,
            which==1?nullptr:v,stride,which==2?nullptr:out,
            which==3?0U:2U,which==6?0U:64U);};
        Need(call(AccumulateAttentionValuesF32_AVX2,a)==call(AccumulateAttentionValuesI15,b),"argument status differs");
        Need(std::memcmp(a,b,sizeof(a))==0,"invalid argument output differs");++cases;
    }
}
void Aliases() {
    // output overlaps values, probabilities, and both; disjoint input arrays are
    // not part of the public kernel contract, so those cases must use fallback.
    for (size_t offset:{0U,1U,32U,64U,128U,256U,300U}) {
        std::vector<float> a(512),b;
        for (size_t i=0;i<a.size();++i)a[i]=float(i%13)/13.f;
        b=a;
        auto x=AccumulateAttentionValuesF32_AVX2(a.data()+300,3,a.data(),64,a.data()+offset,3,64);
        auto y=AccumulateAttentionValuesI15(b.data()+300,3,b.data(),64,b.data()+offset,3,64);
        Need(x==y&&Same(a,b),"aliased output differs");++cases;
    }
    for(size_t offset:{0U,1U,63U,64U,4096U,8256U,8300U}) {
        std::vector<float> a(8512,0.5f),b=a;
        auto x=AccumulateAttentionValuesF32_AVX2(a.data()+8300,129,a.data(),64,a.data()+offset,129,64);
        auto y=AccumulateAttentionValuesI15(b.data()+8300,129,b.data(),64,b.data()+offset,129,64);
        Need(x==y&&Same(a,b),"long aliased output differs");++cases;
    }
    Compare(std::vector<float>(128,1),128,std::vector<float>(128*64,1),64,64,0);
}
void GuardPages() {
#ifdef _WIN32
    SYSTEM_INFO info;GetSystemInfo(&info);size_t page=info.dwPageSize;
    auto alloc=[&]() {
        auto p=static_cast<char*>(VirtualAlloc(nullptr,page*2,MEM_COMMIT|MEM_RESERVE,PAGE_READWRITE));
        Need(p!=nullptr,"guard allocation failed");DWORD old;
        Need(VirtualProtect(p+page,page,PAGE_NOACCESS,&old)!=0,"guard protection failed");return p;
    };
    char* ps=alloc(),*vs=alloc(),*os=alloc();
    float* p=reinterpret_cast<float*>(ps+page-4),*v=reinterpret_cast<float*>(vs+page-256),
         *o=reinterpret_cast<float*>(os+page-256);
    *p=1;std::fill_n(v,64,3.0f);
    Need(AccumulateAttentionValuesI15(p,1,v,SIZE_MAX/4,o,1,64)==AttentionValueStatus::OK,"guard result");
    for (int i=0;i<64;++i)Need(o[i]==3.f,"guard value");
    VirtualFree(ps,0,MEM_RELEASE);VirtualFree(vs,0,MEM_RELEASE);VirtualFree(os,0,MEM_RELEASE);++cases;
    for(uint32_t n:{128U,129U,159U}) {
        auto sized=[&](size_t bytes) {
            const size_t capacity=(bytes+page-1)/page*page;
            auto base=static_cast<char*>(VirtualAlloc(nullptr,capacity+page,MEM_COMMIT|MEM_RESERVE,PAGE_READWRITE));
            Need(base!=nullptr,"sized guard allocation failed");DWORD old;
            Need(VirtualProtect(base+capacity,page,PAGE_NOACCESS,&old)!=0,"sized guard protection failed");
            return std::pair<char*,float*>(base,reinterpret_cast<float*>(base+capacity-bytes));
        };
        auto p=sized(n*4),v=sized(n*256),o=sized(256);
        std::fill_n(p.second,n,1.f);std::fill_n(v.second,n*64,0.5f);
        std::vector<float> expected(64);
        auto a=AccumulateAttentionValuesF32_AVX2(p.second,n,v.second,64,expected.data(),n,64);
        auto b=AccumulateAttentionValuesI15(p.second,n,v.second,64,o.second,n,64);
        Need(a==b&&std::memcmp(expected.data(),o.second,256)==0,"optimized guarded tail differs");
        for(auto base:{p.first,v.first,o.first})VirtualFree(base,0,MEM_RELEASE);
        ++cases;
    }
#endif
}
}
int main() {
    const unsigned csr=_mm_getcsr();
    try {RandomShapes();RangeContract();TileBoundaries();Failures();Aliases();GuardPages();_mm_setcsr(csr);
        std::cout<<"PASS tiled_attention_value_cases="<<cases<<"\n";return 0;
    } catch (const std::exception& e) {_mm_setcsr(csr);std::cerr<<e.what()<<'\n';return 1;}
}

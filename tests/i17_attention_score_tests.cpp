#include "AI/I17AttentionScore.h"
#include "AI/AIComputKrnlInternal.h"
#include <immintrin.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>
#ifdef _WIN32
#include <windows.h>
#endif

namespace {
size_t cases=0;
void Need(bool ok,const char* text) { if(!ok){std::cerr<<"I17: "<<text<<" case="<<cases<<'\n';std::abort();} }
uint32_t Rand(uint32_t& state) { state=state*1664525U+1013904223U;return state; }
float Finite(uint32_t& state) {
    uint32_t bits=Rand(state);if((bits&0x7f800000U)==0x7f800000U)bits^=0x00800000U;
    return std::bit_cast<float>(bits);
}
bool Baseline(const float* q,const float* k,size_t stride,float* scores,uint32_t rows,uint32_t dim,double den,double& maximum)
{
    maximum=-std::numeric_limits<double>::infinity();uint32_t t=0;
    auto consume=[&](const double* dots,uint32_t count){
        for(uint32_t i=0;i<count;++i){const double score=dots[i]/den;
            if(!std::isfinite(score))return false;scores[t+i]=static_cast<float>(score);maximum=std::max(maximum,score);}
        return true;};
    for(;rows-t>=4;t+=4){const float* r=k+static_cast<size_t>(t)*stride;double dots[4];
        micius::internal::AttentionScoreDotTimeQuadF32(q,r,r+stride,r+2*stride,r+3*stride,dim,dots);
        if(!consume(dots,4))return false;}
    for(;t<rows;++t){double dot=0;const float* r=k+static_cast<size_t>(t)*stride;
        for(uint32_t d=0;d<dim;++d)dot+=static_cast<double>(q[d])*r[d];
        if(!consume(&dot,1))return false;}
    return true;
}
void Compare(std::vector<float>& query,std::vector<float>& keys,size_t stride,uint32_t rows,unsigned control,bool finite,double den=8.0)
{
    const auto qb=query,kb=keys;const float* q=query.data()+1;const float* k=keys.data()+3;
    std::vector<float> expected(rows+2,17.25f),actual=expected;
    double em=123,am=123;
    _mm_setcsr(control);const bool ok=Baseline(q,k,stride,expected.data()+1,rows,64,den,em);const unsigned ef=_mm_getcsr();
    _mm_setcsr(control);const auto status=ComputeAttentionScoresI17(q,k,stride,actual.data()+1,rows,64,den,&am);const unsigned af=_mm_getcsr();
    _mm_setcsr(0x1f80);
    Need(status==(ok?I11ScoreStatus::OK:I11ScoreStatus::NUMERIC_ERROR),"status");
    Need(std::memcmp(actual.data(),expected.data(),actual.size()*sizeof(float))==0,"score/partial output/guard bits");
    Need(std::bit_cast<uint64_t>(am)==std::bit_cast<uint64_t>(em),"pre-rounded maximum bits");
    Need(af==ef,"MXCSR controls or sticky flags");
    Need(std::memcmp(query.data(),qb.data(),query.size()*sizeof(float))==0 && std::memcmp(keys.data(),kb.data(),keys.size()*sizeof(float))==0,"modified input");
    if(finite){
        // Separate row-by-row scalar FMA oracle; no production helper and no SIMD.
        _mm_setcsr(control);double maximum=-std::numeric_limits<double>::infinity();
        for(uint32_t t=0;t<rows;++t){double dot=0;for(uint32_t d=0;d<64;++d)
            dot=std::fma(static_cast<double>(q[d]),static_cast<double>(k[static_cast<size_t>(t)*stride+d]),dot);
            const double score=dot/den;const float s=static_cast<float>(score);maximum=std::max(maximum,score);
            Need(std::bit_cast<uint32_t>(s)==std::bit_cast<uint32_t>(actual[t+1]),"independent scalar score oracle");}
        Need(std::bit_cast<uint64_t>(maximum)==std::bit_cast<uint64_t>(am),"independent scalar maximum oracle");
        _mm_setcsr(0x1f80);
    }
    ++cases;
}
void FiniteCases()
{
    uint32_t seed=0x11641234;
    for(unsigned control:{0x1f80U,0x3f80U,0x5f80U,0x7f80U,0x1fc0U,0x9f80U,0x9fc0U})
    for(uint32_t rows:{8U,9U,11U,12U,15U,16U,17U,31U,32U,63U,127U,128U,129U,131U,135U,256U,257U,2048U,2049U})
    for(size_t stride:{size_t(64),size_t(67),size_t(320)})
    for(unsigned pattern=0;pattern<4;++pattern){
        std::vector<float> q(66),k((static_cast<size_t>(rows)-1)*stride+68,3.0f);
        for(auto& v:q)v=Finite(seed);
        for(uint32_t t=0;t<rows;++t)for(uint32_t d=0;d<64;++d){
            float v=Finite(seed);
            if(pattern==1){q[d+1]=(d%2?1.0f:-1.0f);v=(d%4<2?0x1p25f:0x1p-20f);}
            if(pattern==2){q[d+1]=d%2?0.0f:-0.0f;v=t%2?-0.0f:0.0f;}
            if(pattern==3){q[d+1]=std::bit_cast<float>(uint32_t(d+1));v=std::numeric_limits<float>::max();}
            k[3+static_cast<size_t>(t)*stride+d]=v;
        }
        Compare(q,k,stride,rows,control|0x20U,true);
    }
}
void NumericAndUnsupported()
{
    const uint32_t bads[]={0x7f800000U,0xff800000U,0x7fc12345U,0x7f812345U,0xffc12345U};
    for(uint32_t rows:{21U,259U})for(uint32_t bad:bads)
    for(uint32_t row:{0U,3U,4U,7U,8U,15U,16U,19U,20U,127U,128U,135U,255U,256U,258U})for(uint32_t d:{0U,31U,63U}){
        if(row>=rows)continue;
        std::vector<float> q(66,0.5f),k(rows*67+4,0.75f);k[3+static_cast<size_t>(row)*67+d]=std::bit_cast<float>(bad);
        // A later sNaN must not leave speculative flags if an earlier row failed.
        if(row<8)k[3+14*67]=std::bit_cast<float>(0x7f812345U);
        Compare(q,k,67,rows,0x1f80U,false);
    }
    for(uint32_t rows:{21U,259U})for(uint32_t bad:bads)for(uint32_t d:{0U,31U,63U}){
        std::vector<float> q(66,0.5f),k(rows*64+4,0.75f);q[d+1]=std::bit_cast<float>(bad);Compare(q,k,64,rows,0x1f80U,false);
    }
    // Invalid in the second chronological quartet must not leak a speculative
    // invalid flag after the first quartet has already failed on infinity.
    for(uint32_t row:{0U,3U,128U,131U}) {
        std::vector<float> q(66,0.5f),k(259*67+4,0.75f);
        k[3+static_cast<size_t>(row)*67]=std::numeric_limits<float>::infinity();
        k[3+static_cast<size_t>(row/8*8+6)*67]=std::bit_cast<float>(0x7f812345U);
        Compare(q,k,67,259,0x1fa0U,false);
    }
    std::vector<float> q(64,1.0f),k(128*64,0.5f),s(128,9.0f);double maximum=123;
    for(uint32_t dim:{0U,1U,32U,63U,65U,128U}){
        Need(ComputeAttentionScoresI17(q.data(),k.data(),64,s.data(),128,dim,8,&maximum)==I11ScoreStatus::UNSUPPORTED,"dimension fallback");++cases;
    }
    for(uint32_t n=0;n<8;++n){Need(ComputeAttentionScoresI17(q.data(),k.data(),64,s.data(),n,64,8,&maximum)==I11ScoreStatus::UNSUPPORTED,"short fallback");++cases;}
    for(unsigned csr:{0x1f00U,0x1d80U}){
        _mm_setcsr(csr);Need(ComputeAttentionScoresI17(q.data(),k.data(),64,s.data(),128,64,8,&maximum)==I11ScoreStatus::UNSUPPORTED,"unmasked fallback");
        Need(_mm_getcsr()==csr,"fallback flags");_mm_setcsr(0x1f80);++cases;}
    Need(ComputeAttentionScoresI17(q.data(),k.data(),SIZE_MAX,s.data(),128,64,8,&maximum)==I11ScoreStatus::UNSUPPORTED,"overflow fallback");++cases;
    Need(ComputeAttentionScoresI17(q.data(),k.data(),64,q.data(),128,64,8,&maximum)==I11ScoreStatus::UNSUPPORTED,"alias fallback");++cases;
    Need(ComputeAttentionScoresI17(nullptr,k.data(),64,s.data(),128,64,8,&maximum)==I11ScoreStatus::UNSUPPORTED,"null fallback");++cases;
    for(double den:{0.0,-1.0,0.5,std::numeric_limits<double>::infinity(),
                    std::numeric_limits<double>::quiet_NaN()}) {
        Need(ComputeAttentionScoresI17(q.data(),k.data(),64,s.data(),128,64,den,&maximum)==I11ScoreStatus::UNSUPPORTED,"denominator fallback");++cases;
    }
    Need(ComputeAttentionScoresI17(q.data(),k.data(),63,s.data(),128,64,8,&maximum)==I11ScoreStatus::UNSUPPORTED,"stride fallback");++cases;
    Need(ComputeAttentionScoresI17(q.data(),k.data(),64,s.data(),128,64,8,reinterpret_cast<double*>(s.data()+4))==I11ScoreStatus::UNSUPPORTED,"maximum alias fallback");++cases;
    Need(maximum==123 && std::all_of(s.begin(),s.end(),[](float x){return x==9.0f;}),"unsupported wrote output");
}
void GuardPages()
{
#ifdef _WIN32
    SYSTEM_INFO info{};GetSystemInfo(&info);const size_t page=info.dwPageSize;
    for(uint32_t rows:{128U,129U,132U,135U}) {
        const size_t kb=static_cast<size_t>(rows)*64*sizeof(float);
        const size_t committed=(kb+page-1)/page*page;
        auto* qm=static_cast<char*>(VirtualAlloc(nullptr,page*2,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE));
        auto* km=static_cast<char*>(VirtualAlloc(nullptr,committed+page,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE));
        Need(qm && km,"guard allocation");DWORD old=0;
        Need(VirtualProtect(qm+page,page,PAGE_NOACCESS,&old)!=0,"query guard");
        Need(VirtualProtect(km+committed,page,PAGE_NOACCESS,&old)!=0,"Key guard");
        auto* q=reinterpret_cast<float*>(qm+page-64*sizeof(float));
        auto* k=reinterpret_cast<float*>(km+committed-kb);
        for(uint32_t d=0;d<64;++d)q[d]=static_cast<float>(d%7)*0.125f;
        for(size_t d=0;d<static_cast<size_t>(rows)*64;++d)k[d]=static_cast<float>(d%11)*0.0625f;
        std::vector<float> expected(rows),actual(rows);double em=0,am=0;
        Need(Baseline(q,k,64,expected.data(),rows,64,8.0,em),"guard reference");
        Need(ComputeAttentionScoresI17(q,k,64,actual.data(),rows,64,8.0,&am)==I11ScoreStatus::OK,"guard status");
        Need(std::memcmp(expected.data(),actual.data(),rows*sizeof(float))==0 && em==am,"guard output");
        Need(VirtualFree(qm,0,MEM_RELEASE)!=0 && VirtualFree(km,0,MEM_RELEASE)!=0,"guard release");++cases;
    }
#endif
}
}
int main(){const unsigned csr=_mm_getcsr();FiniteCases();NumericAndUnsupported();GuardPages();
    uint32_t seed=0x175164U;
    for(double den:{1.0,std::nextafter(1.0,2.0),3.0,std::sqrt(64.5),1e200}) {
        std::vector<float> q(66),k(259*67+4);for(auto& x:q)x=Finite(seed);for(auto& x:k)x=Finite(seed);
        Compare(q,k,67,259,0x1f80,true,den);
    }
_mm_setcsr(csr);
    std::cout<<"ordered_attention_score_cases="<<cases<<"\nAll I17 score contracts passed\n";}

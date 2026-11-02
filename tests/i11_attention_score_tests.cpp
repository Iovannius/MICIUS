#include "AI/I11AttentionScore.h"
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

namespace {
size_t cases=0;
void Need(bool ok,const char* text) { if(!ok){std::cerr<<"I11: "<<text<<" case="<<cases<<'\n';std::abort();} }
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
void Compare(std::vector<float>& query,std::vector<float>& keys,size_t stride,uint32_t rows,unsigned control,bool finite)
{
    const auto qb=query,kb=keys;const float* q=query.data()+1;const float* k=keys.data()+3;
    std::vector<float> expected(rows+2,17.25f),actual=expected;
    double em=123,am=123;
    _mm_setcsr(control);const bool ok=Baseline(q,k,stride,expected.data()+1,rows,64,8.0,em);const unsigned ef=_mm_getcsr();
    _mm_setcsr(control);const auto status=ComputeAttentionScoresI11(q,k,stride,actual.data()+1,rows,64,8.0,&am);const unsigned af=_mm_getcsr();
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
            const double score=dot/8.0;const float s=static_cast<float>(score);maximum=std::max(maximum,score);
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
    for(uint32_t rows:{8U,9U,11U,12U,15U,16U,17U,31U,32U,63U,128U,256U,2048U})
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
    for(uint32_t bad:bads)for(uint32_t row:{0U,3U,4U,7U,8U,15U,16U,19U,20U})for(uint32_t d:{0U,31U,63U}){
        std::vector<float> q(66,0.5f),k(21*67+4,0.75f);k[3+static_cast<size_t>(row)*67+d]=std::bit_cast<float>(bad);
        // A later sNaN must not leave speculative flags if an earlier row failed.
        if(row<8)k[3+14*67]=std::bit_cast<float>(0x7f812345U);
        Compare(q,k,67,21,0x1f80U,false);
    }
    for(uint32_t bad:bads)for(uint32_t d:{0U,31U,63U}){
        std::vector<float> q(66,0.5f),k(21*64+4,0.75f);q[d+1]=std::bit_cast<float>(bad);Compare(q,k,64,21,0x1f80U,false);
    }
    std::vector<float> q(64,1.0f),k(32*64,0.5f),s(32,9.0f);double maximum=123;
    for(uint32_t dim:{0U,1U,32U,63U,65U,128U}){
        Need(ComputeAttentionScoresI11(q.data(),k.data(),64,s.data(),8,dim,8,&maximum)==I11ScoreStatus::UNSUPPORTED,"dimension fallback");++cases;
    }
    for(uint32_t n=0;n<8;++n){Need(ComputeAttentionScoresI11(q.data(),k.data(),64,s.data(),n,64,8,&maximum)==I11ScoreStatus::UNSUPPORTED,"short fallback");++cases;}
    for(unsigned csr:{0x1f00U,0x1d80U}){
        _mm_setcsr(csr);Need(ComputeAttentionScoresI11(q.data(),k.data(),64,s.data(),8,64,8,&maximum)==I11ScoreStatus::UNSUPPORTED,"unmasked fallback");
        Need(_mm_getcsr()==csr,"fallback flags");_mm_setcsr(0x1f80);++cases;}
    Need(ComputeAttentionScoresI11(q.data(),k.data(),SIZE_MAX,s.data(),8,64,8,&maximum)==I11ScoreStatus::UNSUPPORTED,"overflow fallback");++cases;
    Need(ComputeAttentionScoresI11(q.data(),k.data(),64,q.data(),8,64,8,&maximum)==I11ScoreStatus::UNSUPPORTED,"alias fallback");++cases;
    Need(ComputeAttentionScoresI11(nullptr,k.data(),64,s.data(),8,64,8,&maximum)==I11ScoreStatus::UNSUPPORTED,"null fallback");++cases;
    Need(maximum==123 && std::all_of(s.begin(),s.end(),[](float x){return x==9.0f;}),"unsupported wrote output");
}
}
int main(){const unsigned csr=_mm_getcsr();FiniteCases();NumericAndUnsupported();_mm_setcsr(csr);
    std::cout<<"ordered_attention_score_cases="<<cases<<"\nAll I11 score contracts passed\n";}

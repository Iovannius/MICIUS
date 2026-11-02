#include "AI/I19AttentionScore.h"
#include "AI/I17AttentionScore.h"
#include <immintrin.h>
#ifdef _WIN32
#include <windows.h>
#endif
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <iostream>
#include <limits>
#include <vector>
#include <cstdlib>

namespace {
size_t cases=0,hits=0,misses=0;
void Need(bool ok,const char* why) {if(!ok){std::cerr<<why<<" case="<<cases<<'\n';std::abort();}}
uint32_t Next(uint32_t& s){return s=s*1664525U+1013904223U;}
float Random(uint32_t& s){return float(int(Next(s)%65537)-32768)/4096.0f;}
float Finite(uint32_t& s){uint32_t b=Next(s);if((b&0x7f800000U)==0x7f800000U)b^=0x00800000U;return std::bit_cast<float>(b);}

void Compare(std::vector<float>& q,std::vector<float>& k,size_t stride,uint32_t rows,unsigned csr,
             bool scalar=false,int fail_consumer=-1)
{
    const auto qs=q,ks=k;const size_t pitch=rows+3;
    std::vector<float> old(3*pitch+2,13.25f),actual=old,scratch(3*pitch+2,27.5f);
    double oldmax[3]={123,123,123},actualmax[3]={123,123,123},mx[5]={11,22,33,44,55};
    I11ScoreStatus astatus[3]={I11ScoreStatus::UNSUPPORTED,I11ScoreStatus::UNSUPPORTED,I11ScoreStatus::UNSUPPORTED};
    I11ScoreStatus bstatus[3]={I11ScoreStatus::UNSUPPORTED,I11ScoreStatus::UNSUPPORTED,I11ScoreStatus::UNSUPPORTED};
    _mm_setcsr(csr);
    const bool ready=TryPrepareAttentionScoresI19(q.data(),k.data(),stride,scratch.data()+1,pitch,mx+1,rows,64,3,8);
    Need(_mm_getcsr()==csr,"preparation leaked MXCSR");
    Need(scratch.front()==27.5f && scratch.back()==27.5f && mx[0]==11 && mx[4]==55,"private scratch guard");
    for(unsigned h=0;h<3;++h)for(size_t j=rows;j<pitch;++j)
        Need(scratch[1+h*pitch+j]==27.5f,"private score padding");
    if(ready)++hits;else ++misses;
    for(unsigned h=0;h<3;++h) {
        if(ready){std::memcpy(actual.data()+1+h*pitch,scratch.data()+1+h*pitch,rows*sizeof(float));actualmax[h]=mx[h+1];bstatus[h]=I11ScoreStatus::OK;}
        else bstatus[h]=ComputeAttentionScoresI17(q.data()+h*64,k.data(),stride,actual.data()+1+h*pitch,rows,64,8,&actualmax[h]);
        if(bstatus[h]!=I11ScoreStatus::OK || int(h)==fail_consumer)break;
        // Model consumers may accumulate, but never clear incoming sticky flags.
        _mm_setcsr(_mm_getcsr()|0x20U);
    }
    const unsigned bf=_mm_getcsr();_mm_setcsr(csr);
    for(unsigned h=0;h<3;++h) {
        astatus[h]=ComputeAttentionScoresI17(q.data()+h*64,k.data(),stride,old.data()+1+h*pitch,rows,64,8,&oldmax[h]);
        if(astatus[h]!=I11ScoreStatus::OK || int(h)==fail_consumer)break;
        _mm_setcsr(_mm_getcsr()|0x20U);
    }
    const unsigned af=_mm_getcsr();_mm_setcsr(0x1f80);
    Need(af==bf,"head-order flags");Need(std::memcmp(astatus,bstatus,sizeof astatus)==0,"head-order status");
    Need(std::memcmp(old.data(),actual.data(),old.size()*sizeof(float))==0,"score/first-failure/guard output");
    Need(std::memcmp(oldmax,actualmax,sizeof oldmax)==0,"maximum bits");
    Need(std::memcmp(q.data(),qs.data(),q.size()*sizeof(float))==0 && std::memcmp(k.data(),ks.data(),k.size()*sizeof(float))==0,"input changed");
    if(scalar && fail_consumer<0){
        _mm_setcsr(csr);
        for(unsigned h=0;h<3;++h){
            double maximum=-std::numeric_limits<double>::infinity();
            for(uint32_t t=0;t<rows;++t){double dot=0;
                for(uint32_t d=0;d<64;++d)dot=std::fma(double(q[h*64+d]),double(k[size_t(t)*stride+d]),dot);
                const double score=dot/8;const float s=float(score);maximum=std::max(maximum,score);
                Need(std::bit_cast<uint32_t>(s)==std::bit_cast<uint32_t>(actual[1+h*pitch+t]),"independent scalar score");
            }
            Need(std::bit_cast<uint64_t>(maximum)==std::bit_cast<uint64_t>(actualmax[h]),"independent scalar maximum");
        }
        _mm_setcsr(0x1f80);
    }
    ++cases;
}

void Cases(){
    uint32_t seed=12345;
    for(unsigned csr:{0x1f80U,0x1fa0U,0x3fa0U,0x5fa0U,0x7fa0U,0x1fe0U,0x9fa0U,0x9fe0U,0x1fbfU})
    for(uint32_t rows:{8U,32U,127U,128U,129U,131U,132U,135U,255U,256U,257U,2048U,2049U})
    for(size_t stride:{size_t(64),size_t(67),size_t(320)})
    for(unsigned pattern=0;pattern<3;++pattern){
        std::vector<float> q(192),k(size_t(rows-1)*stride+64);
        for(auto& x:q)x=pattern==2?Finite(seed):Random(seed);
        for(auto& x:k)x=pattern==2?Finite(seed):Random(seed);
        if(pattern==1){for(unsigned h=0;h<3;++h)for(unsigned d=0;d<64;++d)q[h*64+d]=d%2?float(h+1):-float(h+1);
            for(uint32_t t=0;t<rows;++t)for(unsigned d=0;d<64;++d)k[size_t(t)*stride+d]=d%4<2?0x1p25f:0x1p-20f;}
        Compare(q,k,stride,rows,csr,pattern<2);
    }
    for(uint32_t rows:{8191U,8192U,8193U}){std::vector<float> q(192),k(size_t(rows)*64);for(auto&x:q)x=Random(seed);for(auto&x:k)x=Random(seed);Compare(q,k,64,rows,0x1fa0,true);}
    for(uint32_t rows:{128U,259U})for(uint32_t bad:{0x7f800000U,0xff800000U,0x7fc12345U,0x7f812345U}){
        for(uint32_t row:{0U,3U,4U,7U,8U,127U,128U,255U,258U})if(row<rows)
        for(uint32_t d:{0U,31U,63U}){
            std::vector<float> q(192,.5f),k(size_t(rows)*67,.75f);k[size_t(row)*67+d]=std::bit_cast<float>(bad);
            Compare(q,k,67,rows,0x1f80);}
        for(unsigned h=0;h<3;++h)for(unsigned d:{0U,31U,63U}){
            std::vector<float> q(192,.5f),k(size_t(rows)*64,.75f);q[h*64+d]=std::bit_cast<float>(bad);
            Compare(q,k,64,rows,0x1f80);Compare(q,k,64,rows,0x1fa0,false,0);
        }
    }
    for(unsigned csr:{0x1f80U,0x1fa0U,0x9fe0U})for(int h:{0,1,2}){
        std::vector<float> q(192),k(259*67);for(auto&x:q)x=Random(seed);for(auto&x:k)x=Random(seed);Compare(q,k,67,259,csr,false,h);
    }
    // DAZ/subnormal treatment and signed zero are also head-specific.
    for(unsigned csr:{0x1f80U,0x1fa0U,0x1fc0U,0x9f80U,0x9fc0U,0x1fbfU}){
        std::vector<float> q(192),k(135*64);
        for(unsigned d=0;d<192;++d)q[d]=d%3?std::bit_cast<float>(uint32_t(d+1)):-0.f;
        for(auto&x:k)x=std::numeric_limits<float>::max();Compare(q,k,64,135,csr,true);
    }
}
void Guards(){
    std::vector<float> q(192,.5f),k(128*64,.5f),s(3*128,7);double m[3]={1,2,3};
    auto unchanged=[&](){Need(std::all_of(s.begin(),s.end(),[](float x){return x==7;}) && m[0]==1 && m[1]==2 && m[2]==3,"unsupported wrote scratch");};
    for(unsigned csr:{0x1f00U,0x1d80U}){_mm_setcsr(csr);Need(!TryPrepareAttentionScoresI19(q.data(),k.data(),64,s.data(),128,m,128,64,3,8),"unmasked accepted");Need(_mm_getcsr()==csr,"unmasked changed flags");_mm_setcsr(0x1f80);++cases;}
    for(unsigned dim:{0U,32U,63U,65U}){Need(!TryPrepareAttentionScoresI19(q.data(),k.data(),64,s.data(),128,m,128,dim,3,8),"wrong dimension");++cases;}
    for(unsigned heads:{0U,1U,2U,4U}){Need(!TryPrepareAttentionScoresI19(q.data(),k.data(),64,s.data(),128,m,128,64,heads,8),"wrong grouping");++cases;}
    Need(!TryPrepareAttentionScoresI19(q.data(),k.data(),SIZE_MAX,s.data(),128,m,128,64,3,8),"stride overflow");
    Need(!TryPrepareAttentionScoresI19(q.data(),k.data(),64,s.data(),SIZE_MAX,m,128,64,3,8),"score overflow");
    Need(!TryPrepareAttentionScoresI19(q.data(),k.data(),64,q.data(),128,m,128,64,3,8),"query alias");
    Need(!TryPrepareAttentionScoresI19(q.data(),k.data(),64,k.data(),128,m,128,64,3,8),"Key alias");
    Need(!TryPrepareAttentionScoresI19(q.data(),k.data(),64,s.data(),128,reinterpret_cast<double*>(s.data()),128,64,3,8),"maximum alias");
    Need(!TryPrepareAttentionScoresI19(nullptr,k.data(),64,s.data(),128,m,128,64,3,8),"null");
    for(double den:{0.,-1.,1.,std::numeric_limits<double>::infinity(),std::bit_cast<double>(uint64_t{0x7ff0000000000001})}){
        _mm_setcsr(0x1f80);Need(!TryPrepareAttentionScoresI19(q.data(),k.data(),64,s.data(),128,m,128,64,3,den),"denominator guard");Need(_mm_getcsr()==0x1f80,"denominator guard flags");++cases;}
    unchanged();cases+=6;
#ifdef _WIN32
    SYSTEM_INFO info{};GetSystemInfo(&info);const size_t page=info.dwPageSize;
    for(uint32_t rows:{128U,129U,131U,135U}){
        const size_t kb=size_t(rows)*64*sizeof(float),committed=(kb+page-1)/page*page;
        auto* qm=static_cast<char*>(VirtualAlloc(nullptr,page*2,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE));
        auto* km=static_cast<char*>(VirtualAlloc(nullptr,committed+page,MEM_RESERVE|MEM_COMMIT,PAGE_READWRITE));
        DWORD old;Need(qm && km,"guard allocation");Need(VirtualProtect(qm+page,page,PAGE_NOACCESS,&old) && VirtualProtect(km+committed,page,PAGE_NOACCESS,&old),"guard protect");
        auto* qp=reinterpret_cast<float*>(qm+page-192*sizeof(float));auto* kp=reinterpret_cast<float*>(km+committed-kb);
        std::fill_n(qp,192,.5f);std::fill_n(kp,size_t(rows)*64,.25f);std::vector<float> out(3*rows);
        _mm_setcsr(0x1f80);Need(TryPrepareAttentionScoresI19(qp,kp,64,out.data(),rows,m,rows,64,3,8),"guard preparation");
        Need(std::all_of(out.begin(),out.end(),[](float x){return x==1.f;}),"guard output");
        Need(VirtualFree(qm,0,MEM_RELEASE) && VirtualFree(km,0,MEM_RELEASE),"guard release");++cases;
    }
#endif
}
}
int main(){const unsigned csr=_mm_getcsr();Cases();Guards();Need(hits>=100,"fast path not exercised");_mm_setcsr(csr);
 std::cout<<"group_score_cases="<<cases<<" fast="<<hits<<" fallback="<<misses<<"\nAll I19 group score contracts passed\n";}

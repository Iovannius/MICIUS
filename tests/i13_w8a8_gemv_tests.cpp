#include "AI/I13W8A8GEMV.h"
#include <algorithm>
#include <array>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#ifdef NDEBUG
#error "I13 checks require active assertions"
#endif
using namespace micius::internal;
using Status=Q8_32_GEMVStatus;
static size_t dot_cases=0,gemv_cases=0,guard_cases=0;
static void DotOracle(const std::array<int8_t,32>& a,const std::array<int8_t,128>& w) {
    int32_t out[4];
    _mm_storeu_si128(reinterpret_cast<__m128i*>(out),i13::Dot4(
        _mm256_loadu_si256(reinterpret_cast<const __m256i*>(a.data())),w.data(),w.data()+32,w.data()+64,w.data()+96));
    for (size_t r=0;r<4;++r) {
        int64_t expected=0;
        for (size_t k=0;k<32;++k) expected+=static_cast<int64_t>(a[k])*w[r*32+k];
        assert(expected==out[r]);
    }
    ++dot_cases;
}
static void TestDots() {
    std::array<int8_t,32> a;std::array<int8_t,128> w;
    // Every admissible activation/weight byte combination, including maximal
    // adjacent pairs. Other rows exercise sign changes and lane placement.
    for (int x=-127;x<=127;++x) for (int y=-128;y<=127;++y) {
        a.fill(static_cast<int8_t>(x));
        for (size_t k=0;k<32;++k) {
            w[k]=static_cast<int8_t>(y);
            w[32+k]=static_cast<int8_t>(k%2 ? -128 : 127);
            w[64+k]=static_cast<int8_t>(k%2 ? y : 0);
            w[96+k]=static_cast<int8_t>(127-static_cast<int>(k)*8);
        }
        DotOracle(a,w);
    }
    for (size_t lane=0;lane<32;++lane) {
        a.fill(0);a[lane]=-127;
        for(size_t r=0;r<4;++r) for(size_t k=0;k<32;++k)
            w[r*32+k]=static_cast<int8_t>(-128+static_cast<int>(r*32+k));
        DotOracle(a,w);
    }
}
static bool Same(float a,float b) {return std::bit_cast<uint32_t>(a)==std::bit_cast<uint32_t>(b);}
static void Check(uint32_t n,uint32_t rows,int mode,std::mt19937& rng) {
    const uint32_t blocks=(n+31)/32;
    std::vector<float> input(n),out(rows+2,-99),oracle(rows+2,-99),old(rows+2,-99),scalar(rows+2,-99);
    std::vector<int8_t> code(n),weights(static_cast<size_t>(rows)*n);
    std::vector<double> dx(blocks);
    std::vector<QuantDescriptor> dw(static_cast<size_t>(rows)*blocks);
    for(auto& v:input) {
        const int k=static_cast<int>(rng()%255)-127;
        v=mode==0?0.0f:mode==1?static_cast<float>(k):mode==2?
          std::numeric_limits<float>::denorm_min()*static_cast<float>(k):
          mode==3?static_cast<float>(k)*(std::numeric_limits<float>::max()/128.0f):static_cast<float>(k)/19.0f;
    }
    for(auto& v:weights)v=static_cast<int8_t>(static_cast<int>(rng()%256)-128);
    for(auto& v:dw)v=QuantDescriptor(std::ldexp(static_cast<float>(static_cast<int>(rng()%31)-15),mode==3?-40:-12),0);
    PreparedW8A8Input x;assert(PrepareW8A8Input(input,code,dx,x)==Status::OK);
    const auto input_before=input;const auto w_before=weights;const auto code_before=code;const auto dx_before=dx;
    for(uint32_t r=0;r<rows;++r) {
        double sum=0.0;
        for(uint32_t b=0;b<blocks;++b) {
            int64_t dot=0;
            for(uint32_t k=b*32;k<std::min(n,b*32+32);++k)
                dot+=static_cast<int64_t>(weights[static_cast<size_t>(r)*n+k])*code[k];
            const double weighted=static_cast<double>(dot)*static_cast<double>(dw[static_cast<size_t>(r)*blocks+b].scale);
            sum+=weighted*dx[b];
        }
        oracle[r]=static_cast<float>(sum);
    }
    assert(GEMV_W8A8Rows(weights.data(),dw.data(),x,out.data(),rows,0,rows,true,false)==Status::OK);
    assert(GEMV_W8A8Rows(weights.data(),dw.data(),x,old.data(),rows,0,rows,false,false)==Status::OK);
    assert(GEMV_W8A8Rows(weights.data(),dw.data(),x,scalar.data(),rows,0,rows,true,true)==Status::OK);
    for(uint32_t r=0;r<rows+2;++r)assert(Same(out[r],oracle[r])&&Same(out[r],old[r])&&Same(out[r],scalar[r]));
    // Odd start and incomplete row group, with write sentinels on both sides.
    if(rows>2) {
        std::fill(out.begin(),out.end(),-99);
        assert(GEMV_W8A8Rows(weights.data(),dw.data(),x,out.data(),rows,1,rows-1,true,false)==Status::OK);
        assert(out[0]==-99 && out[rows-1]==-99 && out[rows]==-99);
        for(uint32_t r=1;r<rows-1;++r)assert(Same(out[r],oracle[r]));
    }
    assert(input==input_before && weights==w_before && code==code_before && dx==dx_before);
    ++gemv_cases;
}
static void Failures() {
    std::vector<float> in(32,127);std::vector<int8_t> q(32),w(128,-128);
    std::vector<double> d(1);std::vector<QuantDescriptor> s(4,QuantDescriptor(1,0));
    PreparedW8A8Input x;assert(PrepareW8A8Input(in,q,d,x)==Status::OK);float out[4]={9,9,9,9};
    assert(GEMV_W8A8Rows(w.data(),s.data(),x,out,4,0,4,true,false)==Status::OK);
    for(float y:out)assert(y==-520192.0f);
    s[3].scale=std::numeric_limits<float>::max();
    assert(GEMV_W8A8Rows(w.data(),s.data(),x,out,4,0,4,true,false)==Status::NUMERIC_ERROR);
    s[3]=QuantDescriptor(1,1);
    assert(GEMV_W8A8Rows(w.data(),s.data(),x,out,4,0,4,false,false)==Status::INVALID_MODEL);
    s[3]=QuantDescriptor(std::numeric_limits<float>::quiet_NaN(),0);
    assert(GEMV_W8A8Rows(w.data(),s.data(),x,out,4,0,4,false,false)==Status::INVALID_MODEL);
    for(uint32_t control:{0x3f80U,0x5f80U,0x7f80U,0x9f80U,0x1fc0U}) {
        _mm_setcsr(control);
        assert(GEMV_W8A8Rows(w.data(),s.data(),x,out,4,0,4,true,false)==Status::INVALID_ARGUMENT);
    }
    _mm_setcsr(0x1f80U);
    assert(GEMV_W8A8Rows(nullptr,s.data(),x,out,4,0,4,true,false)==Status::INVALID_ARGUMENT);
    assert(GEMV_W8A8Rows(w.data(),s.data(),x,out,4,3,2,true,false)==Status::INVALID_ARGUMENT);
    assert(GEMV_W8A8Rows(w.data(),s.data(),x,out,4,0,5,true,false)==Status::INVALID_ARGUMENT);
    assert(GEMV_W8A8Rows(w.data(),s.data(),x,out,4,4,4,true,false)==Status::OK);
}
#ifdef _WIN32
static void GuardPages() {
    SYSTEM_INFO info;GetSystemInfo(&info);const size_t page=info.dwPageSize;
    auto* memory=static_cast<int8_t*>(VirtualAlloc(nullptr,page*4,MEM_COMMIT|MEM_RESERVE,PAGE_READWRITE));
    assert(memory);DWORD previous=0;
    assert(VirtualProtect(memory+page,page,PAGE_NOACCESS,&previous));
    assert(VirtualProtect(memory+page*3,page,PAGE_NOACCESS,&previous));
    for(uint32_t n=1;n<=128;++n) {
        constexpr uint32_t rows=5;
        auto* w=memory+page-rows*n;auto* codes=memory+page*3-n;
        std::fill_n(w,rows*n,int8_t{-128});std::vector<float> input(n,127);
        std::vector<double>d((n+31)/32);std::vector<QuantDescriptor>s(rows*((n+31)/32),QuantDescriptor(1,0));
        PreparedW8A8Input x;assert(PrepareW8A8Input(input,std::span<int8_t>(codes,n),d,x)==Status::OK);
        float out[rows];assert(GEMV_W8A8Rows(w,s.data(),x,out,rows,0,rows,true,false)==Status::OK);
        for(float y:out)assert(y==static_cast<float>(-128*127*static_cast<int>(n)));
        ++guard_cases;
    }
    assert(VirtualFree(memory,0,MEM_RELEASE));
}
#endif
int main() {
    const auto saved=_mm_getcsr();_mm_setcsr(0x1f80U);TestDots();std::mt19937 rng(0x1313U);
    for(uint32_t n:{1U,15U,16U,31U,32U,33U,63U,64U,65U,96U,127U,128U,129U,256U,576U,960U,1536U,2560U})
        for(uint32_t rows:{1U,3U,4U,5U,7U,8U,9U,17U})
            for(int mode=0;mode<5;++mode)Check(n,rows,mode,rng);
    Check(960,2560,4,rng);Check(2560,960,4,rng);
    Failures();
#ifdef _WIN32
    GuardPages();
#endif
    _mm_setcsr(saved);
    std::cout<<"integer_dot_cases="<<dot_cases<<"\ngemv_cases="<<gemv_cases<<"\nguard_page_cases="<<guard_cases<<"\nstatus=PASS\n";
}

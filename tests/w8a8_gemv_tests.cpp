#include "AI/PreparedW8A8Input.h"
#include <algorithm>
#include <bit>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#include <immintrin.h>
#ifndef _WIN32
#include <sys/mman.h>
#include <unistd.h>
#endif
#ifdef NDEBUG
#error "I1 tests require active assertions"
#endif
using namespace micius::internal;
using Status=Q8_32_GEMVStatus;
static int8_t QuantizeOracle(double x,double d) {
    if (!d) return 0;
    const double u=std::abs(x/d),f=std::floor(u),fraction=u-f;
    int q=static_cast<int>(f);
    if (fraction>0.5 || (fraction==0.5 && (q&1))) ++q;
    q=std::min(q,127);
    return static_cast<int8_t>(std::signbit(x)?-q:q);
}
static uint32_t cases=0;
static void Check(uint32_t n,uint32_t rows,std::mt19937& random,int mode) {
    const uint32_t blocks=(n+31U)/32U;
    std::vector<float> input(n);
    std::vector<int8_t> quant(n),expected(n),weights(static_cast<size_t>(rows)*n);
    std::vector<double> scales(blocks),expected_scales(blocks);
    std::vector<QuantDescriptor> descriptors(static_cast<size_t>(rows)*blocks);
    for (auto& x:input) {
        const int v=static_cast<int>(random()%255U)-127;
        x=mode==0 ? 0.0f : mode==1 ? static_cast<float>(v)*0.5f :
          mode==2 ? std::numeric_limits<float>::denorm_min()*static_cast<float>(v) :
          mode==3 ? static_cast<float>(v)*(std::numeric_limits<float>::max()/128.0f) : static_cast<float>(v)/19.0f;
    }
    for (auto& w:weights) w=static_cast<int8_t>(static_cast<int>(random()%256U)-128);
    for (auto& s:descriptors) s=QuantDescriptor(std::ldexp(static_cast<float>(static_cast<int>(random()%31U)-15),mode==3?-32:-12),0);
    PreparedW8A8Input view;
    assert(PrepareW8A8Input(input,quant,scales,view)==Status::OK);
    for (uint32_t b=0;b<blocks;++b) {
        const uint32_t end=std::min(n,b*32U+32U);
        double largest=0;
        for (uint32_t k=b*32U;k<end;++k) largest=std::max(largest,std::abs(static_cast<double>(input[k])));
        expected_scales[b]=largest/127.0;
        assert(std::bit_cast<uint64_t>(scales[b])==std::bit_cast<uint64_t>(expected_scales[b]));
        for (uint32_t k=b*32U;k<end;++k) {
            expected[k]=QuantizeOracle(input[k],expected_scales[b]);
            assert(quant[k]==expected[k]);
            assert(std::abs(static_cast<double>(input[k])-quant[k]*scales[b]) <= scales[b]*(0.5+1e-12));
        }
    }
    std::vector<float> oracle(rows),a(rows,-99),b(rows,-99),checked(rows,-99),slice(rows,-99);
    for (uint32_t r=0;r<rows;++r) {
        double sum=0;
        for (uint32_t block=0;block<blocks;++block) {
            int64_t dot=0;
            for (uint32_t k=block*32U;k<std::min(n,block*32U+32U);++k)
                dot+=static_cast<int64_t>(weights[static_cast<size_t>(r)*n+k])*expected[k];
            assert(std::abs(dot)<=520192);
            double term=static_cast<double>(dot)*descriptors[static_cast<size_t>(r)*blocks+block].scale;
            sum+=term*expected_scales[block];
        }
        oracle[r]=static_cast<float>(sum);
    }
    assert(GEMV_W8A8Rows(weights.data(),descriptors.data(),view,a.data(),rows,0,rows,false,false)==Status::OK);
    assert(GEMV_W8A8Rows(weights.data(),descriptors.data(),view,b.data(),rows,0,rows,false,true)==Status::OK);
    assert(GEMV_W8A8Rows(weights.data(),descriptors.data(),view,checked.data(),rows,0,rows,true,false)==Status::OK);
    for (uint32_t r=0;r<rows;++r) {
        assert(std::bit_cast<uint32_t>(a[r])==std::bit_cast<uint32_t>(oracle[r]));
        assert(std::bit_cast<uint32_t>(a[r])==std::bit_cast<uint32_t>(b[r]));
        assert(std::bit_cast<uint32_t>(a[r])==std::bit_cast<uint32_t>(checked[r]));
        assert(GEMV_W8A8Rows(weights.data(),descriptors.data(),view,slice.data(),rows,r,r+1,false,false)==Status::OK);
    }
    assert(slice==a);
    assert(GEMV_W8A8Rows(weights.data(),descriptors.data(),view,a.data(),rows,rows,rows,false,false)==Status::OK);
    assert(GEMV_W8A8Rows(weights.data(),descriptors.data(),view,a.data(),rows,0,rows+1,false,false)==Status::INVALID_ARGUMENT);
    descriptors[0].zero_point=1;
    assert(GEMV_W8A8Rows(weights.data(),descriptors.data(),view,a.data(),rows,0,rows,false,false)==Status::INVALID_MODEL);
    descriptors[0]=QuantDescriptor(std::numeric_limits<float>::quiet_NaN(),0);
    assert(GEMV_W8A8Rows(weights.data(),descriptors.data(),view,a.data(),rows,0,rows,false,false)==Status::INVALID_MODEL);
    input[0]=std::numeric_limits<float>::infinity();
    assert(PrepareW8A8Input(input,quant,scales,view)==Status::NUMERIC_ERROR);
    assert(view.size()==0 && view.values()==nullptr);
    ++cases;
}
int main() {
    const auto saved=_mm_getcsr();_mm_setcsr(0x1f80U);
    std::mt19937 random(0x10c7U);
    for (uint32_t n:{1U,2U,15U,16U,17U,31U,32U,33U,63U,64U,65U,127U,128U,129U,256U,960U,2560U})
        for (uint32_t rows:{1U,2U,3U,5U,16U}) for (int mode=0;mode<5;++mode) Check(n,rows,random,mode);
    std::vector<float> x(32,127.0f);x[0]=0.5f;x[1]=1.5f;x[2]=2.5f;x[3]=-0.5f;x[4]=-1.5f;
    std::vector<int8_t> q(32),w(32,-128);std::vector<double> d(1);PreparedW8A8Input view;
    assert(PrepareW8A8Input(x,q,d,view)==Status::OK);
    assert(q[0]==0 && q[1]==2 && q[2]==2 && q[3]==0 && q[4]==-2);
    // Extremal signed products: no byte-pair saturation is permitted.
    std::fill(x.begin(),x.end(),127.0f);
    assert(PrepareW8A8Input(x,q,d,view)==Status::OK);
    QuantDescriptor s(1.0f,0);float y=0;
    assert(GEMV_W8A8Rows(w.data(),&s,view,&y,1,0,1,false,false)==Status::OK && y==-520192.0f);
    s.scale=std::numeric_limits<float>::max();
    assert(GEMV_W8A8Rows(w.data(),&s,view,&y,1,0,1,false,false)==Status::NUMERIC_ERROR);
    for (uint32_t ctrl:{0x3f80U,0x5f80U,0x7f80U,0x9f80U,0x1fc0U}) {
        _mm_setcsr(ctrl);
        assert(PrepareW8A8Input(x,q,d,view)==Status::INVALID_ARGUMENT);
    }
    _mm_setcsr(0x1f80U);
    assert(PrepareW8A8Input(x,std::span<int8_t>(q.data(),31),d,view)==Status::INVALID_ARGUMENT);
#ifndef _WIN32
    // Every remainder ends at an unreadable page, catching unmasked vector tails.
    const size_t page=static_cast<size_t>(sysconf(_SC_PAGESIZE));
    auto* mapping=static_cast<unsigned char*>(mmap(nullptr,page*2,PROT_READ|PROT_WRITE,MAP_PRIVATE|MAP_ANONYMOUS,-1,0));
    assert(mapping!=MAP_FAILED && mprotect(mapping+page,page,PROT_NONE)==0);
    for (uint32_t n=1;n<=65;++n) {
        std::vector<float> input(n,1.0f);std::vector<int8_t> codes(n);std::vector<double> scale((n+31U)/32U);
        std::vector<QuantDescriptor> sd((n+31U)/32U,QuantDescriptor(1,0));
        auto* weight=reinterpret_cast<int8_t*>(mapping+page-n);std::fill_n(weight,n,int8_t{-128});
        assert(PrepareW8A8Input(input,codes,scale,view)==Status::OK);
        assert(GEMV_W8A8Rows(weight,sd.data(),view,&y,1,0,1,false,false)==Status::OK);
    }
    assert(munmap(mapping,page*2)==0);
#endif
    _mm_setcsr(saved);
    std::cout<<"i1_cases="<<cases<<"\nstatus=PASS\n";
}

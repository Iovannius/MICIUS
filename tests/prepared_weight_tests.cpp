#include "AI/PreparedGEMVWeights.h"
#include "AI/PreparedGEMVInput.h"
#include <immintrin.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <new>
#include <vector>

// Deterministic optional-allocation failure after the first matrix succeeds.
// This override belongs only to this single-threaded test executable.
static int fail_array_after = -1;
void* operator new[](size_t n) {
    if (fail_array_after == 0) throw std::bad_alloc();
    if (fail_array_after > 0) --fail_array_after;
    if (void* p = std::malloc(n ? n : 1)) return p;
    throw std::bad_alloc();
}
void operator delete[](void* p) noexcept { std::free(p); }
void operator delete[](void* p, size_t) noexcept { std::free(p); }

namespace r8_frozen {
Q8_32_GEMVStatus RowPair(const int8_t*, const QuantDescriptor*, const float*,
                       float*, uint32_t, uint32_t) noexcept;
}
using namespace micius::internal;
using Status = Q8_32_GEMVStatus;
static void Check(bool ok) { if (!ok) { std::cerr << "PW2 check failed\n"; std::abort(); } }
static void Equal(const std::vector<float>& a, const std::vector<float>& b) {
    Check(a.size() == b.size());
    for (size_t i=0; i<a.size(); ++i)
        Check(std::bit_cast<uint32_t>(a[i]) == std::bit_cast<uint32_t>(b[i]));
}
static uint32_t Next(uint32_t& x) { x=x*1664525U+1013904223U; return x; }
static void Shape(uint32_t rows, uint32_t cols, uint32_t trial) {
    uint32_t rng=0x94167acdU+trial;
    const size_t blocks=(cols+31U)/32U, n=size_t{rows}*cols;
    std::vector<int8_t> raw(n+1);
    std::vector<float> x(cols), expected(rows), checked(rows), pre(rows), part(rows);
    std::vector<double> scratch(cols);
    std::vector<QuantDescriptor> scales(rows*blocks);
    for (size_t i=0; i<n; ++i) raw[i+1]=static_cast<int8_t>(int(Next(rng)&255U)-128);
    for (uint32_t i=0; i<cols; ++i) {
        x[i]=std::ldexp(static_cast<float>(int(Next(rng)%2001)-1000)/257.0f,
            trial ? int(i%41)-20 : 0);
        if (i%19==0) x[i]=-0.0f;
    }
    for (size_t i=0; i<scales.size(); ++i)
        scales[i]=QuantDescriptor(static_cast<float>(int(i%17)-8)/521.0f,0);
    WeightSource source{{raw.data()+1,n},rows,cols}; uint64_t bytes=0;
    std::vector<PreparedF32GEMVWeights> w;
    Check(PrepareF32WeightGroup({&source,1},uint64_t(n)*4,w,bytes)==WeightPrepareResult::READY);
    Check(bytes==n*4 && w.size()==1);
    for (size_t i=0; i<n; ++i) Check(w[0].data()[i]==double(raw[i+1]));
    PreparedF64GEMVInput input;
    Check(PrepareF64GEMVInput(x,scratch,input)==Status::OK);
    Check(r8_frozen::RowPair(raw.data()+1,scales.data(),x.data(),expected.data(),cols,rows)==Status::OK);
    Check(GEMV_F32WeightsPreparedRows(w[0],scales.data(),input,checked.data(),0,rows)==Status::OK);
    Check(GEMV_F32WeightsPreparedRowsPrevalidatedScales(w[0],scales.data(),input,pre.data(),0,rows)==Status::OK);
    for (uint32_t end=rows; end;) {
        const uint32_t begin=(end-1U)&~1U;
        Check(GEMV_F32WeightsPreparedRows(w[0],scales.data(),input,part.data(),begin,end)==Status::OK);
        end=begin;
    }
    Equal(expected,checked); Equal(expected,pre); Equal(expected,part);
    // Both operands are snapshots. Moving the owner preserves data; no raw
    // pointer hit or per-call expansion is used after the sources change.
    auto moved=std::move(w[0]);
    std::fill(raw.begin(),raw.end(),127); std::fill(x.begin(),x.end(),99.0f);
    Check(GEMV_F32WeightsPreparedRows(moved,scales.data(),input,pre.data(),0,rows)==Status::OK);
    Equal(expected,pre);
    Check(GEMV_F32WeightsPreparedRows(w[0],scales.data(),input,pre.data(),0,rows)==Status::INVALID_ARGUMENT);
}
static void FailureAndTail() {
    std::vector<int8_t> raw(3*33,0); std::vector<float> x(33,0), y(3,-7);
    for (size_t r=0; r<3; ++r) { raw[r*33]=127; raw[r*33+32]=-127; }
    const float value=std::bit_cast<float>(0x3fc00001U);
    const float scale=std::bit_cast<float>(0x3fc00003U);
    std::vector<double> scratch(33); std::vector<QuantDescriptor> s(6,QuantDescriptor(scale,0));
    WeightSource source{raw,3,33}; std::vector<PreparedF32GEMVWeights> w; uint64_t bytes=9;
    Check(PrepareF32WeightGroup({&source,1},0,w,bytes)==WeightPrepareResult::BUDGET);
    Check(w.empty() && bytes==0);
    const WeightSource pair[]={source,source};
    fail_array_after=1;
    Check(PrepareF32WeightGroup(pair,792,w,bytes)==WeightPrepareResult::ALLOCATION);
    fail_array_after=-1;
    Check(w.empty() && bytes==0);
    Check(PrepareF32WeightGroup({&source,1},396,w,bytes)==WeightPrepareResult::READY);
    WeightSource bad{raw,3,32};
    Check(PrepareF32WeightGroup({&bad,1},396,w,bytes)==WeightPrepareResult::INVALID);
    Check(w.empty() && bytes==0);
    Check(PrepareF32WeightGroup({&source,1},396,w,bytes)==WeightPrepareResult::READY);
    // cols=33 contract fixture: separately rounded cancellation, no FMA.
    x[0]=value; x[32]=value;
    PreparedF64GEMVInput in;
    Check(PrepareF64GEMVInput(x,scratch,in)==Status::OK);
    std::vector<float> expected(3);
    Check(r8_frozen::RowPair(raw.data(),s.data(),x.data(),expected.data(),33,3)==Status::OK);
    Check(GEMV_F32WeightsPreparedRows(w[0],s.data(),in,y.data(),0,3)==Status::OK);
    Equal(expected,y);
    for (float v : y) Check(std::bit_cast<uint32_t>(v)==0U);
    volatile double rounded=(double(value)*127.0)*scale;
    Check(std::bit_cast<uint32_t>(static_cast<float>(
        std::fma(-double(value)*127.0,double(scale),rounded)))==0xa8800000U);
    Check(GEMV_F32WeightsPreparedRows(w[0],s.data(),in,y.data(),1,3)==Status::INVALID_ARGUMENT);
    Check(GEMV_F32WeightsPreparedRows(w[0],s.data(),in,y.data(),0,1)==Status::INVALID_ARGUMENT);
    Check(GEMV_F32WeightsPreparedRows(w[0],nullptr,in,nullptr,2,2)==Status::OK);
    Check(GEMV_F32WeightsPreparedRows(w[0],s.data(),in,
        const_cast<float*>(w[0].data())+98,0,3)==Status::INVALID_ARGUMENT);
    Check(GEMV_F32WeightsPreparedRows(w[0],s.data(),in,reinterpret_cast<float*>(scratch.data()),0,3)==Status::INVALID_ARGUMENT);
    const uint32_t csr=_mm_getcsr(); _mm_setcsr(csr^0x2000U);
    Check(GEMV_F32WeightsPreparedRows(w[0],s.data(),in,y.data(),0,3)==Status::INVALID_ARGUMENT);
    _mm_setcsr(csr);
    s[0].zero_point=1;
    Check(GEMV_F32WeightsPreparedRows(w[0],s.data(),in,y.data(),0,3)==Status::INVALID_MODEL);
    s[0]=QuantDescriptor(std::numeric_limits<float>::infinity(),0);
    Check(GEMV_F32WeightsPreparedRows(w[0],s.data(),in,y.data(),0,3)==Status::INVALID_MODEL);
    s[0]=QuantDescriptor(std::numeric_limits<float>::max(),0);
    x[0]=std::numeric_limits<float>::max();
    Check(PrepareF64GEMVInput(x,scratch,in)==Status::OK);
    Check(GEMV_F32WeightsPreparedRowsPrevalidatedScales(w[0],s.data(),in,y.data(),0,3)==Status::NUMERIC_ERROR);
    x[0]=std::numeric_limits<float>::quiet_NaN();
    Check(PrepareF64GEMVInput(x,scratch,in)==Status::NUMERIC_ERROR);
    Check(GEMV_F32WeightsPreparedRows(w[0],s.data(),in,y.data(),0,3)==Status::INVALID_ARGUMENT);
}
static void ExactRawDomainAndFPControls() {
    const uint32_t saved=_mm_getcsr();
    std::vector<int8_t> raw(256);
    for (int i=0;i<256;++i) raw[size_t(i)]=static_cast<int8_t>(i-128);
    WeightSource source{raw,2,128};
    std::vector<PreparedF32GEMVWeights> weights; uint64_t bytes=0;
    _mm_setcsr(0x9fc0U); // Preparation can use a different FP control state.
    Check(PrepareF32WeightGroup({&source,1},1023,weights,bytes)==WeightPrepareResult::BUDGET);
    Check(weights.empty() && bytes==0);
    Check(PrepareF32WeightGroup({&source,1},1024,weights,bytes)==WeightPrepareResult::READY);
    Check(bytes==1024);
    for (int i=0;i<256;++i) Check(double(weights[0].data()[i])==double(i-128));
    std::vector<float> x(128), expected(2), checked(2), pre(2);
    std::vector<double> scratch(128);
    std::vector<QuantDescriptor> scales(8);
    _mm_setcsr(0x1f80U);
    for (size_t i=0;i<x.size();++i)
        x[i]=std::ldexp(float(int(i%17)-8)/19.0f,int(i%61)-30);
    x[0]=std::bit_cast<float>(uint32_t{1}); x[1]=-0.0f;
    for (size_t i=0;i<scales.size();++i) scales[i]=QuantDescriptor(float(int(i)-4)/521.0f,0);
    scales[0]=QuantDescriptor(std::bit_cast<float>(uint32_t{0x00800001}),0);
    for (uint32_t rounding : {0U,0x2000U,0x4000U,0x6000U})
        for (uint32_t denormals : {0U,0x40U,0x8000U,0x8040U}) {
            _mm_setcsr(0x1f80U|rounding|denormals);
            PreparedF64GEMVInput input;
            Check(PrepareF64GEMVInput(x,scratch,input)==Status::OK);
            Check(r8_frozen::RowPair(raw.data(),scales.data(),x.data(),expected.data(),128,2)==Status::OK);
            Check(GEMV_F32WeightsPreparedRows(weights[0],scales.data(),input,checked.data(),0,2)==Status::OK);
            Check(GEMV_F32WeightsPreparedRowsPrevalidatedScales(weights[0],scales.data(),input,pre.data(),0,2)==Status::OK);
            Equal(expected,checked); Equal(expected,pre);
        }
    _mm_setcsr(saved);
}
int main() {
    unsigned cases=0;
    for (uint32_t cols : {1U,7U,31U,32U,33U,63U,64U,65U,129U,256U,960U,2560U})
        for (uint32_t rows : {1U,2U,3U,4U,5U,7U,16U})
            for (uint32_t trial=0; trial<2; ++trial) { Shape(rows,cols,trial); ++cases; }
    Shape(1600,960,1); Shape(960,960,1); cases+=2;
    FailureAndTail();
    ExactRawDomainAndFPControls();
    std::cout << "pw2_shape_bitwise_cases=" << cases
              << "\npw2_raw_values=256\npw2_fp_control_cases=16\nstatus=PASS\n";
}

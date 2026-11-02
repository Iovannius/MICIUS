#include "I19AttentionScore.h"
#include <immintrin.h>
#include <algorithm>
#include <bit>
#include <cmath>
#include <limits>

#pragma STDC FENV_ACCESS ON

namespace {
inline void Update(__m128 column,const double* q,uint32_t d,
                   __m256d& a,__m256d& b,__m256d& c) noexcept
{
    const __m256d key=_mm256_cvtps_pd(column);
    a=_mm256_fmadd_pd(_mm256_set1_pd(q[d]),key,a);
    b=_mm256_fmadd_pd(_mm256_set1_pd(q[64+d]),key,b);
    c=_mm256_fmadd_pd(_mm256_set1_pd(q[128+d]),key,c);
}

// Three chronological four-row accumulators share each loaded, transposed,
// widened Key column. No horizontal reduction or per-row reassociation.
[[gnu::noinline]] void I19Dot4x3x64(const double* q,const float* k,
                                  size_t stride,double* dots) noexcept
{
    __m256d a=_mm256_setzero_pd(),b=a,c=a;
    const float* k1=k+stride;const float* k2=k1+stride;const float* k3=k2+stride;
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
    for(uint32_t d=0;d<64;d+=4) {
        __m128 x0=_mm_loadu_ps(k+d),x1=_mm_loadu_ps(k1+d);
        __m128 x2=_mm_loadu_ps(k2+d),x3=_mm_loadu_ps(k3+d);
        _MM_TRANSPOSE4_PS(x0,x1,x2,x3);
        Update(x0,q,d,a,b,c);
        Update(x1,q,d+1,a,b,c);
        Update(x2,q,d+2,a,b,c);
        Update(x3,q,d+3,a,b,c);
    }
    _mm256_storeu_pd(dots,a);_mm256_storeu_pd(dots+4,b);_mm256_storeu_pd(dots+8,c);
}

bool Disjoint(const void* a,size_t na,const void* b,size_t nb) noexcept
{
    const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    return na<=UINTPTR_MAX-x && nb<=UINTPTR_MAX-y && (x+na<=y || y+nb<=x);
}
bool Store(double dot,double den,float* out,double& maximum) noexcept
{
    const double score=dot/den;
    if(!std::isfinite(score))return false;
    const float narrowed=static_cast<float>(score);
    if(!std::isfinite(narrowed))return false;
    *out=narrowed;maximum=std::max(maximum,score);return true;
}
}

bool TryPrepareAttentionScoresI19(
    const float* q,const float* k,size_t stride,float* scratch,size_t score_stride,
    double* maxima,uint32_t rows,uint32_t dim,uint32_t group,double den) noexcept
{
    // Integer-only eligibility checks cannot raise a floating exception.
    if(!q || !k || !scratch || !maxima || rows<128 || rows>8192 || dim!=64 ||
       group!=3 || stride<64 || score_stride<rows ||
       std::bit_cast<uint64_t>(den)!=std::bit_cast<uint64_t>(8.0))return false;
    const size_t last=size_t(rows)-1;
    if(last>(SIZE_MAX-64)/stride || score_stride>(SIZE_MAX-rows)/2)return false;
    const size_t nk=last*stride+64,ns=2*score_stride+rows;
    if(nk>SIZE_MAX/sizeof(float) || ns>SIZE_MAX/sizeof(float))return false;
    if(!Disjoint(scratch,ns*sizeof(float),q,192*sizeof(float)) ||
       !Disjoint(scratch,ns*sizeof(float),k,nk*sizeof(float)) ||
       !Disjoint(maxima,3*sizeof(double),q,192*sizeof(float)) ||
       !Disjoint(maxima,3*sizeof(double),k,nk*sizeof(float)) ||
       !Disjoint(scratch,ns*sizeof(float),maxima,3*sizeof(double)))return false;
    const unsigned csr=_mm_getcsr();
    if((csr&0x1f80U)!=0x1f80U)return false;
    const auto reject=[&]() noexcept {_mm_setcsr(csr);return false;};
    alignas(32) double prepared[192];
    for(uint32_t d=0;d<192;++d)prepared[d]=static_cast<double>(q[d]);
    for(uint32_t h=0;h<3;++h)maxima[h]=-std::numeric_limits<double>::infinity();
    uint32_t t=0;
    for(;rows-t>=4;t+=4) {
        double dots[12];I19Dot4x3x64(prepared,k+size_t(t)*stride,stride,dots);
        for(uint32_t h=0;h<3;++h)for(uint32_t lane=0;lane<4;++lane)
            if(!Store(dots[h*4+lane],den,scratch+size_t(h)*score_stride+t+lane,maxima[h]))return reject();
    }
    // I17's scalar tail is a dimension-ordered F64 FMA. Retain q F32 widening
    // here (including DAZ) and never append a fictitious padded Key row.
    for(;t<rows;++t)for(uint32_t h=0;h<3;++h) {
        double dot=0;const float* row=k+size_t(t)*stride;
        for(uint32_t d=0;d<64;++d)
            dot=std::fma(static_cast<double>(q[h*64+d]),static_cast<double>(row[d]),dot);
        if(!Store(dot,den,scratch+size_t(h)*score_stride+t,maxima[h]))return reject();
    }
    // A later head must never leak a new sticky flag into an earlier head's
    // consumer/failure path. Successful speculation is invisible to MXCSR.
    if(_mm_getcsr()!=csr)return reject();
    return true;
}

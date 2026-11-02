#include "I17AttentionScore.h"
#include "AIComputKrnlInternal.h"
#include <immintrin.h>
#include <algorithm>
#include <cmath>
#include <limits>

namespace {
inline void Columns(const float* a, const float* b, const float* c, const float* d,
                    __m128& x0, __m128& x1, __m128& x2, __m128& x3) noexcept
{
    x0=_mm_loadu_ps(a);x1=_mm_loadu_ps(b);x2=_mm_loadu_ps(c);x3=_mm_loadu_ps(d);
    _MM_TRANSPOSE4_PS(x0,x1,x2,x3);
}

// Keep two chronological four-row dependency chains and the original dimension
// order. The query has been widened once, outside every historical-key batch.
[[gnu::noinline]] void I17Dot8x64(const double* q,const float* k,size_t stride,
                                 double* dots) noexcept
{
    __m256d a=_mm256_setzero_pd(),b=a;
    const float* k1=k+stride;const float* k2=k1+stride;const float* k3=k2+stride;
    const float* k4=k3+stride;const float* k5=k4+stride;const float* k6=k5+stride;const float* k7=k6+stride;
#if defined(__clang__)
#pragma clang loop unroll(disable)
#endif
    for(uint32_t d=0;d<64;d+=4) {
        __m128 x0,x1,x2,x3,y0,y1,y2,y3;
        Columns(k+d,k1+d,k2+d,k3+d,x0,x1,x2,x3);
        Columns(k4+d,k5+d,k6+d,k7+d,y0,y1,y2,y3);
        const __m256d q0=_mm256_set1_pd(q[d]);
        a=_mm256_fmadd_pd(q0,_mm256_cvtps_pd(x0),a);
        b=_mm256_fmadd_pd(q0,_mm256_cvtps_pd(y0),b);
        const __m256d q1=_mm256_set1_pd(q[d+1]);
        a=_mm256_fmadd_pd(q1,_mm256_cvtps_pd(x1),a);
        b=_mm256_fmadd_pd(q1,_mm256_cvtps_pd(y1),b);
        const __m256d q2=_mm256_set1_pd(q[d+2]);
        a=_mm256_fmadd_pd(q2,_mm256_cvtps_pd(x2),a);
        b=_mm256_fmadd_pd(q2,_mm256_cvtps_pd(y2),b);
        const __m256d q3=_mm256_set1_pd(q[d+3]);
        a=_mm256_fmadd_pd(q3,_mm256_cvtps_pd(x3),a);
        b=_mm256_fmadd_pd(q3,_mm256_cvtps_pd(y3),b);
    }
    _mm256_storeu_pd(dots,a);_mm256_storeu_pd(dots+4,b);
}

bool Consume(const double* dots,uint32_t n,double denominator,float* scores,double* maximum) noexcept
{
    for(uint32_t i=0;i<n;++i) {
        const double score=dots[i]/denominator;
        if(!std::isfinite(score))return false;
        scores[i]=static_cast<float>(score);
        *maximum=std::max(*maximum,score);
    }
    return true;
}

bool Disjoint(const void* a,size_t na,const void* b,size_t nb) noexcept
{
    const auto x=reinterpret_cast<uintptr_t>(a),y=reinterpret_cast<uintptr_t>(b);
    return na<=UINTPTR_MAX-x && nb<=UINTPTR_MAX-y && (x+na<=y || y+nb<=x);
}
}

I11ScoreStatus ComputeAttentionScoresI17(
    const float* q,const float* k,size_t stride,float* scores,uint32_t rows,
    uint32_t dim,double denominator,double* maximum) noexcept
{
    const auto original=[&]() noexcept {
        return ComputeAttentionScoresI11(q,k,stride,scores,rows,dim,denominator,maximum);
    };
    if(!q || !k || !scores || !maximum || rows<128 || dim!=64 || stride<64 ||
       !std::isfinite(denominator) || denominator<1.0)return original();
    const size_t last=static_cast<size_t>(rows)-1;
    if(last>(SIZE_MAX-64)/stride)return original();
    const size_t count=last*stride+64;
    if(count>SIZE_MAX/sizeof(float) || static_cast<size_t>(rows)>SIZE_MAX/sizeof(float))return original();
    const size_t bytes=static_cast<size_t>(rows)*sizeof(float);
    if(!Disjoint(scores,bytes,q,64*sizeof(float)) || !Disjoint(scores,bytes,k,count*sizeof(float)) ||
       !Disjoint(maximum,sizeof(double),q,64*sizeof(float)) || !Disjoint(maximum,sizeof(double),k,count*sizeof(float)) ||
       !Disjoint(scores,bytes,maximum,sizeof(double)))return original();
    const unsigned csr=_mm_getcsr();
    if((csr&0x1f80U)!=0x1f80U)return original();
    alignas(32) double prepared[64];
    for(uint32_t d=0;d<64;++d)prepared[d]=static_cast<double>(q[d]);
    *maximum=-std::numeric_limits<double>::infinity();
    uint32_t t=0;
    for(;rows-t>=8;t+=8) {
        double dots[8];
        I17Dot8x64(prepared,k+static_cast<size_t>(t)*stride,stride,dots);
        bool valid=true;
        for(double dot:dots)valid=valid && std::isfinite(dot);
        if(!valid) {
            // Remove speculative widening/row flags and replay accepted I11.
            _mm_setcsr(csr);
            return original();
        }
        Consume(dots,8,denominator,scores+t,maximum);
    }
    // Retain the accepted four-row and scalar remainder operations verbatim.
    for(;rows-t>=4;t+=4) {
        const float* row=k+static_cast<size_t>(t)*stride;double dots[4];
        micius::internal::AttentionScoreDotTimeQuadF32(q,row,row+stride,row+2*stride,row+3*stride,64,dots);
        if(!Consume(dots,4,denominator,scores+t,maximum))return I11ScoreStatus::NUMERIC_ERROR;
    }
    for(;t<rows;++t) {
        const float* row=k+static_cast<size_t>(t)*stride;double dot=0.0;
        for(uint32_t d=0;d<64;++d)dot+=static_cast<double>(q[d])*row[d];
        if(!Consume(&dot,1,denominator,scores+t,maximum))return I11ScoreStatus::NUMERIC_ERROR;
    }
    return I11ScoreStatus::OK;
}

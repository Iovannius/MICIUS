#include "PanelGEMV.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

#if defined(__FAST_MATH__) || (defined(__FINITE_MATH_ONLY__) && __FINITE_MATH_ONLY__)
#error "I23 requires strict floating point"
#endif
namespace micius::i23 {
void Matrix::Swap(Matrix& b) noexcept {
    std::swap(rows_,b.rows_);std::swap(cols_,b.cols_);
    weights_.swap(b.weights_);scales_.swap(b.scales_);
}
PackStatus Matrix::Pack(std::span<const int8_t> w,std::span<const QuantDescriptor> s,
                       uint32_t rows,uint32_t cols,size_t budget) noexcept {
    if(!rows || !cols || rows%4 || cols%32)return PackStatus::UNSUPPORTED;
    if(size_t(rows)>SIZE_MAX/cols)return PackStatus::INVALID;
    const size_t count=size_t(rows)*cols,nb=count/32;
    if(w.size()!=count || s.size()!=nb || nb>(SIZE_MAX-count)/4)return PackStatus::INVALID;
    if(count+nb*4>budget)return PackStatus::RESOURCE;
    for(const auto& x:s)if(!std::isfinite(x.scale)||x.zero_point||x.padding[0]||x.padding[1])return PackStatus::INVALID;
    try {
        Matrix next;next.rows_=rows;next.cols_=cols;
        next.weights_.resize(count);next.scales_.resize(nb);
        const size_t blocks=cols/32;
        for(size_t g=0;g<rows/4;++g)for(size_t b=0;b<blocks;++b) {
            const size_t panel=g*blocks+b;
            for(size_t j=0;j<4;++j)for(size_t h=0;h<2;++h)for(size_t r=0;r<4;++r) {
                const auto* src=w.data()+(4*g+r)*cols+32*b+8*j+4*h;
                std::memcpy(next.weights_.data()+128*panel+32*j+16*h+4*r,src,4);
            }
            for(size_t r=0;r<4;++r)
                std::memcpy(next.scales_.data()+4*panel+r,&s[(4*g+r)*blocks+b].scale,4);
        }
        Swap(next);return PackStatus::OK;
    } catch(const std::bad_alloc&) {return PackStatus::RESOURCE;}
      catch(const std::length_error&) {return PackStatus::RESOURCE;}
}
Status Expand(const Input& x,std::span<int8_t> scratch,Expanded& view) noexcept {
    view=Expanded{};
    if(!x.size()||x.size()%32||!x.values()||!x.scales()||size_t(x.size())>SIZE_MAX/4||
       scratch.size()<size_t(x.size())*4||(_mm_getcsr()&0xffc0U)!=0x1f80U)
        return Status::INVALID_ARGUMENT;
    for(size_t b=0;b<x.size()/32;++b)for(size_t j=0;j<4;++j)for(size_t h=0;h<2;++h)for(size_t r=0;r<4;++r)
        std::memcpy(scratch.data()+b*128+j*32+h*16+r*4,x.values()+32*b+8*j+4*h,4);
    view.values_=scratch.data();view.scales_=x.scales();view.cols_=x.size();return Status::OK;
}
[[gnu::noinline]] Status Consume(const Matrix& m,const Expanded& x,std::span<float> out) noexcept {
    if(!m.rows()||!m.cols()||m.cols()!=x.cols()||!x.values()||!x.scales()||out.size()!=m.rows()||
       (_mm_getcsr()&0xffc0U)!=0x1f80U)return Status::INVALID_ARGUMENT;
    const size_t blocks=m.blocks();
    for(size_t g=0;g<m.rows()/4;++g) {
        auto sum=_mm256_setzero_pd();
        const int8_t* w=m.weights()+g*blocks*128;
        const float* s=m.scales()+g*blocks*4;
        for(size_t b=0;b<blocks;++b) {
            const auto dot=_mm256_cvtepi32_pd(Dot(x.values()+128*b,w+128*b));
            const auto scale=_mm256_cvtps_pd(_mm_loadu_ps(s+4*b));
            const auto weighted=_mm256_mul_pd(dot,scale);
            sum=_mm256_add_pd(sum,_mm256_mul_pd(weighted,_mm256_set1_pd(x.scales()[b])));
        }
        const auto mag=_mm256_andnot_pd(_mm256_set1_pd(-0.0),sum);
        const auto valid=_mm256_cmp_pd(mag,_mm256_set1_pd(std::numeric_limits<float>::max()),_CMP_LE_OQ);
        if(_mm256_movemask_pd(valid)!=15)return Status::NUMERIC_ERROR;
        _mm_storeu_ps(out.data()+4*g,_mm256_cvtpd_ps(sum));
    }
    return Status::OK;
}
}

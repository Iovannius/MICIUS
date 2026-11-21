#include "WeightFormat.h"
#include "app/MiciusTensorCodec.h"
#include <cmath>
#include <cstring>
#include <limits>

namespace micius::quant {
Layout Block(Type t) noexcept {
    switch(t) {
    case Type::F32:return {1,4}; case Type::F16:return {1,2};
    case Type::Q8_0:return {32,34}; case Type::Q4_K:return {256,144};
    case Type::Q6_K:return {256,210};
    } return {};
}
size_t RowBytes(Type t,size_t n) noexcept {
    const auto b=Block(t);
    if(!n || !b.elements || n%b.elements || n/b.elements>SIZE_MAX/b.bytes)return 0;
    return n/b.elements*b.bytes;
}
static float Half(const uint8_t* p) noexcept {
    uint16_t bits;std::memcpy(&bits,p,2);return MiciusTensorCodec::DecodeFloat16(bits);
}
static int Signed(uint8_t b) noexcept {return b<128 ? b : int(b)-256;}
static void DecodeBlock(Type t,const uint8_t* p,float* y) noexcept {
    if(t==Type::F32) {std::memcpy(y,p,4);return;}
    if(t==Type::F16) {y[0]=Half(p);return;}
    if(t==Type::Q8_0) {
        const float d=Half(p);
        for(int i=0;i<32;++i)y[i]=d*Signed(p[2+i]);
        return;
    }
    if(t==Type::Q4_K) {
        const float d=Half(p), dm=Half(p+2);const uint8_t* s=p+4;
        for(int group=0;group<8;++group) {
            const int scale=group<4 ? s[group]&63 : (s[group+4]&15)|((s[group-4]>>6)<<4);
            const int min=group<4 ? s[group+4]&63 : (s[group+4]>>4)|((s[group]>>6)<<4);
            const float ds=d*scale, ms=dm*min;
            for(int j=0;j<32;++j) {
                const unsigned q=(p[16+(group/2)*32+j]>>(4*(group%2)))&15;
                y[group*32+j]=ds*q-ms;
            }
        }
        return;
    }
    const float d=Half(p+208);
    for(int i=0;i<256;++i) {
        const int half=i/128, sub=(i%128)/32, j=i%32;
        const unsigned low=(p[half*64+(sub%2)*32+j]>>(sub>=2?4:0))&15;
        const unsigned high=(p[128+half*32+j]>>(2*sub))&3;
        const int q=int(low|(high<<4))-32;
        y[i]=(d*Signed(p[192+i/16]))*q;
    }
}
bool Decode(Type t,const void* src,size_t bytes,float* out,size_t n) noexcept {
    const size_t expected=RowBytes(t,n);
    if(!src || !out || !expected || bytes!=expected)return false;
    const auto b=Block(t);const auto* p=static_cast<const uint8_t*>(src);
    for(size_t i=0;i<n;i+=b.elements,p+=b.bytes)DecodeBlock(t,p,out+i);
    for(size_t i=0;i<n;++i)if(!std::isfinite(out[i]))return false;
    return true;
}
bool Matrix::Valid() const noexcept {
    const size_t rb=RowBytes(type,cols);
    if(!data || !rows || !rb || rows>SIZE_MAX/rb || bytes!=rows*rb)return false;
    const auto b=Block(type);const auto* p=static_cast<const uint8_t*>(data);float values[256];
    for(size_t i=0;i<bytes;i+=b.bytes)if(!Decode(type,p+i,b.bytes,values,b.elements))return false;
    return true;
}
bool Matrix::Row(uint32_t row,float* out) const noexcept {
    const auto rb=RowBytes(type,cols);
    return row<rows && data && rb && Decode(type,static_cast<const uint8_t*>(data)+row*rb,rb,out,cols);
}
bool Matrix::Multiply(const float* x,float* y) const noexcept {
    const auto b=Block(type);const size_t rb=RowBytes(type,cols);
    if(!x || !y || !data || !b.elements || !rb)return false;
    for(uint32_t i=0;i<cols;++i)if(!std::isfinite(x[i]))return false;
    float values[256];const auto* p=static_cast<const uint8_t*>(data);
    for(uint32_t row=0;row<rows;++row) {
        double sum=0;
        for(uint32_t col=0;col<cols;col+=static_cast<uint32_t>(b.elements),p+=b.bytes) {
            DecodeBlock(type,p,values);
            for(size_t j=0;j<b.elements;++j)sum+=double(values[j])*double(x[col+j]);
        }
        if(!std::isfinite(sum) || std::abs(sum)>std::numeric_limits<float>::max())return false;
        y[row]=static_cast<float>(sum);
    }
    return true;
}
}

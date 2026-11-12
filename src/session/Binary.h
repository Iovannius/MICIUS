#pragma once
#include <bit>
#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>
namespace micius::state {
inline size_t Product(size_t a,size_t b) {
    if (a && b>SIZE_MAX/a) throw std::runtime_error("Snapshot size overflow");
    return a*b;
}
struct Writer {
    std::vector<uint8_t> bytes;
    void Raw(const void* p,size_t n) { if(n) { auto* b=static_cast<const uint8_t*>(p); bytes.insert(bytes.end(),b,b+n); } }
    void U32(uint32_t x) { for(int i=0;i<4;++i) bytes.push_back(static_cast<uint8_t>(x>>(8*i))); }
    void U64(uint64_t x) { U32(static_cast<uint32_t>(x)); U32(static_cast<uint32_t>(x>>32)); }
    void Float(float x) { U32(std::bit_cast<uint32_t>(x)); }
    void Double(double x) { U64(std::bit_cast<uint64_t>(x)); }
    void Text(const std::string& x) { if(x.size()>UINT32_MAX) throw std::runtime_error("String too large"); U32(static_cast<uint32_t>(x.size()));Raw(x.data(),x.size()); }
};
struct Reader {
    std::span<const uint8_t> bytes; size_t pos=0;
    std::span<const uint8_t> Raw(size_t n) { if(n>bytes.size()-pos)throw std::runtime_error("Truncated snapshot");auto r=bytes.subspan(pos,n);pos+=n;return r; }
    uint32_t U32() { auto b=Raw(4);return uint32_t(b[0])|(uint32_t(b[1])<<8)|(uint32_t(b[2])<<16)|(uint32_t(b[3])<<24); }
    uint64_t U64() { uint64_t lo=U32();return lo|(uint64_t(U32())<<32); }
    float Float() { return std::bit_cast<float>(U32()); }
    double Double() { return std::bit_cast<double>(U64()); }
    std::string Text(size_t max=4096) { auto n=U32();if(n>max)throw std::runtime_error("Oversized snapshot string");auto b=Raw(n);return {reinterpret_cast<const char*>(b.data()),b.size()}; }
    void End() { if(pos!=bytes.size())throw std::runtime_error("Trailing snapshot data"); }
};
}

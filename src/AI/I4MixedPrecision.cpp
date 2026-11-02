#include "I4MixedPrecision.h"
#include "PreparedGEMVInput.h"
#include <immintrin.h>
#ifndef MICIUS_I4_MODE
#define MICIUS_I4_MODE 0
#endif

namespace micius::i4 {
namespace { thread_local RouteAudit* sink=nullptr; }
Policy CompiledPolicy() noexcept { return static_cast<Policy>(MICIUS_I4_MODE); }
const char* PolicyName(Policy p) noexcept {
    switch(p) {
        case Policy::None:return "NONE";
        case Policy::L31GU:return "L31_GU";
        case Policy::BothFFN:return "L3_L31_FFN";
        case Policy::AllF64:return "F64_ALL";
    }
    return "INVALID";
}
bool Select(Policy p,uint32_t layer,Op op) noexcept {
    if(p==Policy::AllF64)return true;
    if(p==Policy::L31GU)return layer==31 && (op==Op::Gate || op==Op::Up);
    return p==Policy::BothFFN && (layer==3 || layer==31) &&
        (op==Op::Gate || op==Op::Up || op==Op::Down);
}
Q8_32_GEMVStatus F64(const int8_t* w,const QuantDescriptor* s,const float* x,float* y,
                    uint32_t cols,uint32_t rows,bool validated) noexcept {
    if(!w || !s || !x || !y || !cols || !rows || (_mm_getcsr()&0xffc0U)!=0x1f80U)
        return Q8_32_GEMVStatus::INVALID_ARGUMENT;
    try {
        std::vector<double> scratch(cols);
        internal::PreparedF64GEMVInput view;
        auto status=internal::PrepareF64GEMVInput({x,cols},scratch,view);
        if(status!=Q8_32_GEMVStatus::OK)return status;
        return validated ? internal::GEMV_Q8_32_PreparedF64RowsPrevalidatedScales(w,s,view,y,rows,0,rows)
                         : internal::GEMV_Q8_32_PreparedF64Rows(w,s,view,y,rows,0,rows);
    } catch(...) { return Q8_32_GEMVStatus::INVALID_MODEL; }
}
RouteAudit::RouteAudit(uint32_t layers) noexcept : layers_(layers) {
    if(sink || !layers || layers>4096) {good_=false;return;}
    try {seen_.reserve(static_cast<size_t>(layers)*5U+1U);}
    catch(...) {good_=false;return;}
    active_=true;sink=this;
}
RouteAudit::~RouteAudit() {if(active_)sink=nullptr;}
bool RecordSelected(uint32_t layer,Op op) noexcept {
    if(!sink)return true;
    if(!sink->good_ || sink->finished_ || layer>sink->layers_)return false;
    try {sink->seen_.emplace_back(layer,op);return true;}
    catch(...) {sink->good_=false;return false;}
}
bool RouteAudit::finish() noexcept {
    if(!good_ || finished_)return false;
    finished_=true;
    // Explicit expected routes, independent of the engine's Select predicate.
    size_t index=0;
    auto consume=[&](uint32_t layer,Op op) {
        if(index>=seen_.size() || seen_[index]!=std::pair{layer,op})return false;
        ++index;return true;
    };
    switch(CompiledPolicy()) {
        case Policy::None:break;
        case Policy::L31GU:
            if(layers_<=31 || !consume(31,Op::Gate) || !consume(31,Op::Up))return false;
            break;
        case Policy::BothFFN:
            if(layers_<=31)return false;
            for(uint32_t layer:{3U,31U})
                for(Op op:{Op::Gate,Op::Up,Op::Down})if(!consume(layer,op))return false;
            break;
        case Policy::AllF64:
            for(uint32_t layer=0;layer<layers_;++layer)
                for(Op op:{Op::QKV,Op::O,Op::Gate,Op::Up,Op::Down})if(!consume(layer,op))return false;
            if(!consume(layers_,Op::LM))return false;
            break;
    }
    return index==seen_.size();
}
} // namespace micius::i4

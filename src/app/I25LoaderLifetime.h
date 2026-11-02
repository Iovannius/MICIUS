#pragma once
#include "AI/I25Binding.h"
#include <algorithm>
#include <immintrin.h>
#include <vector>
namespace micius::i25 {
// Validate the entire retirement set before the first free. An engine-owned
// replacement must exist for every alias. F32 tensors/mappings stay borrowed.
inline bool ReleaseWeights(std::vector<void*>& owners,
    const std::vector<MiciusTensorEntryV1>& entries,
    std::span<const RetiredWeight> retired,size_t& released) noexcept {
    released=0;
    for(size_t i=0;i<retired.size();++i) {
        const auto& r=retired[i];const auto p=reinterpret_cast<const void*>(r.address);
        if(!r.address || !r.bytes || r.bytes>UINTPTR_MAX-r.address ||
           std::count(owners.begin(),owners.end(),p)!=1 || r.bytes>SIZE_MAX-released) return false;
        for(size_t j=0;j<i;++j) if(retired[j].address==r.address) return false;
        bool found=false;
        for(const auto& e:entries) {
            const auto a=reinterpret_cast<uintptr_t>(e.tensor_data);
            if(!a || e.byte_size>UINTPTR_MAX-a) return false;
            if(a<r.address+r.bytes && r.address<a+e.byte_size) {
                if(a!=r.address || e.byte_size!=r.bytes || e.data_type!=MICIUS_TENSOR_Q8_32_WEIGHT) return false;
                found=true;
            }
        }
        if(!found) return false;
        released+=r.bytes;
    }
    for(const auto& r:retired) {
        auto it=std::find(owners.begin(),owners.end(),reinterpret_cast<void*>(r.address));
        _mm_free(*it);owners.erase(it);
    }
    return true;
}
}

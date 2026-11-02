#pragma once
#include "PanelGEMV.h"
#include "AI/I25Binding.h"
#include <memory>
#include <stdexcept>
namespace micius::i25 {
struct Panels {
    std::vector<i23::Matrix> matrices;
    std::vector<int8_t> scratch;
    std::vector<RetiredWeight> retired;
    Stats stats;
    void Pack(const int8_t* w,const QuantDescriptor* s,uint32_t rows,uint32_t cols) {
        i23::Matrix next;
        const auto result=next.Pack({w,size_t(rows)*cols},{s,size_t(rows)*(cols/32)},rows,cols);
        if(result==i23::PackStatus::RESOURCE) throw std::bad_alloc();
        if(result!=i23::PackStatus::OK) throw std::runtime_error("I25 invalid panel source");
        stats.payload_bytes+=next.payload();stats.capacity_bytes+=next.capacity_bytes();
        matrices.push_back(std::move(next));++stats.matrices;
    }
    void Retire(const void* p,size_t bytes) {
        const auto addr=reinterpret_cast<uintptr_t>(p);
        for(const auto& r:retired) if(r.address==addr) {
            if(r.bytes!=bytes) throw std::runtime_error("I25 inconsistent aliased weight size");
            return;
        }
        retired.push_back({addr,bytes});stats.retired_weight_bytes+=bytes;++stats.retired_weight_buffers;
    }
};
}

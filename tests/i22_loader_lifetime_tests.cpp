#include "app/I22LoaderLifetime.h"
#include <cassert>
#include <iostream>
#include <cstring>
#include <stdexcept>
void need(bool x){if(!x)throw std::runtime_error("ownership contract failed");}
int main() {
    using namespace micius::i22;
    size_t cases=0;
    for(unsigned n:{0,1,2,224,226}) {
        std::vector<void*> owners;std::vector<MiciusTensorEntryV1> entries;
        auto* raw=_mm_malloc(64,64);need(raw!=nullptr);std::memset(raw,0x5a,64);owners.push_back(raw);
        MiciusTensorEntryV1 w{};w.tensor_data=raw;w.byte_size=64;w.data_type=MICIUS_TENSOR_Q8_32_WEIGHT;entries.push_back(w);
        for(unsigned i=0;i<n;++i){auto* p=_mm_malloc(16*(i+1),64);need(p!=nullptr);owners.push_back(p);
            MiciusTensorEntryV1 s{};s.tensor_data=p;s.byte_size=16*(i+1);s.data_type=MICIUS_TENSOR_Q8_32_SCALE;entries.push_back(s);}
        ReleaseStats stats;need(ReleasePublicScales(owners,entries,stats));
        need(stats.buffers==n && stats.bytes==size_t(n)*(n+1)*8 && owners.size()==1 && owners[0]==raw);
        for(unsigned i=0;i<64;++i)need(static_cast<unsigned char*>(raw)[i]==0x5a);
        if(n)need(!ReleasePublicScales(owners,entries,stats)); // repeated cleanup must fail without freeing raw
        _mm_free(raw);++cases;
    }
    for(int kind=0;kind<4;++kind){auto* p=_mm_malloc(64,64);auto* q=_mm_malloc(64,64);need(p&&q);
        std::vector<void*> owners{p,q};std::vector<MiciusTensorEntryV1> entries(2);
        for(int i=0;i<2;++i){entries[i].tensor_data=i?q:p;entries[i].byte_size=64;entries[i].data_type=MICIUS_TENSOR_Q8_32_SCALE;}
        if(kind==0)owners.push_back(q);
        if(kind==1)entries[1].tensor_data=p;
        if(kind==2)entries[1].tensor_data=nullptr;
        if(kind==3)entries[1].byte_size=0;
        const auto before=owners;ReleaseStats stats;need(!ReleasePublicScales(owners,entries,stats));need(owners==before);
        std::memset(p,1,64);std::memset(q,2,64);_mm_free(p);_mm_free(q);++cases;
    }
    std::cout<<"PASS ownership cases="<<cases<<"\n";
}

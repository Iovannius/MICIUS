#include "micius_sampling.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <vector>

extern "C" int Micius_InitSamplingV1(MiciusSamplingOptionsV1* p, size_t size)
{
    if (!p || size < sizeof(*p)) return 0;
    *p={MICIUS_SAMPLING_ABI_V1,sizeof(*p),0,1,1,0,0}; return 1;
}
extern "C" int Micius_SampleTokenV1(const MiciusSamplingOptionsV1* p,
    const float* logits, size_t count, const uint8_t* allowed, const uint32_t* history, size_t length,
    uint64_t* rng, uint32_t* output)
{
    if (!p || p->abi_version!=MICIUS_SAMPLING_ABI_V1 || p->struct_size<sizeof(*p) ||
        !logits || !count || count>UINT32_MAX || (!history && length) || !rng || !output ||
        !std::isfinite(p->temperature) || p->temperature<0 ||
        !std::isfinite(p->top_p) || p->top_p<=0 || p->top_p>1 ||
        !std::isfinite(p->repeat_penalty) || p->repeat_penalty<=0) return 0;
    try {
        struct Item { double value; uint32_t id; };
        std::vector<Item> scores; scores.reserve(count);
        std::vector<bool> repeated(count,false);
        for (size_t i=length-std::min<size_t>(length,p->repeat_last_n); i<length; ++i) {
            if (history[i]>=count) return 0;
            repeated[history[i]]=true;
        }
        uint32_t best=0; double highest=-std::numeric_limits<double>::infinity();
        for (size_t i=0;i<count;++i) {
            double v=logits[i]; if (!std::isfinite(v)) return 0;
            if (allowed && !allowed[i]) continue;
            if (repeated[i]) v=v<0 ? v*p->repeat_penalty : v/p->repeat_penalty;
            if (!std::isfinite(v)) return 0;
            if (v>highest) { highest=v; best=static_cast<uint32_t>(i); }
            scores.push_back({v,static_cast<uint32_t>(i)});
        }
        if (scores.empty()) return 0;
        if (p->temperature==0) { *output=best; return 1; }
        auto order=[](const Item& a,const Item& b) {
            return a.value!=b.value ? a.value>b.value : a.id<b.id;
        };
        const size_t k=p->top_k ? std::min<size_t>(scores.size(),p->top_k) : scores.size();
        std::partial_sort(scores.begin(),scores.begin()+k,scores.end(),order);
        scores.resize(k);
        double total=0;
        for (auto& item:scores) { item.value=std::exp((item.value-highest)/p->temperature); total+=item.value; }
        double kept=0; size_t n=0;
        do { kept+=scores[n++].value; } while (n<k && kept<p->top_p*total);
        uint64_t state=*rng+UINT64_C(0x9e3779b97f4a7c15), z=state;
        z=(z^(z>>30))*UINT64_C(0xbf58476d1ce4e5b9);
        z=(z^(z>>27))*UINT64_C(0x94d049bb133111eb);
        z^=z>>31;
        const double choice=static_cast<double>(z>>11)*0x1.0p-53*kept;
        double cumulative=0; uint32_t selected=scores[n-1].id;
        for (size_t i=0;i<n;++i) {
            cumulative+=scores[i].value;
            if (choice<cumulative) { selected=scores[i].id; break; }
        }
        *output=selected; *rng=state; return 1;
    } catch (...) { return 0; }
}

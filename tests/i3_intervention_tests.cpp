#include "AI/I3Intervention.h"
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <random>
#include <string>
#include <vector>
#ifdef NDEBUG
#error "I3 tests require active assertions"
#endif
namespace r8_frozen {
Q8_32_GEMVStatus RowPair(const int8_t*,const QuantDescriptor*,const float*,float*,uint32_t,uint32_t) noexcept;
}
using namespace micius::i3;
using Status = Q8_32_GEMVStatus;
int main() {
    const auto saved = _mm_getcsr(); _mm_setcsr(0x1f80U);
    std::mt19937 rng(20261011);
    size_t shapes = 0;
    for (uint32_t cols : {1U,15U,31U,32U,33U,63U,65U,960U,2560U})
      for (uint32_t rows : {1U,2U,3U,5U}) {
        std::vector<float> x(cols), expected(rows), y(rows+2,-997.0f);
        std::vector<int8_t> w(static_cast<size_t>(cols)*rows);
        std::vector<QuantDescriptor> d(static_cast<size_t>((cols+31U)/32U)*rows);
        for (auto& v : x) v = std::ldexp(static_cast<float>(static_cast<int>(rng()%2001U)-1000)/321.0f,static_cast<int>(rng()%33U)-16);
        x.back() = -0.0f;
        for (auto& v : w) v = static_cast<int8_t>(static_cast<int>(rng()%256U)-128);
        for (auto& v : d) v = QuantDescriptor(static_cast<float>(static_cast<int>(rng()%17U)-8)/521.0f,0);
        const auto original = x;
        assert(r8_frozen::RowPair(w.data(),d.data(),x.data(),expected.data(),cols,rows)==Status::OK);
        for (bool validated : {false,true}) {
            assert(F64(w.data(),d.data(),x.data(),y.data()+1,cols,rows,validated)==Status::OK);
            assert(std::memcmp(expected.data(),y.data()+1,static_cast<size_t>(rows)*4)==0);
            assert(std::memcmp(original.data(),x.data(),static_cast<size_t>(cols)*4)==0);
            assert(y.front()==-997.0f && y.back()==-997.0f);
        }
        x[0]=std::numeric_limits<float>::quiet_NaN();
        assert(F64(w.data(),d.data(),x.data(),y.data()+1,cols,rows,false)==Status::NUMERIC_ERROR);
        ++shapes;
    }
    std::vector<float> x(33,1.0f), y(3);
    std::vector<int8_t> w(99,-128);
    std::vector<QuantDescriptor> d(6,QuantDescriptor(0.5f,0));
    assert(F64(nullptr,d.data(),x.data(),y.data(),33,3,false)==Status::INVALID_ARGUMENT);
    d[0].zero_point=1;
    assert(F64(w.data(),d.data(),x.data(),y.data(),33,3,false)==Status::INVALID_MODEL);
    d[0].zero_point=0;
    _mm_setcsr(0x9f80U);
    assert(F64(w.data(),d.data(),x.data(),y.data(),33,3,false)!=Status::OK);
    _mm_setcsr(0x1f80U);
    const unsigned expected_counts[] = {0,2,1,3,2,1,3,6,161};
    for (unsigned p=0;p<9;++p) {
        const auto policy=static_cast<Policy>(p);
        Policy parsed; assert(ParsePolicy(PolicyName(policy),parsed) && parsed==policy);
        unsigned hits=0;
        for (uint32_t layer=0;layer<32;++layer) for (unsigned op=0;op<5;++op) {
            const bool selected=Select(policy,layer,static_cast<Op>(op));
            if (selected) {
                ++hits;
                assert(p==8 || ((layer==3 || layer==31) && op>=2));
                if (p==1 || p==4) assert(op==2 || op==3);
                if (p==2 || p==5) assert(op==4);
                if (p>=1 && p<=3) assert(layer==3);
                if (p>=4 && p<=6) assert(layer==31);
            }
        }
        if (Select(policy,32,Op::LM)) { ++hits; assert(p==8); }
        assert(hits==expected_counts[p]);
    }
    Policy unused; assert(!ParsePolicy("l3_gu",unused)); assert(!ParsePolicy("NONE ",unused));
    const auto root=std::filesystem::current_path()/("i3-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    std::filesystem::create_directory(root);
    auto path=[&](const char* name){return (root/name).string();};
    assert(!Session(42,0,32,"INVALID",path("bad-policy").c_str()).good());
    assert(!Session(42,1,32,"NONE",path("bad-position").c_str()).good());
    assert(!Session(42,0,2,"L31_GU",path("bad-layer").c_str()).good());
    assert(!Session(42,0,32,"L3_GU","").good());
    assert(!Session(42,0,32,"",path("output-only").c_str()).good());
    assert(!Session(42,0,32,"NONE",root.string().c_str()).good());
    { Session disabled(42,4,32,"",""); assert(disabled.good() && !disabled.selected() && disabled.complete()); }
    auto prefix=[&](Session& session) {
        for (Op op : {Op::QKV,Op::O}) {
            session.set(0,op); assert(session.begin(x.data(),33,3)); assert(session.end(x.data(),y.data()));
        }
    };
    for (const char* policy : {"NONE","F64_ALL"}) {
        Session session(42,0,1,policy,path(policy).c_str()); assert(session.good());
        assert(session.boundary(-1,"embedding",x.data(),33));
        prefix(session);
        assert(session.start_ffn(0,x.data(),33));
        for (Op op : {Op::Gate,Op::Up}) {
            session.set(0,op); assert(session.begin(x.data(),33,3)); assert(session.end(x.data(),y.data()));
        }
        assert(session.end_ffn(x.data(),33));
        session.set(0,Op::Down); assert(session.begin(x.data(),33,3)); assert(session.end(x.data(),y.data()));
        session.set(1,Op::LM); assert(session.begin(x.data(),33,3)); assert(session.end(x.data(),y.data()));
        const auto before=_mm_getcsr(); assert(session.complete()); assert(_mm_getcsr()==before);
        assert(!session.complete());
    }
    { Session s(42,0,1,"NONE",path("mutated-consumption").c_str());
      s.set(0,Op::QKV); assert(s.begin(x.data(),33,3)); x[0]+=1; assert(!s.end(x.data(),y.data())); x[0]-=1; }
    { Session s(42,0,1,"NONE",path("mutated-shared-preparation").c_str());
      prefix(s); assert(s.start_ffn(0,x.data(),33)); x[0]+=1;
      for (Op op : {Op::Gate,Op::Up}) { s.set(0,op); assert(s.begin(x.data(),33,3)); assert(s.end(x.data(),y.data())); }
      assert(!s.end_ffn(x.data(),33)); x[0]-=1; }
    { Session s(42,0,1,"NONE",path("incomplete").c_str()); assert(!s.complete()); }
    { Session s(42,0,1,"NONE",path("wrong-order").c_str()); s.set(0,Op::Down); assert(!s.begin(x.data(),33,3)); }
    _mm_setcsr(saved);
    std::cout << "I3 PASS: frozen kernel parity " << shapes << " shapes x 2 scale routes; 9 routing policies; lifecycle, input mutation and invalid protocol checks\n";
}

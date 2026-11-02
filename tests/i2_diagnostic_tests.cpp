#include "AI/I2Diagnostic.h"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <immintrin.h>
#include <iostream>
#include <limits>
#include <random>
#include <vector>
#ifdef NDEBUG
#error "I2 tests need active assertions"
#endif
using namespace micius;
template<class F> void rejects(F&& f) { bool caught=false; try { f(); } catch(const std::exception&) { caught=true; } assert(caught); }
int main() {
    const auto saved=_mm_getcsr(); _mm_setcsr(0x1f80U);
    std::mt19937 rng(20261010);
    size_t cases=0;
    for (uint32_t cols : {1U,15U,31U,32U,33U,63U,65U,960U,2560U})
      for (uint32_t rows : {1U,2U,3U}) {
        const uint32_t blocks=(cols+31U)/32U;
        std::vector<float> x(cols);
        std::vector<int8_t> w(static_cast<size_t>(cols)*rows), q(cols);
        std::vector<double> scales(blocks);
        std::vector<QuantDescriptor> d(static_cast<size_t>(blocks)*rows,QuantDescriptor(0.015625f,0));
        for(auto& v:x) v=static_cast<float>(static_cast<int>(rng()%2001U)-1000)/321.0f;
        for(auto& v:w) v=static_cast<int8_t>(static_cast<int>(rng()%256U)-128);
        auto e=i2::Evaluate(x.data(),cols,rows,w.data(),d.data());
        assert(e.ok && e.json.find("\"ok\":true")!=std::string::npos);
        assert(i2::Evaluate(x.data(),cols,rows,w.data(),d.data(),e.avx.data()).ok);
        assert(i2::Evaluate(x.data(),cols,rows,w.data(),d.data(),e.f64.data(),0).ok);
        auto corrupt=e.avx; corrupt[0]=std::nextafter(corrupt[0],std::numeric_limits<float>::infinity());
        auto bad=i2::Evaluate(x.data(),cols,rows,w.data(),d.data(),corrupt.data());
        assert(!bad.ok && bad.json.find("\"actual_output\":1")!=std::string::npos);
        internal::PreparedW8A8Input view;
        assert(internal::PrepareW8A8Input(x,q,scales,view)==Q8_32_GEMVStatus::OK);
        std::vector<int32_t> saved_dots(static_cast<size_t>(rows)*blocks);
        for(uint32_t r=0;r<rows;++r) for(uint32_t b=0;b<blocks;++b)
            for(uint32_t k=b*32U;k<std::min(cols,b*32U+32U);++k)
                saved_dots[static_cast<size_t>(r)*blocks+b]+=static_cast<int32_t>(w[static_cast<size_t>(r)*cols+k])*q[k];
        ++saved_dots.back();
        bad=i2::Evaluate(x.data(),cols,rows,w.data(),d.data(),nullptr,2,&view,saved_dots.data());
        assert(!bad.ok && bad.json.find("\"actual_dots\":1")!=std::string::npos);
        assert(bad.worst_row==rows-1);
        q[0]=q[0]==0?1:0;
        bad=i2::Evaluate(x.data(),cols,rows,w.data(),d.data(),nullptr,2,&view);
        assert(!bad.ok && bad.json.find("\"prepared_codes\":1")!=std::string::npos);
        assert(bad.json.find("\"avx_dots\":0")!=std::string::npos); // preparation is isolated from dot arithmetic
        x[0]=std::numeric_limits<float>::quiet_NaN();
        rejects([&]{i2::Evaluate(x.data(),cols,rows,w.data(),d.data());});
        ++cases;
      }
    std::vector<float> x(33, std::numeric_limits<float>::denorm_min());
    std::vector<int8_t> w(66,-128);
    std::vector<QuantDescriptor> d(4,QuantDescriptor(0.5f,0));
    assert(i2::Evaluate(x.data(),33,2,w.data(),d.data()).ok);
    x.assign(33,0.0f); assert(i2::Evaluate(x.data(),33,2,w.data(),d.data()).ok);
    x[0]=127.0f; x[1]=0.5f; x[2]=1.5f; x[3]=-2.5f;
    auto ties=i2::Evaluate(x.data(),33,2,w.data(),d.data());
    assert(ties.ok && ties.codes[1]==0 && ties.codes[2]==2 && ties.codes[3]==-2);
    _mm_setcsr(0x9f80U); rejects([&]{i2::Evaluate(x.data(),33,2,w.data(),d.data());}); _mm_setcsr(0x1f80U);
    d[0].padding[1]=1; rejects([&]{i2::Evaluate(x.data(),33,2,w.data(),d.data());}); d[0].padding[1]=0;
    const auto dir=std::filesystem::current_path()/("i2-test-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
    assert(!std::filesystem::exists(dir));
    {
        i2::Session session(42,0,2,dir.string().c_str());
        session.set(0,"tail_test");
        auto capture=session.capture(x.data(),33,2);
        std::vector<int8_t> codes(33); std::vector<double> scales(2); std::vector<float> y(2);
        internal::PreparedW8A8Input prepared;
        assert(internal::PrepareW8A8Input(x,codes,scales,prepared)==Q8_32_GEMVStatus::OK);
        assert(internal::GEMV_W8A8Rows(w.data(),d.data(),prepared,y.data(),2,0,2,false,false)==Q8_32_GEMVStatus::OK);
        const auto csr=_mm_getcsr();
        assert(session.finish(capture,w.data(),d.data(),x.data(),y.data(),2,&prepared));
        assert(_mm_getcsr()==csr && session.complete());
    }
    i2::Replay(dir/"gemv-0.i2r",dir/"replay.json");
    rejects([&]{i2::Replay(dir/"gemv-0.i2r",dir/"replay.json");});
    { std::ofstream f(dir/"truncated.i2r",std::ios::binary); f << "MICI2R1"; }
    rejects([&]{i2::Replay(dir/"truncated.i2r",dir/"truncated.json");});
    { i2::Session duplicate(42,0,2,dir.string().c_str()); assert(!duplicate.good()); }
    // Keep the small record/replay fixture for inspection; do not delete user files.
    _mm_setcsr(saved);
    std::cout << "I2 diagnosis: " << cases << " shapes; corruption, tails, finite/FP, recording and replay PASS\n";
}

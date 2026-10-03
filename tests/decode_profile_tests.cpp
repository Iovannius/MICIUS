#define MICIUS_D1_TEST_CLOCK 1
#include "AI/DecodeProfile.h"
#include <cassert>
#include <iostream>
#include <stdexcept>
static uint64_t tick=0, reads=0;
uint64_t micius::d1::TestNow() noexcept { ++reads; return tick; }
using namespace micius::d1;
static const Counter& counter(Part p) { return state.counters[static_cast<unsigned>(p)]; }
int main() {
    // Disabled instrumentation must not read the clock or count calls.
    { Session s(false); Scope f(Part::Forward); Scope q(Part::QKV); }
    assert(reads==0 && counter(Part::Forward).calls==0);
    {
        Session s(true); assert(s.valid()); tick=100;
        { Scope f(Part::Forward); tick=110;
          { Scope q(Part::QKV); tick=115;
            { Scope p(Part::PrepQKV); tick=120; }
            tick=150;
          }
          tick=200;
        }
        assert(s.stop());
    }
    assert(reads==6);
    assert(counter(Part::Forward).inclusive_ns==100 && counter(Part::Forward).exclusive_ns==60);
    assert(counter(Part::QKV).inclusive_ns==40 && counter(Part::QKV).exclusive_ns==35);
    assert(counter(Part::PrepQKV).inclusive_ns==5 && counter(Part::PrepQKV).calls==1);
    // Unwind a failed consumer, prevent nested reset, then reset cleanly.
    { Session s(true); tick=300;
      try { Scope f(Part::Forward); assert(!Begin(true)); tick=310; throw std::runtime_error("test"); }
      catch(const std::runtime_error&) {}
      assert(state.top==nullptr && counter(Part::Forward).inclusive_ns==10);
    }
    { Session s(true); assert(counter(Part::Forward).calls==0);
      tick=400; { Scope f(Part::Forward); assert(!End()); tick=410; }
      assert(s.stop());
    }
    assert(!state.enabled && state.top==nullptr);
    std::cout << "profile_accounting=PASS disabled_clock=PASS failure_unwind=PASS reset=PASS\n";
}

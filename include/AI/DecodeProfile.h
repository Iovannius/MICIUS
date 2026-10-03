#pragma once
// D1 diagnostic only. No public ABI, allocation, atomics, or numerical work.
#include <cstdint>
#include <ostream>
#ifndef MICIUS_DECODE_PROFILE
#define MICIUS_DECODE_PROFILE 0
#endif
#if MICIUS_DECODE_PROFILE
#include <array>
#include <chrono>
#endif
namespace micius::d1 {
inline constexpr bool compiled = MICIUS_DECODE_PROFILE != 0;
enum class Part : unsigned { Forward, QKV, O, Gate, Up, Down, LM,
    PrepQKV, PrepO, PrepFFN, PrepDown, PrepLM, Count };
#if MICIUS_DECODE_PROFILE
class Scope;
struct Counter { uint64_t calls=0, inclusive_ns=0, exclusive_ns=0; };
struct State {
    bool enabled=false;
    Scope* top=nullptr;
    Part current=Part::QKV;
    std::array<Counter,static_cast<unsigned>(Part::Count)> counters{};
};
inline thread_local State state;
#ifdef MICIUS_D1_TEST_CLOCK
uint64_t TestNow() noexcept;
inline uint64_t Now() noexcept { return TestNow(); }
#else
inline uint64_t Now() noexcept {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count());
}
#endif
inline bool Begin(bool enabled) noexcept {
    if (state.enabled || state.top) return false;
    state=State{}; state.enabled=enabled; return true;
}
inline bool End() noexcept {
    if (state.top) return false;
    state.enabled=false; return true;
}
class Scope {
    bool active_;
    Part part_;
    uint64_t start_=0, child_ns_=0;
    Scope* parent_=nullptr;
public:
    explicit Scope(Part part) noexcept : active_(state.enabled), part_(part) {
        if (active_) { start_=Now(); parent_=state.top; state.top=this; }
    }
    ~Scope() {
        if (!active_) return;
        const uint64_t elapsed=Now()-start_;
        state.top=parent_;
        auto& c=state.counters[static_cast<unsigned>(part_)];
        ++c.calls; c.inclusive_ns+=elapsed; c.exclusive_ns+=elapsed-child_ns_;
        if (parent_) parent_->child_ns_+=elapsed;
    }
    Scope(const Scope&)=delete;
    Scope& operator=(const Scope&)=delete;
};
inline Part Current() noexcept { return state.current; }
inline void Tag(Part part) noexcept { if (state.enabled) state.current=part; }
inline Part Preparation() noexcept {
    switch (state.current) {
        case Part::QKV: return Part::PrepQKV;
        case Part::O: return Part::PrepO;
        case Part::Down: return Part::PrepDown;
        case Part::LM: return Part::PrepLM;
        default: return Part::PrepFFN;
    }
}
#else
inline bool Begin(bool enabled) noexcept { return !enabled; }
inline bool End() noexcept { return true; }
#endif
class Session {
    bool valid_, armed_;
public:
    explicit Session(bool enabled) noexcept : valid_(Begin(enabled)), armed_(valid_) {}
    bool valid() const noexcept { return valid_; }
    bool stop() noexcept {
        if (!armed_) return valid_;
        const bool ok=End(); if(ok) armed_=false; return ok;
    }
    ~Session() { if(armed_) End(); }
    Session(const Session&)=delete;
    Session& operator=(const Session&)=delete;
};
inline void Emit(std::ostream& out, const char* phase, uint64_t wall_ns,
                 uint32_t layers, uint64_t steps) {
#if MICIUS_DECODE_PROFILE
    const State saved=state;
    // Separate diagnostic, outside decode ROI. Never subtract from category times.
    constexpr uint64_t n=10000;
    Begin(true);
    const uint64_t begin=Now();
    for(uint64_t i=0;i<n;++i) { Scope empty(Part::PrepQKV); }
    const uint64_t probe=Now()-begin;
    End(); state=saved;
    constexpr const char* names[]={"forward","qkv","o","gate","up","down","lm_head",
        "prepare_qkv","prepare_o","prepare_ffn_shared","prepare_down","prepare_lm_head"};
    out << std::dec << "d1_profile={\"revision\":\"d1-2026-09-21\",\"phase\":\"" << phase
        << "\",\"layers\":" << layers << ",\"steps\":" << steps << ",\"wall_ns\":" << wall_ns
        << ",\"empty_scope_iterations\":" << n << ",\"empty_scope_total_ns\":" << probe
        << ",\"parts\":{";
    for(unsigned i=0;i<static_cast<unsigned>(Part::Count);++i) {
        if(i) out << ',';
        const auto& c=saved.counters[i];
        out << '"' << names[i] << "\":{\"calls\":" << c.calls
            << ",\"inclusive_ns\":" << c.inclusive_ns << ",\"exclusive_ns\":" << c.exclusive_ns << '}';
    }
    out << "}}\n";
#else
    (void)out; (void)phase; (void)wall_ns; (void)layers; (void)steps;
#endif
}
} // namespace micius::d1
#define MICIUS_D1_JOIN_(a,b) a##b
#define MICIUS_D1_JOIN(a,b) MICIUS_D1_JOIN_(a,b)
#if MICIUS_DECODE_PROFILE
#define MICIUS_D1_SCOPE(part) ::micius::d1::Scope MICIUS_D1_JOIN(d1_scope_,__LINE__)(part)
#define MICIUS_D1_TAG(part) ::micius::d1::Tag(::micius::d1::Part::part)
#else
#define MICIUS_D1_SCOPE(part) ((void)0)
#define MICIUS_D1_TAG(part) ((void)0)
#endif

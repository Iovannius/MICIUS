#pragma once
#include "AIComputKrnl.h"
#include <cstdint>
#include <utility>
#include <vector>

namespace micius::i4 {
enum class Policy { None=0, L31GU=1, BothFFN=2, AllF64=3 };
enum class Op { QKV, O, Gate, Up, Down, LM };
Policy CompiledPolicy() noexcept;
const char* PolicyName(Policy) noexcept;
bool Select(Policy, uint32_t layer, Op) noexcept;
Q8_32_GEMVStatus F64(const int8_t*,const QuantDescriptor*,const float*,float*,
                    uint32_t cols,uint32_t rows,bool validated) noexcept;

// Optional caller-thread route audit, owned by one evaluation step. No model
// state, activation cache, pointer-keyed cache, or persistent input is stored.
class RouteAudit final {
public:
    explicit RouteAudit(uint32_t layers) noexcept;
    ~RouteAudit();
    RouteAudit(const RouteAudit&)=delete;
    RouteAudit& operator=(const RouteAudit&)=delete;
    bool good() const noexcept { return good_; }
    bool finish() noexcept;
    uint32_t count() const noexcept { return static_cast<uint32_t>(seen_.size()); }
private:
    friend bool RecordSelected(uint32_t,Op) noexcept;
    bool good_=true,active_=false,finished_=false;
    uint32_t layers_=0;
    std::vector<std::pair<uint32_t,Op>> seen_;
};
bool RecordSelected(uint32_t,Op) noexcept;
} // namespace micius::i4

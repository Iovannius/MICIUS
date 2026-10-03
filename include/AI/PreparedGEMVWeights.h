#pragma once

#include "AIComputKrnl.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <vector>

struct MiciusContext;

namespace micius::internal {
class PreparedF64GEMVInput;

// PW2 private C++ representation. Exact raw signed integers stored as F32, never scaled.
// A bound-model snapshot; no activation, scale, pointer-key or token cache.
class PreparedF32GEMVWeights final {
public:
    PreparedF32GEMVWeights() noexcept = default;
    PreparedF32GEMVWeights(PreparedF32GEMVWeights&&) noexcept = default;
    PreparedF32GEMVWeights& operator=(PreparedF32GEMVWeights&&) noexcept = default;
    PreparedF32GEMVWeights(const PreparedF32GEMVWeights&) = delete;
    PreparedF32GEMVWeights& operator=(const PreparedF32GEMVWeights&) = delete;
    const float* data() const noexcept { return data_.get(); }
    uint32_t rows() const noexcept { return rows_; }
    uint32_t cols() const noexcept { return cols_; }
private:
    std::unique_ptr<float[]> data_;
    uint32_t rows_ = 0, cols_ = 0;
    friend struct WeightPreparationAccess;
};

struct WeightSource {
    std::span<const int8_t> raw;
    uint32_t rows, cols;
};
enum class WeightPrepareResult { READY, BUDGET, ALLOCATION, INVALID };

// All-or-nothing group. Failure clears out; no partially published matrices.
// Budget counts extra float payload only, not existing Q8, scales or allocator
// metadata. Preparation includes allocation and touching every expanded byte.
WeightPrepareResult PrepareF32WeightGroup(
    std::span<const WeightSource> sources, uint64_t budget_bytes,
    std::vector<PreparedF32GEMVWeights>& out, uint64_t& bytes) noexcept;

// Same global row-range / output visibility / MXCSR contract as prepared input.
// The owner and all operands must remain alive and read-only until completion.
Q8_32_GEMVStatus GEMV_F32WeightsPreparedRows(
    const PreparedF32GEMVWeights&, const QuantDescriptor*,
    const PreparedF64GEMVInput&, float*, uint32_t row_begin, uint32_t row_end) noexcept;
// Only use with model-owned V1 validated scale descriptors.
Q8_32_GEMVStatus GEMV_F32WeightsPreparedRowsPrevalidatedScales(
    const PreparedF32GEMVWeights&, const QuantDescriptor*,
    const PreparedF64GEMVInput&, float*, uint32_t row_begin, uint32_t row_end) noexcept;

struct PreparedWeightStats {
    uint32_t selection = 0; // 0: NONE, 1: packed QKV, 2: O; PW2 has no combined mode
    uint32_t matrices = 0;
    uint64_t budget_bytes = 0, extra_bytes = 0;
    double preparation_seconds = 0;
    const char* status = "disabled";
};
// Diagnostic observation, not an addition to the public C ABI.
PreparedWeightStats GetPreparedWeightStats(const MiciusContext*) noexcept;
} // namespace micius::internal

#include "PreparedGEMVWeights.h"
#include <limits>
#include <new>

namespace micius::internal {
struct WeightPreparationAccess {
    static void Build(const WeightSource& source, PreparedF32GEMVWeights& result) {
        // Default initialization avoids an unnecessary zero-fill pass.
        std::unique_ptr<float[]> data(new float[source.raw.size()]);
        for (size_t i = 0; i < source.raw.size(); ++i)
            data[i] = static_cast<float>(source.raw[i]);
        result.data_ = std::move(data);
        result.rows_ = source.rows;
        result.cols_ = source.cols;
    }
};

WeightPrepareResult PrepareF32WeightGroup(
    std::span<const WeightSource> sources, uint64_t budget_bytes,
    std::vector<PreparedF32GEMVWeights>& out, uint64_t& bytes) noexcept
{
    out.clear(); bytes = 0;
    uint64_t required = 0;
    if (sources.empty()) return WeightPrepareResult::INVALID;
    for (const auto& source : sources) {
        const uint64_t count = uint64_t{source.rows} * source.cols;
        if (!source.rows || !source.cols || !source.raw.data() ||
            count != source.raw.size() ||
            count > std::numeric_limits<size_t>::max() / sizeof(float) ||
            count > (std::numeric_limits<uint64_t>::max() - required) / sizeof(float))
            return WeightPrepareResult::INVALID;
        required += count * sizeof(float);
    }
    if (required > budget_bytes) return WeightPrepareResult::BUDGET;
    try {
        std::vector<PreparedF32GEMVWeights> candidate(sources.size());
        for (size_t i = 0; i < sources.size(); ++i)
            WeightPreparationAccess::Build(sources[i], candidate[i]);
        out.swap(candidate);
        bytes = required;
        return WeightPrepareResult::READY;
    } catch (const std::bad_alloc&) {
        return WeightPrepareResult::ALLOCATION;
    }
}
} // namespace micius::internal

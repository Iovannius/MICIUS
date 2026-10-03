#include "app/BenchmarkIttRoi.h"
#include "app/MiciusShell.h"

#include <cassert>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#if defined(NDEBUG)
#error "benchmark_itt_roi_tests requires active assert checks"
#endif

namespace {
std::vector<char> events;
bool collecting = false;
bool available = true;
struct RecordingControl {
    static const char* Prepare() {
        Pause();
        return available ? nullptr : "collector unavailable";
    }
    static void Pause() { events.push_back('P'); collecting = false; }
    static void Resume() { events.push_back('R'); collecting = true; }
};
using RecordedRoi = micius::benchmark::CollectionRoi<RecordingControl>;
}

int main()
{
    // Lifecycle checks use a recording control, not a simulated PMU result.
    {
        RecordedRoi roi;
        assert(roi.Prepare(false) == nullptr);
        roi.Begin();
        roi.End();
    }
    assert(events.empty()); // Ordinary benchmark must not control a profiler.
    {
        RecordedRoi roi;
        assert(roi.Prepare(true) == nullptr);
        assert(!collecting); // Load and warmup stay paused.
        roi.Begin();
        assert(collecting);
        roi.Begin(); // No second resume while already active.
        roi.End();
        assert(!collecting); // Logits copying/hashing follows End.
        roi.End();
    }
    assert((events == std::vector<char>{'P', 'R', 'P'}));
    events.clear();
    try {
        RecordedRoi roi;
        assert(roi.Prepare(true) == nullptr);
        roi.Begin();
        throw 1;
    } catch (int) {}
    assert(!collecting);
    assert((events == std::vector<char>{'P', 'R', 'P'}));
    events.clear();
    available = false;
    {
        RecordedRoi roi;
        assert(roi.Prepare(true) != nullptr);
        roi.Begin();
    }
    assert((events == std::vector<char>{'P'})); // Failed init never resumes.

#if defined(MICIUS_BENCHMARK_ITT)
    // CTest runs without a collector. Check the real SDK initialization state;
    // its no-op dispatch pointers can remain non-null after failure.
    micius::benchmark::IttRoi actual;
    assert(actual.Prepare(true) != nullptr);
    assert(__itt_get_collection_state() != __itt_collection_init_successful);
#endif

    // Exercise the real CLI parser: unavailable ITT must fail before loading.
    MiciusShell shell;
    std::ostringstream output;
    auto* saved_output = std::cout.rdbuf(output.rdbuf());
    const int result = shell.RunModelBenchmark({
        "--itt-roi", "--model", "must-not-be-loaded.gguf",
        "--warmup", "0", "--steps", "1", "--token", "1"});
    std::cout.rdbuf(saved_output);
    assert(result != 0);
    assert(output.str().find("status=FAIL") != std::string::npos);
    assert(output.str().find("[Loader]") == std::string::npos);
#if defined(MICIUS_BENCHMARK_ITT)
    assert(output.str().find("ITT collector initialization/control unavailable") != std::string::npos);
#else
    assert(output.str().find("MICIUS_BENCHMARK_ITT=ON") != std::string::npos);
#endif
    std::cout << "benchmark_itt_roi_tests: PASS (collector hardware NOT TESTED)\n";
}

#pragma once

#if defined(MICIUS_BENCHMARK_ITT)
#include <ittnotify.h>
#endif

namespace micius::benchmark {

struct IttControl {
    static const char* Prepare() noexcept {
#if defined(MICIUS_BENCHMARK_ITT)
        // Non-null dispatch pointers may still be SDK no-op stubs.
        __itt_pause();
        if (__itt_get_collection_state() != __itt_collection_init_successful ||
            !__itt_pause_ptr || !__itt_resume_ptr) {
            return "ITT collector initialization/control unavailable; launch with VTune -start-paused";
        }
        return nullptr;
#else
        return "--itt-roi requires MICIUS_BENCHMARK_ITT=ON";
#endif
    }
    static void Resume() noexcept {
#if defined(MICIUS_BENCHMARK_ITT)
        __itt_resume();
#endif
    }
    static void Pause() noexcept {
#if defined(MICIUS_BENCHMARK_ITT)
        __itt_pause();
#endif
    }
};

// One process-wide sampling interval, never one interval per token or GEMV.
// Prepare before loading; Begin/End bracket the existing timed decode loop.
// The control type lets lifecycle tests run without a hardware collector.
template<class Control>
class CollectionRoi final {
public:
    CollectionRoi() = default;
    CollectionRoi(const CollectionRoi&) = delete;
    CollectionRoi& operator=(const CollectionRoi&) = delete;
    ~CollectionRoi() { End(); }

    const char* Prepare(bool requested) noexcept {
        if (!requested) return nullptr;
        if (const char* error = Control::Prepare()) return error;
        enabled_ = true;
        return nullptr;
    }

    void Begin() noexcept {
        if (enabled_ && !active_) {
            Control::Resume();
            active_ = true;
        }
    }

    void End() noexcept {
        if (active_) {
            Control::Pause();
            active_ = false;
        }
    }

private:
    bool enabled_ = false;
    bool active_ = false;
};

using IttRoi = CollectionRoi<IttControl>;

} // namespace micius::benchmark

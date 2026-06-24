# MICIUS: Multicore Inference Concurrency for Integrated UMA Systems

![ISA](https://img.shields.io/badge/ISA-x86__64-002f6c)
![OS](https://img.shields.io/badge/OS-Cross--Platform-green)
![Compiler](https://img.shields.io/badge/Compiler-clang%2B%2B-red)
![Intrinsic](https://img.shields.io/badge/Intrinsic-AVX2%20%7C%20AVX--512%20%7C%20VNNI-orange)
![Execution](https://img.shields.io/badge/Execution-CPU_Native-yellow)
![UMA](https://img.shields.io/badge/Memory-UMA_Optimized-blueviolet)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**MICIUS** is a CPU-native, low-latency speculative decoding runtime written in modern C++ with explicit AVX2 intrinsics. It is designed to maximize inference throughput on x86 multicore architectures by aggressively minimizing global synchronization, branch mispredictions, and thread migration overhead.

Unlike general-purpose inference frameworks, MICIUS uses a highly specialized worker array topology and lock-free memory structures to keep internal spin times under 10µs, ensuring the engine remains memory-bandwidth and compute-bound rather than scheduler-bound.

## Architecture Highlights (v0.5)

### 1. Lock-Free Concurrent Data Structures

* **Treiber Stack KV Allocator:** The `KVCacheRingBuffer` relies on a 64-bit versioned atomic head (`free_head`) to solve the ABA problem lock-free. Paired with a Thread-Local Allocation Buffer (TLAB) bulk-grab mechanism, it highly reduces atomic CAS bus contention during high-frequency speculative branch spawns/rejects.
* **Chase-Lev Work Stealing:** Worker threads operate on a custom Chase-Lev deque (L1-aligned). They push/pop tasks to their bottom end, and steal from other queues via CAS on the top end. Victim selection uses a fast inline Xorshift32 PRNG to scatter stealing requests and avoid $O(N^2)$ cache-line broadcast storms.
* **AoSoA Draft Proposals:** Tree state is mapped into a `DraftNodeBlock8` Array of Structures of Arrays (AoSoA) layout. Each 8-node block is strictly 128-byte aligned (two cache lines for most modern x86 microarchitecture). Concurrent node spawning uses a single `LOCK XADD` (`fetch_add`), reducing global tree contention to $O(1)$ without standard mutexes.

### 2. SIMD Micro-Kernels (Pure AVX2)

* **VNNI-less INT8 GEMV:** Provides a pure AVX2 `DraftGEMV_INT8` implementation for older hardware lacking AVX-VNNI. It achieves maximum execution port utilization by pre-aligning workspace bytes and concurrently issuing `_mm256_abs_epi8`, `_mm256_sign_epi8`, and `_mm256_maddubs_epi16` to safely saturate limits at the `INT16_MIN` boundary.
* **Branchless Search & Top-K:** Replaces scalar branch-prediction bottlenecks with full vector-domain cascading filters. Both `BranchlessArgMax_AVX2` and `BranchlessTop3_AVX2` utilize `_mm256_blendv_ps` and `_mm256_blendv_epi8` to perform cross-lane reductions and element shifting entirely within the YMM registers.
* **L1D Cache Rectification:** Utilizes 4-way loop unrolling (`LocalMicroCompactKV`) with continuous `_mm256_loadu_ps / _mm256_loadu_si256` parallel streams to reorganize fragmented KV chunks locally, leveraging deep branch prediction optimizations for dense array compaction.

### 3. Topology-Aware Scheduling

* **OS-Bypass Thread Pinning:** Implements aggressive thread affinity mapping (via `pthread_setaffinity_np` or `SetThreadAffinityMask`).
* **Adaptive Deep Spinning:** Replaces OS-level `yield()` or sleep states with an adaptive `_mm_pause` escalation mechanism (up to 4096 pauses). This stabilizes user-mode execution rights and prevents context-switch penalties during microsecond-level synchronization waits.

### 4. Memory Pool & C-ABI State Machine

* **HeteroMemPool:** Explicit OS-level page locking. Attempts to map HugeTLB pages (`MAP_HUGETLB` / `VirtualAlloc` with large pages) and explicitly pins them to RAM (`mlock` / `VirtualLock`) to eliminate page faults during decoding.
* **PumpSingleToken Machine:** Removes legacy monolithic inference loops. The entire forward pass and garbage collection (GC) pipeline is represented as an event-driven `PumpResult` state machine. This allows seamless integration into FFI layers (Python, Rust, Go) via opaque pointers without exposing unsafe internal threading logic.
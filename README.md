# MICIUS: Multicore Inference Concurrency for Integrated UMA Systems

![ISA](https://img.shields.io/badge/ISA-x86__64-002f6c)
![OS](https://img.shields.io/badge/OS-Cross--Platform-green)
![Compiler](https://img.shields.io/badge/Compiler-clang%2B%2B-red)
![Intrinsic](https://img.shields.io/badge/Intrinsic-AVX2%20%7C%20AVX--512-orange)
![Execution](https://img.shields.io/badge/Execution-CPU_Native-yellow)
![UMA](https://img.shields.io/badge/Memory-UMA_Optimized-blueviolet)

[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**MICIUS** is a bare-metal, pure-asynchronous speculative decoding middleware for AI workloads, designed to overcome the memory bandwidth wall on **Unified Memory Architecture (UMA)** systems (e.g. Intel® Arc™ graphics, AMD Radeon™ 780M).
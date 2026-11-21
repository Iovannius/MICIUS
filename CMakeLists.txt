cmake_minimum_required(VERSION 3.16)

project(MICIUS VERSION 0.7.3 LANGUAGES CXX)

include(CTest)
include(GNUInstallDirs)
include(CMakePackageConfigHelpers)
option(MICIUS_PARSER_LLAMA "Build the linkable Llama GGUF parser" ON)
option(MICIUS_PARSER_QWEN2 "Build the linkable Qwen2 GGUF parser" ON)
option(MICIUS_PARSER_QWEN3 "Build the linkable dense Qwen3 GGUF parser" ON)
option(MICIUS_ENABLE_TEXT "Build the UTF-8 tokenizer and textual chat support" ON)
include(CheckCXXCompilerFlag)
include(CheckIPOSupported)

# The rebuilt runtime uses std::construct_at/std::destroy_at and therefore
# cannot be compiled as C++17 without changing source-level lifetime handling.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Release profiles set coherent defaults on a fresh build tree. CUSTOM retains
# the low-level switches for reproducing historical experiments.
set(_default_backend INT8)
if(MSVC OR NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    set(_default_backend F64)
endif()
set(MICIUS_BACKEND "${_default_backend}" CACHE STRING "Numerical backend: INT8, F64, or CUSTOM")
set_property(CACHE MICIUS_BACKEND PROPERTY STRINGS INT8 F64 CUSTOM)
if(NOT MICIUS_BACKEND MATCHES "^(INT8|F64|CUSTOM)$")
    message(FATAL_ERROR "MICIUS_BACKEND must be INT8, F64, or CUSTOM")
endif()
set(_release_simd OFF)
set(_int8 OFF)
set(_f64 OFF)
set(_w8a8 OFF)
if(MICIUS_BACKEND STREQUAL "INT8")
    set(_release_simd ON)
    set(_int8 ON)
    set(_w8a8 AVX2)
elseif(MICIUS_BACKEND STREQUAL "F64")
    set(_release_simd ON)
    set(_f64 ON)
endif()

# Keep the original cache-variable interface used by existing build scripts.
set(ARCH "native" CACHE STRING
    "Target architecture optimization (e.g. native, haswell, znver3)")
set(ALIGN "32" CACHE STRING "Loop alignment boundary in bytes")

option(MICIUS_ENABLE_LEGACY_TUNING
       "Enable the original O3/debug/native/unroll/alignment tuning profile" ON)
option(MICIUS_ENABLE_FAST_MATH
       "Enable the original -ffast-math behavior (weakens model-data validation)" OFF)
option(MICIUS_ENABLE_LTO
       "Enable ThinLTO with Clang or compiler-supported IPO elsewhere" ON)
option(MICIUS_USE_LLD
       "Use the original LLD link path when an LLD executable is available" ON)
option(MICIUS_STATIC_RUNTIME
       "Link the CLI and sandbox compiler runtimes statically where supported" ON)
option(MICIUS_BUILD_CLI "Build the micius_cli executable" ON)
option(MICIUS_BUILD_SANDBOX "Build the reference ABI stress sandbox" ON)
option(MICIUS_BENCHMARK_ITT
       "Enable explicit --itt-roi collection control for the real-model benchmark" OFF)
set(MICIUS_ITT_ROOT "" CACHE PATH
    "ITT SDK root containing include/ittnotify.h and src/ittnotify/ittnotify_static.c")
option(MICIUS_REFERENCE_GEMV_AVX2
       "Route reference-mode GEMV calls through the checked Q8_32 AVX2 backend" ${_release_simd})
option(MICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2
       "Use the two-row shared-input/direct-weight-load Q8_32 AVX2 backend" ${_release_simd})
option(MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES
       "Use the prevalidated GEMV entry only for models owned by the private V1 binding boundary" ${_release_simd})
option(MICIUS_REFERENCE_GEMV_PREPARED_F64
       "Prepare one validated F64 activation snapshot for reference row-pair GEMV" ${_f64})
set(MICIUS_PW2_SELECTION "NONE" CACHE STRING "PW2 experimental weight snapshot: NONE, QKV, O")
set_property(CACHE MICIUS_PW2_SELECTION PROPERTY STRINGS NONE QKV O)
set(MICIUS_PW2_BUDGET_MIB "512" CACHE STRING "Maximum extra raw F32 weight payload per binding")
set(MICIUS_I1_BACKEND "${_w8a8}" CACHE STRING "I1 approximate W8A8: OFF, SCALAR, AVX2")
set_property(CACHE MICIUS_I1_BACKEND PROPERTY STRINGS OFF SCALAR AVX2)
option(MICIUS_I2_DIAGNOSTICS "Private same-input INT8 diagnosis; never use for timing" OFF)
option(MICIUS_I3_INTERVENTION "Private controlled FFN precision experiment; never use for timing" OFF)
option(MICIUS_I4_EVALUATION "Private fixed-token teacher-forced NLL evaluator" OFF)
option(MICIUS_I6_BENCHMARK "Private fixed-natural-input Windows performance evaluator" OFF)
option(MICIUS_I10_VALUE64 "Order-preserving AVX2 Value64" ${_int8})
option(MICIUS_I11_SCORE64 "Ordered chronological Score64" ${_int8})
option(MICIUS_I13_GEMV4 "Exact four-row W8A8 AVX2 consumption" ${_int8})
option(MICIUS_I15_VALUE_TILE32 "Exact tiled Value64" ${_int8})
option(MICIUS_I17_SCORE_REUSE "Reuse exact Score64 query conversion" ${_int8})
option(MICIUS_I19_SCORE_GROUP3 "Reuse Key data across three query heads" ${_int8})
option(MICIUS_I22_RELEASE_LOADER_SCALES "Release copied loader descriptors after successful bind" ${_int8})
option(MICIUS_I25_PANEL_GEMV "I25 private four-row W8A8 panels" ${_int8})
set(MICIUS_I4_POLICY "NONE" CACHE STRING "Private continuous-context precision policy: NONE, L31_GU, L3_L31_FFN, F64_ALL")
set_property(CACHE MICIUS_I4_POLICY PROPERTY STRINGS NONE L31_GU L3_L31_FFN F64_ALL)
if(MICIUS_I4_POLICY STREQUAL "NONE")
    set(MICIUS_I4_MODE 0)
elseif(MICIUS_I4_POLICY STREQUAL "L31_GU")
    set(MICIUS_I4_MODE 1)
elseif(MICIUS_I4_POLICY STREQUAL "L3_L31_FFN")
    set(MICIUS_I4_MODE 2)
elseif(MICIUS_I4_POLICY STREQUAL "F64_ALL")
    set(MICIUS_I4_MODE 3)
else()
    message(FATAL_ERROR "Unknown I4 precision policy")
endif()
if(MICIUS_I1_BACKEND STREQUAL "OFF")
    set(MICIUS_I1_MODE 0)
elseif(MICIUS_I1_BACKEND STREQUAL "SCALAR")
    set(MICIUS_I1_MODE 1)
elseif(MICIUS_I1_BACKEND STREQUAL "AVX2")
    set(MICIUS_I1_MODE 2)
else()
    message(FATAL_ERROR "I1 backend must be OFF, SCALAR or AVX2")
endif()
if(NOT MICIUS_PW2_BUDGET_MIB MATCHES "^(0|[1-9][0-9]*)$" OR MICIUS_PW2_BUDGET_MIB GREATER 1048576)
    message(FATAL_ERROR "PW2 budget must be an integer in [0,1048576] MiB")
endif()
if(MICIUS_PW2_SELECTION STREQUAL "NONE")
    set(MICIUS_PW2_MODE 0)
elseif(MICIUS_PW2_SELECTION STREQUAL "QKV")
    set(MICIUS_PW2_MODE 1)
elseif(MICIUS_PW2_SELECTION STREQUAL "O")
    set(MICIUS_PW2_MODE 2)
else()
    message(FATAL_ERROR "PW2 selection must be NONE, QKV or O")
endif()
option(MICIUS_REFERENCE_GEMV_COMPACT_SCALES
       "Store one private F32 scale per validated Q8_32 block for reference GEMV" OFF)
option(MICIUS_REFERENCE_ATTENTION_VALUE_AVX2
       "Route reference attention V aggregation through the order-preserving AVX2 backend" ${_release_simd})
option(MICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE
       "Convert each reference attention query head to exact F64 once before its key scan" OFF)
option(MICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD
       "Evaluate four ordered reference attention key rows while sharing each query conversion" ${_release_simd})
# A release build must never silently become a historical ablation through
# stale cache entries. Switch profiles in separate build directories.
if(_release_simd AND (NOT MICIUS_REFERENCE_GEMV_AVX2 OR NOT MICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2
   OR NOT MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES OR NOT MICIUS_REFERENCE_ATTENTION_VALUE_AVX2
   OR NOT MICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD OR MICIUS_REFERENCE_GEMV_COMPACT_SCALES
   OR MICIUS_ENABLE_FAST_MATH OR NOT MICIUS_PW2_SELECTION STREQUAL "NONE"
   OR MICIUS_I2_DIAGNOSTICS OR MICIUS_I3_INTERVENTION OR MICIUS_I4_MODE
   OR MICIUS_DECODE_PROFILE OR MICIUS_BENCHMARK_ITT))
    message(FATAL_ERROR "Release profile and cached options disagree. Use a fresh build directory or MICIUS_BACKEND=CUSTOM.")
endif()
if(_int8 AND (NOT MICIUS_I1_MODE EQUAL 2 OR NOT MICIUS_I25_PANEL_GEMV OR NOT MICIUS_I22_RELEASE_LOADER_SCALES
   OR NOT MICIUS_I13_GEMV4 OR NOT MICIUS_I15_VALUE_TILE32 OR NOT MICIUS_I17_SCORE_REUSE OR NOT MICIUS_I19_SCORE_GROUP3))
    message(FATAL_ERROR "The INT8 release requires the complete I25 configuration.")
endif()
if(_f64 AND (MICIUS_I1_MODE OR NOT MICIUS_REFERENCE_GEMV_PREPARED_F64))
    message(FATAL_ERROR "The F64 release requires prepared F64 inputs and W8A8 OFF.")
endif()
message(STATUS "MICIUS ${PROJECT_VERSION}: ${MICIUS_BACKEND} backend")

if(MICIUS_I1_MODE AND (MICIUS_REFERENCE_GEMV_PREPARED_F64 OR
    NOT MICIUS_REFERENCE_GEMV_AVX2 OR NOT MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES OR
    MICIUS_REFERENCE_GEMV_COMPACT_SCALES OR MICIUS_ENABLE_FAST_MATH OR
    NOT MICIUS_PW2_SELECTION STREQUAL "NONE" OR MICIUS_DECODE_PROFILE OR MICIUS_BENCHMARK_ITT))
    message(FATAL_ERROR "I1 requires strict AVX2/prevalidated descriptor scales; prepared-F64, PW2, compact, D1 and ITT OFF")
endif()
if(MICIUS_I2_DIAGNOSTICS AND (NOT MICIUS_REFERENCE_GEMV_AVX2 OR
   NOT MICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2 OR NOT MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES OR
   MICIUS_REFERENCE_GEMV_COMPACT_SCALES OR MICIUS_ENABLE_FAST_MATH OR
   NOT MICIUS_PW2_SELECTION STREQUAL "NONE" OR MICIUS_DECODE_PROFILE OR MICIUS_BENCHMARK_ITT OR
   (NOT MICIUS_I1_MODE AND NOT MICIUS_REFERENCE_GEMV_PREPARED_F64)))
    message(FATAL_ERROR "I2 requires the accepted F64 or I1 configuration; no PW2, fast-math, compact, D1 or ITT")
endif()
if(MICIUS_I3_INTERVENTION AND (NOT MICIUS_I1_BACKEND STREQUAL "AVX2" OR
   NOT MICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2 OR MICIUS_I2_DIAGNOSTICS))
    message(FATAL_ERROR "I3 requires strict I1 AVX2/row-pair configuration and I2 diagnostics OFF")
endif()
if(MICIUS_I4_MODE AND (NOT MICIUS_I1_BACKEND STREQUAL "AVX2" OR
   NOT MICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2 OR MICIUS_I2_DIAGNOSTICS OR MICIUS_I3_INTERVENTION))
    message(FATAL_ERROR "I4 mixed precision requires strict I1 AVX2/row-pair; I2 and I3 OFF")
endif()
if((MICIUS_I4_EVALUATION OR MICIUS_I6_BENCHMARK) AND (MICIUS_I2_DIAGNOSTICS OR MICIUS_I3_INTERVENTION OR
   NOT MICIUS_REFERENCE_GEMV_AVX2 OR NOT MICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2 OR
   NOT MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES OR MICIUS_REFERENCE_GEMV_COMPACT_SCALES OR
   MICIUS_ENABLE_FAST_MATH OR NOT MICIUS_PW2_SELECTION STREQUAL "NONE" OR
   MICIUS_DECODE_PROFILE OR MICIUS_BENCHMARK_ITT OR MICIUS_I1_MODE EQUAL 1 OR
   (NOT MICIUS_I1_MODE AND NOT MICIUS_REFERENCE_GEMV_PREPARED_F64)))
    message(FATAL_ERROR "I4 evaluation requires the accepted strict F64 or I1 AVX2 configuration and no other instrumentation")
endif()

if(MICIUS_I6_BENCHMARK AND (NOT WIN32 OR MICIUS_I4_EVALUATION OR MICIUS_I4_MODE))
    message(FATAL_ERROR "I6 requires Windows and uninstrumented F64/I1; I4 evaluation and mixed policy OFF")
endif()

if(MICIUS_PW2_MODE AND (NOT MICIUS_REFERENCE_GEMV_PREPARED_F64 OR
   NOT MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES OR MICIUS_REFERENCE_GEMV_COMPACT_SCALES))
    message(FATAL_ERROR "PW2 requires r1 prepared input and V1 descriptor scales; compact OFF")
endif()

if(MICIUS_REFERENCE_GEMV_PREPARED_F64 AND
   (NOT MICIUS_REFERENCE_GEMV_AVX2 OR NOT MICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2))
    message(FATAL_ERROR "MICIUS_REFERENCE_GEMV_PREPARED_F64 requires AVX2 and ROW_PAIR_AVX2")
endif()
if(MICIUS_REFERENCE_GEMV_PREPARED_F64 AND MICIUS_ENABLE_FAST_MATH)
    message(FATAL_ERROR "MICIUS_REFERENCE_GEMV_PREPARED_F64 requires strict finite validation")
endif()

if(MICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2 AND
   NOT MICIUS_REFERENCE_GEMV_AVX2)
    message(FATAL_ERROR
        "MICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2 requires "
        "MICIUS_REFERENCE_GEMV_AVX2=ON")
endif()

if(MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES AND
   NOT MICIUS_REFERENCE_GEMV_AVX2)
    message(FATAL_ERROR
        "MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES requires "
        "MICIUS_REFERENCE_GEMV_AVX2=ON")
endif()

if(MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES AND
   MICIUS_ENABLE_FAST_MATH)
    message(FATAL_ERROR
        "MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES cannot be combined with "
        "MICIUS_ENABLE_FAST_MATH: prevalidated model state requires reliable "
        "non-finite checks at the V1 binding boundary")
endif()

if(MICIUS_REFERENCE_GEMV_COMPACT_SCALES AND
   NOT MICIUS_REFERENCE_GEMV_AVX2)
    message(FATAL_ERROR
        "MICIUS_REFERENCE_GEMV_COMPACT_SCALES requires "
        "MICIUS_REFERENCE_GEMV_AVX2=ON")
endif()

if(MICIUS_REFERENCE_GEMV_COMPACT_SCALES AND
   NOT MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES)
    message(FATAL_ERROR
        "MICIUS_REFERENCE_GEMV_COMPACT_SCALES requires "
        "MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES=ON")
endif()

if(MICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE AND
   MICIUS_ENABLE_FAST_MATH)
    message(FATAL_ERROR
        "MICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE cannot be combined with "
        "MICIUS_ENABLE_FAST_MATH: its parity contract requires the ordered "
        "reference attention dot-product accumulation")
endif()

if(MICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD AND
   MICIUS_ENABLE_FAST_MATH)
    message(FATAL_ERROR
        "MICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD cannot be combined with "
        "MICIUS_ENABLE_FAST_MATH: its parity contract requires the ordered "
        "reference attention dot-product accumulation")
endif()

if(MICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD AND
   MICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE)
    message(FATAL_ERROR
        "MICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD and "
        "MICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE are mutually exclusive "
        "reference attention score experiments")
endif()

set(MICIUS_SANDBOX_SOURCE
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/stress/AIEmuSandbox.cpp"
    CACHE FILEPATH "Path to the AIEmuSandbox.cpp source")

if(NOT ARCH MATCHES "^[A-Za-z0-9_.+:-]+$")
    message(FATAL_ERROR "ARCH contains characters that cannot form one compiler argument: ${ARCH}")
endif()

if(NOT ALIGN MATCHES "^[1-9][0-9]*$")
    message(FATAL_ERROR "ALIGN must be a positive integer, got: ${ALIGN}")
endif()

# Preserve the original bin/lib output layout for single- and multi-config
# generators. Shared-library and PDB directories are defined for completeness.
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_PDB_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

foreach(config DEBUG RELEASE RELWITHDEBINFO MINSIZEREL)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_${config} "${CMAKE_BINARY_DIR}/bin")
    set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_${config} "${CMAKE_BINARY_DIR}/lib")
    set(CMAKE_LIBRARY_OUTPUT_DIRECTORY_${config} "${CMAKE_BINARY_DIR}/lib")
    set(CMAKE_PDB_OUTPUT_DIRECTORY_${config} "${CMAKE_BINARY_DIR}/bin")
endforeach()

find_package(Threads REQUIRED)

add_library(micius_build_options INTERFACE)
target_compile_features(micius_build_options INTERFACE cxx_std_20)

if(MSVC)
    target_compile_options(micius_build_options INTERFACE
        /W4
        /permissive-
        /arch:AVX2
    )

    if(MICIUS_ENABLE_LEGACY_TUNING)
        target_compile_options(micius_build_options INTERFACE
            /O2
            /Zi
            /Oy-
        )
    endif()

    if(MICIUS_ENABLE_FAST_MATH)
        target_compile_options(micius_build_options INTERFACE /fp:fast)
        message(WARNING
            "MICIUS_ENABLE_FAST_MATH is ON: non-finite tensor validation is not guaranteed")
    endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    string(MAKE_C_IDENTIFIER "${ARCH}" MICIUS_ARCH_CACHE_SUFFIX)
    set(MICIUS_ARCH_CHECK_VAR "MICIUS_ARCH_FLAG_SUPPORTED_${MICIUS_ARCH_CACHE_SUFFIX}")
    check_cxx_compiler_flag("-march=${ARCH}" ${MICIUS_ARCH_CHECK_VAR})
    if(NOT ${MICIUS_ARCH_CHECK_VAR})
        message(FATAL_ERROR
            "The selected compiler does not support ARCH='${ARCH}' (-march=${ARCH})")
    endif()

    target_compile_options(micius_build_options INTERFACE
        -mavx2
        -mfma
        -Wall
        -Wextra
        -Wpedantic
        -Wconversion
        -Wshadow
    )

    if(MICIUS_ENABLE_LEGACY_TUNING)
        target_compile_options(micius_build_options INTERFACE
            -O3
            -g
            -fno-omit-frame-pointer
            "-march=${ARCH}"
            -funroll-loops
            -falign-functions=32
            "-falign-loops=${ALIGN}"
            -mprefer-vector-width=256
            -fno-stack-protector
        )
    endif()

    if(MICIUS_ENABLE_FAST_MATH)
        target_compile_options(micius_build_options INTERFACE -ffast-math)
        message(WARNING
            "MICIUS_ENABLE_FAST_MATH is ON: non-finite tensor validation is not guaranteed")
    endif()

    if(_int8)
        # Keep code generation and LTO within the qualified no-VNNI contract
        # even when -march=native would enable additional dot instructions.
        foreach(feature avxvnni avx512vnni avxvnniint8 avxvnniint16)
            check_cxx_compiler_flag("-mno-${feature}" MICIUS_HAS_DISABLE_${feature})
            if(MICIUS_HAS_DISABLE_${feature})
                target_compile_options(micius_build_options INTERFACE "-mno-${feature}")
                target_link_options(micius_build_options INTERFACE "-mno-${feature}")
            endif()
        endforeach()
    endif()

    # Retain the original Windows/Clang CodeView behavior.
    if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(micius_build_options INTERFACE -gcodeview)
    endif()
else()
    message(WARNING
        "Compiler '${CMAKE_CXX_COMPILER_ID}' has no MICIUS tuning profile; "
        "only the required C++20 feature level will be applied")
endif()

# Explicit runtime sources. File loading, architecture modules and the CLI are separate archives/targets.
set(ALL_SRC_FILES
    src/micius_runtime.cpp
    src/app/BackendInfo.cpp
    src/AI/AIComputKrnl.cpp
    src/AI/AICoreEngine.cpp
    src/AI/SessionState.cpp
    src/AI/AIDataFront.cpp
    src/AI/HardwareTopology.cpp
    src/AI/I10AttentionValue.cpp
    src/AI/I11AttentionScore.cpp
    src/AI/I13W8A8GEMV.cpp
    src/AI/I15AttentionValue.cpp
    src/AI/I17AttentionScore.cpp
    src/AI/I19AttentionScore.cpp
    src/AI/i25/PanelGEMV.cpp
    src/AI/I2Diagnostic.cpp
    src/AI/I3Intervention.cpp
    src/AI/I4MixedPrecision.cpp
    src/AI/LoggerCore.cpp
    src/AI/PreparedGEMVWeights.cpp
    src/AI/ReferenceAttentionAVX2.cpp
    src/AI/ReferenceF32WeightGEMVAVX2.cpp
    src/AI/ReferenceGEMVAVX2.cpp
    src/AI/ReferenceW8A8GEMVAVX2.cpp
    src/app/I4Evaluation.cpp
)

if(MICIUS_I10_VALUE64)
    if(NOT MICIUS_I1_MODE EQUAL 2 OR NOT MICIUS_REFERENCE_ATTENTION_VALUE_AVX2 OR
       MICIUS_I2_DIAGNOSTICS OR MICIUS_I3_INTERVENTION OR MICIUS_I4_EVALUATION OR MICIUS_I4_MODE)
        message(FATAL_ERROR "I10 requires uninstrumented I1 AVX2 and original Value AVX2 fallback")
    endif()
else()
    list(FILTER ALL_SRC_FILES EXCLUDE REGEX "I10AttentionValue\\.cpp$")
endif()

if(MICIUS_I11_SCORE64)
    if(NOT MICIUS_I10_VALUE64 OR NOT MICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD)
        message(FATAL_ERROR "I11 requires accepted I10 Value64 and original TimeQuad reference")
    endif()
else()
    list(FILTER ALL_SRC_FILES EXCLUDE REGEX "I11AttentionScore\\.cpp$")
endif()

if(MICIUS_I13_GEMV4)
    if(NOT MICIUS_I11_SCORE64 OR NOT MICIUS_I1_MODE EQUAL 2 OR MICIUS_DECODE_PROFILE)
        message(FATAL_ERROR "I13 requires uninstrumented accepted I11 AVX2")
    endif()
else()
    list(FILTER ALL_SRC_FILES EXCLUDE REGEX "I13W8A8GEMV\\.cpp$")
endif()

if(MICIUS_I15_VALUE_TILE32)
    if(NOT MICIUS_I13_GEMV4 OR NOT MICIUS_I10_VALUE64 OR MICIUS_DECODE_PROFILE)
        message(FATAL_ERROR "I15 requires uninstrumented accepted I13")
    endif()
else()
    list(FILTER ALL_SRC_FILES EXCLUDE REGEX "I15AttentionValue\\.cpp$")
endif()

if(MICIUS_I17_SCORE_REUSE)
    if(NOT MICIUS_I15_VALUE_TILE32 OR NOT MICIUS_I11_SCORE64 OR MICIUS_DECODE_PROFILE)
        message(FATAL_ERROR "I17 requires uninstrumented accepted I15")
    endif()
else()
    list(FILTER ALL_SRC_FILES EXCLUDE REGEX "I17AttentionScore\\.cpp$")
endif()

if(MICIUS_I19_SCORE_GROUP3)
    if(NOT MICIUS_I17_SCORE_REUSE OR MICIUS_DECODE_PROFILE)
        message(FATAL_ERROR "I19 requires uninstrumented accepted I17")
    endif()
else()
    list(FILTER ALL_SRC_FILES EXCLUDE REGEX "I19AttentionScore\\.cpp$")
endif()

set(MICIUS_CORE_SOURCES ${ALL_SRC_FILES})
if(NOT MICIUS_I2_DIAGNOSTICS)
    list(FILTER MICIUS_CORE_SOURCES EXCLUDE REGEX "I2Diagnostic\\.cpp$")
endif()
if(NOT MICIUS_I3_INTERVENTION)
    list(FILTER MICIUS_CORE_SOURCES EXCLUDE REGEX "I3Intervention\\.cpp$")
endif()
if(NOT MICIUS_I4_EVALUATION)
    list(FILTER MICIUS_CORE_SOURCES EXCLUDE REGEX "I4Evaluation\\.cpp$")
endif()
if(NOT MICIUS_I4_MODE AND NOT MICIUS_I4_EVALUATION)
    list(FILTER MICIUS_CORE_SOURCES EXCLUDE REGEX "I4MixedPrecision\\.cpp$")
endif()

if(NOT MICIUS_CORE_SOURCES)
    message(FATAL_ERROR "No MICIUS core sources were found below src/")
endif()

# Isolate the reference GEMV FP contract from legacy draft arithmetic. Both
# prepared OFF/ON compile the same no-contraction baseline. The frozen oracle
# uses separate rounding too, including scalar tails. Strict options are source
# options and therefore follow target/global optimization options.
set(MICIUS_GEMV_STRICT_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/src/app/I6Benchmark.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/AI/I4MixedPrecision.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/app/I4Evaluation.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/i4_evaluation_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/AI/I3Intervention.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/i3_intervention_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/AI/I2Diagnostic.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/i2_diagnostic_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/AI/ReferenceW8A8GEMVAVX2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/AI/I13W8A8GEMV.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/AI/i25/PanelGEMV.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/i25_panel_contract.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/i25_ownership_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/AI/I15AttentionValue.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/i15_attention_value_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/i13_w8a8_gemv_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/w8a8_gemv_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/AI/ReferenceF32WeightGEMVAVX2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/AI/PreparedGEMVWeights.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/prepared_weight_tests.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/AI/ReferenceGEMVAVX2.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/reference/r8_q8_frozen.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/tests/prepared_gemv_tests.cpp")
if(MSVC)
    set_property(SOURCE ${MICIUS_GEMV_STRICT_SOURCES} APPEND PROPERTY COMPILE_OPTIONS /fp:strict)
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    set_property(SOURCE ${MICIUS_GEMV_STRICT_SOURCES} APPEND PROPERTY COMPILE_OPTIONS
        -fno-fast-math -ffp-contract=off)
else()
    message(FATAL_ERROR "Reference GEMV requires a compiler with an explicit strict FP policy")
endif()

add_library(micius_core STATIC ${MICIUS_CORE_SOURCES})
add_library(MICIUS::core ALIAS micius_core)
if(MICIUS_I19_SCORE_GROUP3)
    target_compile_definitions(micius_core PRIVATE MICIUS_I19_SCORE_GROUP3=1)
    set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/AI/I19AttentionScore.cpp"
        APPEND PROPERTY COMPILE_OPTIONS -fno-fast-math -ffp-contract=on)
endif()

if(MICIUS_I17_SCORE_REUSE)
    target_compile_definitions(micius_core PRIVATE MICIUS_I17_SCORE_REUSE=1)
    set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/AI/I17AttentionScore.cpp"
        APPEND PROPERTY COMPILE_OPTIONS -fno-fast-math -ffp-contract=on)
endif()

if(MICIUS_I15_VALUE_TILE32)
    target_compile_definitions(micius_core PRIVATE MICIUS_I15_VALUE_TILE32=1)
endif()
if(MICIUS_I13_GEMV4)
    target_compile_definitions(micius_core PRIVATE MICIUS_I13_GEMV4=1)
endif()

if(MICIUS_I11_SCORE64)
    target_compile_definitions(micius_core PRIVATE MICIUS_I11_SCORE64=1)
    # Match the original Score expression contraction policy, while keeping
    # reassociation off. Explicit SIMD uses F64 FMA across independent rows.
    set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/AI/I11AttentionScore.cpp"
        APPEND PROPERTY COMPILE_OPTIONS -fno-fast-math -ffp-contract=on)
endif()

if(MICIUS_I10_VALUE64)
    target_compile_definitions(micius_core PRIVATE MICIUS_I10_VALUE64=1)
    if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
        set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/AI/I10AttentionValue.cpp"
            APPEND PROPERTY COMPILE_OPTIONS -fno-fast-math -ffp-contract=off)
    else()
        message(FATAL_ERROR "I10 experimental build is scoped to Clang/GNU AVX2")
    endif()
endif()

if(MICIUS_I6_BENCHMARK)
    target_compile_definitions(micius_core PRIVATE MICIUS_I6_BENCHMARK=1)
    target_link_libraries(micius_core PRIVATE psapi)
endif()
if(MICIUS_I2_DIAGNOSTICS)
    target_compile_definitions(micius_core PRIVATE MICIUS_I2_DIAGNOSTICS=1)
endif()
if(MICIUS_I3_INTERVENTION)
    target_compile_definitions(micius_core PRIVATE MICIUS_I3_INTERVENTION=1)
endif()
if(MICIUS_I4_EVALUATION)
    target_compile_definitions(micius_core PRIVATE MICIUS_I4_EVALUATION=1)
endif()
target_compile_definitions(micius_core PRIVATE
    MICIUS_I4_MODE=${MICIUS_I4_MODE}
    MICIUS_I1_MODE=${MICIUS_I1_MODE}
    MICIUS_PW2_MODE=${MICIUS_PW2_MODE} MICIUS_PW2_BUDGET_MIB=${MICIUS_PW2_BUDGET_MIB})

if(MICIUS_BENCHMARK_ITT)
    # Compile the SDK C stub with this toolchain, avoiding an MSVC/MinGW
    # prebuilt-library mismatch. Never download a dependency at configure time.
    foreach(itt_file include/ittnotify.h src/ittnotify/ittnotify_static.c
                     src/ittnotify/ittnotify_config.h src/ittnotify/ittnotify_static.h)
        if(NOT EXISTS "${MICIUS_ITT_ROOT}/${itt_file}")
            message(FATAL_ERROR
                "MICIUS_BENCHMARK_ITT requires a complete MICIUS_ITT_ROOT; "
                "missing ${MICIUS_ITT_ROOT}/${itt_file}")
        endif()
    endforeach()
    enable_language(C)
    add_library(micius_ittnotify STATIC
        "${MICIUS_ITT_ROOT}/src/ittnotify/ittnotify_static.c")
    target_include_directories(micius_ittnotify SYSTEM PUBLIC "$<BUILD_INTERFACE:${MICIUS_ITT_ROOT}/include>")
    target_link_libraries(micius_ittnotify PRIVATE ${CMAKE_DL_LIBS} Threads::Threads)
    if(MICIUS_STATIC_RUNTIME AND MSVC)
        set_property(TARGET micius_ittnotify PROPERTY
            MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
    target_link_libraries(micius_core PRIVATE micius_ittnotify)
    set_property(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/app/MiciusShell.cpp"
        APPEND PROPERTY COMPILE_DEFINITIONS MICIUS_BENCHMARK_ITT=1)
    message(STATUS "ITT benchmark support ON; collection requires explicit --itt-roi")
endif()

if(MICIUS_REFERENCE_GEMV_AVX2)
    target_compile_definitions(micius_core PRIVATE MICIUS_REFERENCE_GEMV_AVX2=1)
    message(STATUS "Checked AVX2 kernels enabled")
endif()

if(MICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2)
    target_compile_definitions(micius_core PRIVATE
        MICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2=1)
    message(STATUS "AVX2 row-pair reference kernels available")
endif()

if(MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES)
    target_compile_definitions(micius_core PRIVATE
        MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES=1)
    message(STATUS "V1 binding validates descriptors")
endif()

if(MICIUS_REFERENCE_GEMV_PREPARED_F64)
    target_compile_definitions(micius_core PRIVATE MICIUS_REFERENCE_GEMV_PREPARED_F64=1)
    message(STATUS "Prepared F64 GEMV input ON: invocation-scoped snapshot; Q8 weights retained")
endif()

if(MICIUS_REFERENCE_GEMV_COMPACT_SCALES)
    target_compile_definitions(micius_core PRIVATE
        MICIUS_REFERENCE_GEMV_COMPACT_SCALES=1)
    message(WARNING
        "MICIUS_REFERENCE_GEMV_COMPACT_SCALES is ON: validated V1 models store "
        "one private F32 scale per Q8_32 block; checked fallback remains unchanged")
endif()

if(MICIUS_REFERENCE_ATTENTION_VALUE_AVX2)
    target_compile_definitions(micius_core PRIVATE
        MICIUS_REFERENCE_ATTENTION_VALUE_AVX2=1)
    message(STATUS "Ordered AVX2 Value fallback available")
endif()

if(MICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE)
    target_compile_definitions(micius_core PRIVATE
        MICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE=1)
    message(WARNING
        "MICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE is ON: each reference "
        "attention query head is converted to exact F64 once and reused by "
        "the ordered key dot products")
endif()

if(MICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD)
    target_compile_definitions(micius_core PRIVATE
        MICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD=1)
    message(STATUS "Ordered time-quad Score fallback available")
endif()

target_include_directories(micius_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    # Original include roots remain valid when the historical tree is restored.
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/AI>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/app>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(micius_core
    PUBLIC
        Threads::Threads
    PRIVATE
        $<BUILD_INTERFACE:micius_build_options>
)

if(MICIUS_STATIC_RUNTIME AND MSVC)
    set_property(TARGET micius_core PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

set(MICIUS_IPO_AVAILABLE FALSE)
if(MICIUS_ENABLE_LTO)
    check_ipo_supported(RESULT MICIUS_IPO_AVAILABLE OUTPUT MICIUS_IPO_DIAGNOSTIC LANGUAGES CXX)
    if(NOT MICIUS_IPO_AVAILABLE)
        message(WARNING "LTO/IPO requested but unavailable: ${MICIUS_IPO_DIAGNOSTIC}")
    endif()
endif()

function(micius_enable_ipo target_name)
    if(MICIUS_ENABLE_LTO AND MICIUS_IPO_AVAILABLE)
        set_property(TARGET ${target_name} PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    endif()
endfunction()

set(MICIUS_LLD_AVAILABLE FALSE)
if(MICIUS_USE_LLD AND
   NOT MSVC AND
   CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
    find_program(MICIUS_LLD_EXECUTABLE NAMES ld.lld lld-link)
    if(MICIUS_LLD_EXECUTABLE)
        check_cxx_compiler_flag("-fuse-ld=lld" MICIUS_COMPILER_ACCEPTS_LLD)
        if(MICIUS_COMPILER_ACCEPTS_LLD)
            set(MICIUS_LLD_AVAILABLE TRUE)
        else()
            message(STATUS
                "LLD was found, but ${CMAKE_CXX_COMPILER_ID} cannot link with -fuse-ld=lld; "
                "using the default linker")
        endif()
    else()
        message(STATUS "LLD requested but not found; using the compiler's default linker")
    endif()
endif()

set(MICIUS_STATIC_GNU_RUNTIME_AVAILABLE FALSE)
if(MICIUS_STATIC_RUNTIME AND
   NOT APPLE AND
   NOT MSVC AND
   CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    check_cxx_compiler_flag("-static-libgcc" MICIUS_HAS_STATIC_LIBGCC)
    check_cxx_compiler_flag("-static-libstdc++" MICIUS_HAS_STATIC_LIBSTDCXX)
    if(MICIUS_HAS_STATIC_LIBGCC AND MICIUS_HAS_STATIC_LIBSTDCXX)
        set(MICIUS_STATIC_GNU_RUNTIME_AVAILABLE TRUE)
    else()
        message(WARNING
            "Static compiler runtime requested but unavailable; executables will use shared runtimes")
    endif()
endif()

function(micius_configure_executable target_name)
    target_link_libraries(${target_name} PRIVATE micius_build_options)
    micius_enable_ipo(${target_name})

    if(MICIUS_ENABLE_LEGACY_TUNING AND
       NOT MSVC AND
       CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_link_options(${target_name} PRIVATE -g)
    endif()

    if(MICIUS_LLD_AVAILABLE)
        target_link_options(${target_name} PRIVATE -fuse-ld=lld)
    endif()

    if(MICIUS_STATIC_RUNTIME)
        if(MSVC)
            set_property(TARGET ${target_name} PROPERTY
                MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
        elseif(MICIUS_STATIC_GNU_RUNTIME_AVAILABLE)
            target_link_options(${target_name} PRIVATE
                -static-libgcc
                -static-libstdc++
            )
        endif()
    endif()

    if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        set_target_properties(${target_name} PROPERTIES
            PDB_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
            COMPILE_PDB_OUTPUT_DIRECTORY "${CMAKE_RUNTIME_OUTPUT_DIRECTORY}"
        )

        # MinGW-style Clang/lld needs the explicit linker PDB option retained
        # by the historical build. clang-cl uses the target properties above.
        if(NOT MSVC AND MICIUS_LLD_AVAILABLE)
            target_link_options(${target_name} PRIVATE
                "-Wl,--pdb=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${target_name}.pdb"
            )
        elseif(NOT MSVC)
            message(STATUS
                "CodeView is enabled for ${target_name}, but linker PDB output requires LLD")
        endif()
    endif()
endfunction()

micius_enable_ipo(micius_core)

set_target_properties(micius_core PROPERTIES EXPORT_NAME core)
# Static LTO objects require the corresponding linker in downstream executables.
if(MICIUS_ENABLE_LTO AND MICIUS_LLD_AVAILABLE)
    target_link_options(micius_core INTERFACE -fuse-ld=lld)
endif()
if(_int8 AND NOT MSVC)
    foreach(feature avxvnni avx512vnni avxvnniint8 avxvnniint16)
        if(MICIUS_HAS_DISABLE_${feature})
            target_link_options(micius_core INTERFACE "-mno-${feature}")
        endif()
    endforeach()
endif()

add_library(micius_quant STATIC src/app/MiciusTensorCodec.cpp src/quant/WeightFormat.cpp)
add_library(MICIUS::quant ALIAS micius_quant)
set_target_properties(micius_quant PROPERTIES EXPORT_NAME quant)
target_include_directories(micius_quant PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}> PRIVATE src include/app)
target_compile_features(micius_quant PRIVATE cxx_std_20)
if(MSVC)
    target_compile_options(micius_quant PRIVATE /fp:strict)
else()
    target_compile_options(micius_quant PRIVATE -fno-fast-math -ffp-contract=off)
endif()
target_link_libraries(micius_core PUBLIC micius_quant)
add_library(micius_gguf STATIC
    src/app/MiciusGGUFParser.cpp
    src/gguf/ContextDirectory.cpp src/gguf/FileModel.cpp)
add_library(MICIUS::gguf ALIAS micius_gguf)
set_target_properties(micius_gguf PROPERTIES EXPORT_NAME gguf)
target_link_libraries(micius_gguf PUBLIC micius_core PRIVATE $<BUILD_INTERFACE:micius_build_options>)
target_compile_definitions(micius_gguf PRIVATE $<TARGET_PROPERTY:micius_core,COMPILE_DEFINITIONS>)
if(WIN32)
    target_link_libraries(micius_gguf PRIVATE bcrypt)
else()
    find_package(OpenSSL REQUIRED COMPONENTS Crypto)
    target_link_libraries(micius_gguf PRIVATE OpenSSL::Crypto)
endif()
add_library(micius_sampling STATIC src/sampling/Sampler.cpp)
add_library(MICIUS::sampling ALIAS micius_sampling)
set_target_properties(micius_sampling PROPERTIES EXPORT_NAME sampling)
target_include_directories(micius_sampling PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
target_compile_features(micius_sampling PRIVATE cxx_std_20)
if(MSVC)
    target_compile_options(micius_sampling PRIVATE /fp:strict)
else()
    target_compile_options(micius_sampling PRIVATE -fno-fast-math -ffp-contract=off)
endif()
set(_release_targets micius_core micius_gguf micius_sampling micius_quant)
if(MICIUS_ENABLE_TEXT)
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(MICIUS_PCRE2 REQUIRED IMPORTED_TARGET libpcre2-8)
    add_library(micius_text STATIC src/text/Tokenizer.cpp src/text/TextABI.cpp src/text/Conversation.cpp)
    add_library(MICIUS::text ALIAS micius_text)
    set_target_properties(micius_text PROPERTIES EXPORT_NAME text)
    target_link_libraries(micius_text PUBLIC micius_gguf micius_sampling PRIVATE PkgConfig::MICIUS_PCRE2)
    add_library(micius_session STATIC src/session/Session.cpp src/session/Storage.cpp src/session/SamplingConfig.cpp)
    add_library(MICIUS::session ALIAS micius_session)
    set_target_properties(micius_session PROPERTIES EXPORT_NAME session)
    target_link_libraries(micius_session PUBLIC micius_text)
    list(APPEND _release_targets micius_text micius_session)
    install(FILES include/micius_text.h include/micius_session.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
set(_parser_targets)
set(_cli_parsers)
foreach(arch Llama Qwen2 Qwen3)
    string(TOUPPER ${arch} flag)
    string(TOLOWER ${arch} key)
    if(MICIUS_PARSER_${flag})
        set(target micius_parser_${key})
        add_library(${target} STATIC src/gguf/${arch}Parser.cpp)
        add_library(MICIUS::parser_${key} ALIAS ${target})
        set_target_properties(${target} PROPERTIES EXPORT_NAME parser_${key})
        target_include_directories(${target} PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
            $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)
        target_compile_features(${target} PRIVATE cxx_std_20)
        list(APPEND _parser_targets ${target})
        list(APPEND _release_targets ${target})
        list(APPEND _cli_parsers MICIUS_LINK_${flag}=1)
    endif()
endforeach()
if(MSVC AND MICIUS_STATIC_RUNTIME)
    set_property(TARGET micius_gguf ${_parser_targets} PROPERTY
        MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

if(_parser_targets AND (MICIUS_BUILD_CLI OR BUILD_TESTING))
    add_library(micius_shell STATIC src/app/MiciusShell.cpp)
    if(MICIUS_I6_BENCHMARK)
        target_sources(micius_shell PRIVATE src/app/I6Benchmark.cpp)
    endif()
    target_compile_definitions(micius_shell PRIVATE ${_cli_parsers} $<TARGET_PROPERTY:micius_core,COMPILE_DEFINITIONS>)
    target_link_libraries(micius_shell PUBLIC micius_gguf PRIVATE ${_parser_targets} micius_build_options)
    if(MICIUS_ENABLE_TEXT)
        target_sources(micius_shell PRIVATE src/app/ChatShell.cpp)
        target_link_libraries(micius_shell PRIVATE micius_session)
        target_compile_definitions(micius_shell PUBLIC MICIUS_HAS_TEXT=1)
    endif()
    if(MICIUS_BENCHMARK_ITT)
        target_link_libraries(micius_shell PRIVATE micius_ittnotify)
    endif()
    if(MSVC AND MICIUS_STATIC_RUNTIME)
        set_property(TARGET micius_shell PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    endif()
endif()
if(MICIUS_BUILD_CLI)
    if(NOT _parser_targets)
        message(FATAL_ERROR "The CLI requires at least one architecture parser")
    endif()
    add_executable(micius_cli src/app/MiciusCLIMain.cpp)
    set(_cli_artwork "${CMAKE_CURRENT_SOURCE_DIR}/src/app/MICIUS.txt")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_cli_artwork}")
    file(READ "${_cli_artwork}" MICIUS_CLI_ARTWORK)
    string(REPLACE "\r\n" "\n" MICIUS_CLI_ARTWORK "${MICIUS_CLI_ARTWORK}")
    configure_file(cmake/CliArtwork.h.in generated/CliArtwork.h @ONLY)
    target_include_directories(micius_cli PRIVATE "${CMAKE_CURRENT_BINARY_DIR}/generated")
    target_compile_definitions(micius_cli PRIVATE MICIUS_VERSION_STRING="${PROJECT_VERSION}")
    target_link_libraries(micius_cli PRIVATE micius_shell)
    micius_configure_executable(micius_cli)
    install(TARGETS micius_cli RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
endif()

# Keep the original sandbox target name and output behavior. The rebuilt source
# exercises the current reference ABI; a custom source can still be supplied.
if(MICIUS_BUILD_SANDBOX)
    if(EXISTS "${MICIUS_SANDBOX_SOURCE}")
        add_executable(micius_sandbox "${MICIUS_SANDBOX_SOURCE}")
        target_link_libraries(micius_sandbox PRIVATE micius_core Threads::Threads)
        if(MICIUS_REFERENCE_GEMV_AVX2)
            target_compile_definitions(micius_sandbox PRIVATE MICIUS_REFERENCE_GEMV_AVX2=1)
        endif()
        if(MICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2)
            target_compile_definitions(micius_sandbox PRIVATE
                MICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2=1)
        endif()
        if(MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES)
            target_compile_definitions(micius_sandbox PRIVATE
                MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES=1)
        endif()
        if(MICIUS_REFERENCE_GEMV_PREPARED_F64)
            target_compile_definitions(micius_sandbox PRIVATE MICIUS_REFERENCE_GEMV_PREPARED_F64=1)
        endif()
        if(MICIUS_REFERENCE_GEMV_COMPACT_SCALES)
            target_compile_definitions(micius_sandbox PRIVATE
                MICIUS_REFERENCE_GEMV_COMPACT_SCALES=1)
        endif()
        if(MICIUS_REFERENCE_ATTENTION_VALUE_AVX2)
            target_compile_definitions(micius_sandbox PRIVATE
                MICIUS_REFERENCE_ATTENTION_VALUE_AVX2=1)
        endif()
        if(MICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE)
            target_compile_definitions(micius_sandbox PRIVATE
                MICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE=1)
        endif()
        if(MICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD)
            target_compile_definitions(micius_sandbox PRIVATE
                MICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD=1)
        endif()
        micius_configure_executable(micius_sandbox)
    else()
        message(STATUS
            "micius_sandbox not generated: source is absent at "
            "${MICIUS_SANDBOX_SOURCE}; set MICIUS_SANDBOX_SOURCE to restore it")
    endif()
endif()

function(micius_add_contract_test target_name source_file)
    if(NOT EXISTS "${source_file}")
        message(FATAL_ERROR "Missing contract test ${target_name}: ${source_file}")
    endif()

    add_executable(${target_name} "${source_file}")
    target_link_libraries(${target_name} PRIVATE micius_core)
    micius_configure_executable(${target_name})

    # The historical contract tests use assert() as their check mechanism.
    # Release configurations define NDEBUG globally, which would otherwise
    # compile most checks out and turn CTest success into a false positive.
    if(MSVC)
        target_compile_options(${target_name} PRIVATE /UNDEBUG)
    elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang")
        target_compile_options(${target_name} PRIVATE -UNDEBUG)
    endif()

    add_test(NAME ${target_name} COMMAND $<TARGET_FILE:${target_name}>)
endfunction()

if(MICIUS_I2_DIAGNOSTICS)
    add_executable(micius_i2_replay tests/reference/i2_replay.cpp)
    target_link_libraries(micius_i2_replay PRIVATE micius_core)
    micius_configure_executable(micius_i2_replay)
    if(BUILD_TESTING)
        micius_add_contract_test(i2_diagnostic_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/i2_diagnostic_tests.cpp")
    endif()
endif()

if(BUILD_TESTING)
    if(MICIUS_I4_EVALUATION)
        micius_add_contract_test(i4_evaluation_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/i4_evaluation_tests.cpp")
        target_sources(i4_evaluation_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests/reference/r8_q8_frozen.cpp")
    endif()
    if(MICIUS_I3_INTERVENTION)
        micius_add_contract_test(i3_intervention_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/i3_intervention_tests.cpp")
        target_sources(i3_intervention_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/tests/reference/r8_q8_frozen.cpp")
    endif()
    if(NOT MICIUS_I1_MODE)
    micius_add_contract_test(runtime_contract_tests
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime_contract_tests.cpp")
    else()
    micius_add_contract_test(w8a8_runtime_contract_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/w8a8_runtime_contract_tests.cpp")
    endif()
    micius_add_contract_test(w8a8_gemv_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/w8a8_gemv_tests.cpp")
    if(MICIUS_PARSER_LLAMA)
        micius_add_contract_test(gguf_parser_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/gguf_parser_tests.cpp")
        target_link_libraries(gguf_parser_tests PRIVATE micius_gguf micius_parser_llama)
    endif()
    micius_add_contract_test(tensor_codec_tests
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/tensor_codec_tests.cpp")
    target_link_libraries(tensor_codec_tests PRIVATE micius_gguf)
    micius_add_contract_test(prepared_gemv_tests
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/prepared_gemv_tests.cpp")
    target_sources(prepared_gemv_tests PRIVATE tests/reference/r8_q8_frozen.cpp)
    micius_add_contract_test(prepared_weight_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/prepared_weight_tests.cpp")
    target_sources(prepared_weight_tests PRIVATE tests/reference/r8_q8_frozen.cpp)
    micius_add_contract_test(q8_32_gemv_tests
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/q8_32_gemv_tests.cpp")
    micius_add_contract_test(attention_value_avx2_tests
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/attention_value_avx2_tests.cpp")
    micius_add_contract_test(attention_query_f64_cache_tests
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/attention_query_f64_cache_tests.cpp")
    micius_add_contract_test(attention_score_time_quad_tests
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/attention_score_time_quad_tests.cpp")
    if(TARGET micius_shell)
        micius_add_contract_test(benchmark_itt_roi_tests
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/benchmark_itt_roi_tests.cpp")
        target_link_libraries(benchmark_itt_roi_tests PRIVATE micius_shell)
        if(MICIUS_BENCHMARK_ITT)
            target_compile_definitions(benchmark_itt_roi_tests PRIVATE MICIUS_BENCHMARK_ITT=1)
            target_link_libraries(benchmark_itt_roi_tests PRIVATE micius_ittnotify)
        endif()
    endif()
endif()

# D1 isolated diagnostics. OFF remains the uninstrumented NONE baseline.
option(MICIUS_DECODE_PROFILE "Compile D1 decode category measurement" OFF)
if(MICIUS_DECODE_PROFILE)
    if(NOT MICIUS_PW2_SELECTION STREQUAL "NONE" OR
       NOT MICIUS_REFERENCE_GEMV_PREPARED_F64 OR
       NOT MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES OR
       MICIUS_REFERENCE_GEMV_COMPACT_SCALES OR MICIUS_ENABLE_FAST_MATH OR MICIUS_BENCHMARK_ITT)
        message(FATAL_ERROR "D1 requires NONE, prepared input, prevalidated descriptor scales, strict math and ITT OFF")
    endif()
    target_compile_definitions(micius_core PRIVATE MICIUS_DECODE_PROFILE=1)
endif()
if(BUILD_TESTING)
    micius_add_contract_test(decode_profile_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/decode_profile_tests.cpp")
    target_compile_definitions(decode_profile_tests PRIVATE MICIUS_DECODE_PROFILE=1)
    if(MICIUS_DECODE_PROFILE)
        micius_add_contract_test(d1_runtime_contract_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/d1_runtime_contract_tests.cpp")
        target_compile_definitions(d1_runtime_contract_tests PRIVATE MICIUS_DECODE_PROFILE=1)
    endif()
endif()

if(BUILD_TESTING AND MICIUS_I10_VALUE64)
    micius_add_contract_test(i10_attention_value_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/i10_attention_value_tests.cpp")
endif()

if(BUILD_TESTING AND MICIUS_I11_SCORE64)
    micius_add_contract_test(i11_attention_score_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/i11_attention_score_tests.cpp")
endif()

if(BUILD_TESTING AND MICIUS_I13_GEMV4)
    micius_add_contract_test(i13_w8a8_gemv_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/i13_w8a8_gemv_tests.cpp")
endif()

if(BUILD_TESTING AND MICIUS_I15_VALUE_TILE32)
    micius_add_contract_test(i15_attention_value_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/i15_attention_value_tests.cpp")
endif()

if(BUILD_TESTING AND MICIUS_I17_SCORE_REUSE)
    micius_add_contract_test(i17_attention_score_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/i17_attention_score_tests.cpp")
endif()

if(BUILD_TESTING AND MICIUS_I19_SCORE_GROUP3)
    micius_add_contract_test(i19_attention_score_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/i19_attention_score_tests.cpp")
endif()

# Descriptor copies become releasable after a successful checked bind.
if(MICIUS_I22_RELEASE_LOADER_SCALES)
  if(NOT MICIUS_I19_SCORE_GROUP3 OR NOT MICIUS_I1_MODE EQUAL 2)
    message(FATAL_ERROR "Loader descriptor retirement requires the I19 AVX2 binding configuration.")
  endif()
  set_property(SOURCE src/gguf/FileModel.cpp APPEND PROPERTY COMPILE_DEFINITIONS MICIUS_I22_RELEASE_LOADER_SCALES=1)
endif()
if(BUILD_TESTING)
  micius_add_contract_test(i22_loader_lifetime_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/i22_loader_lifetime_tests.cpp")
  target_include_directories(i22_loader_lifetime_tests PRIVATE include src)
endif()

if(MICIUS_I25_PANEL_GEMV)
  if(NOT MICIUS_I22_RELEASE_LOADER_SCALES OR NOT MICIUS_I1_MODE EQUAL 2 OR
     MICIUS_ENABLE_FAST_MATH OR MICIUS_REFERENCE_GEMV_COMPACT_SCALES OR
     NOT MICIUS_PW2_SELECTION STREQUAL "NONE" OR MICIUS_I2_DIAGNOSTICS OR
     MICIUS_I3_INTERVENTION OR MICIUS_I4_EVALUATION OR MICIUS_DECODE_PROFILE)
    message(FATAL_ERROR "I25 requires uninstrumented strict I22 AVX2 configuration")
  endif()
  set_property(SOURCE src/AI/AICoreEngine.cpp APPEND PROPERTY COMPILE_DEFINITIONS MICIUS_I25_PANEL_GEMV=1)
endif()
target_include_directories(micius_core PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/src/AI/i25")
if(BUILD_TESTING)
  micius_add_contract_test(i25_panel_contract "${CMAKE_CURRENT_SOURCE_DIR}/tests/i25_panel_contract.cpp")
  target_include_directories(i25_panel_contract PRIVATE src/AI/i25)
  if(MICIUS_I25_PANEL_GEMV)
    micius_add_contract_test(i25_ownership_tests "${CMAKE_CURRENT_SOURCE_DIR}/tests/i25_ownership_tests.cpp")
  endif()
endif()

if(BUILD_TESTING AND MICIUS_BUILD_CLI AND MICIUS_PARSER_LLAMA AND _release_simd)
    find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter)
    set(_trace_check)
    if(WIN32 AND MINGW AND CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(_trace_check --golden)
    endif()
    add_test(NAME release_cli_contract
        COMMAND "${Python3_EXECUTABLE}" -B "${CMAKE_CURRENT_SOURCE_DIR}/tests/cli_release_tests.py"
                $<TARGET_FILE:micius_cli> "${MICIUS_BACKEND}" ${_trace_check})
endif()

if(TARGET micius_ittnotify)
    list(APPEND _release_targets micius_ittnotify)
endif()
install(TARGETS ${_release_targets} EXPORT MICIUSTargets ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR})
install(FILES include/micius_runtime.h include/micius_gguf.h include/micius_sampling.h include/micius_state.h DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
set(_micius_package "${CMAKE_INSTALL_LIBDIR}/cmake/MICIUS")
configure_package_config_file(cmake/MICIUSConfig.cmake.in "${CMAKE_CURRENT_BINARY_DIR}/MICIUSConfig.cmake"
    INSTALL_DESTINATION "${_micius_package}")
write_basic_package_version_file("${CMAKE_CURRENT_BINARY_DIR}/MICIUSConfigVersion.cmake"
    VERSION ${PROJECT_VERSION} COMPATIBILITY SameMinorVersion)
install(EXPORT MICIUSTargets NAMESPACE MICIUS:: DESTINATION "${_micius_package}")
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/MICIUSConfig.cmake" "${CMAKE_CURRENT_BINARY_DIR}/MICIUSConfigVersion.cmake"
    DESTINATION "${_micius_package}")

if(BUILD_TESTING)
    enable_language(C)
    add_executable(m3_sampling_contract tests/sampling_contract.cpp src/session/SamplingConfig.cpp)
    target_include_directories(m3_sampling_contract PRIVATE src)
    target_link_libraries(m3_sampling_contract PRIVATE micius_sampling)
    add_test(NAME sampling_contract COMMAND m3_sampling_contract)
    add_executable(m2_logger_contract tests/logger_contract.cpp)
    target_link_libraries(m2_logger_contract PRIVATE micius_core)
    add_test(NAME logger_contract COMMAND m2_logger_contract)
    if(MICIUS_ENABLE_TEXT)
        add_executable(m2_token_probe examples/text-consumer/token_probe.c)
        target_link_libraries(m2_token_probe PRIVATE micius_text)
        set_target_properties(m2_token_probe PROPERTIES LINKER_LANGUAGE CXX C_STANDARD 11)
        if(MICIUS_BUILD_CLI AND MICIUS_PARSER_LLAMA AND MICIUS_PARSER_QWEN2 AND _release_simd)
            find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter)
            add_executable(m3_session_probe examples/session-consumer/session_probe.c)
            target_link_libraries(m3_session_probe PRIVATE micius_session ${_parser_targets})
            if(MICIUS_PARSER_QWEN3)
                target_compile_definitions(m3_session_probe PRIVATE LINK_QWEN3)
            endif()
            set_target_properties(m3_session_probe PROPERTIES LINKER_LANGUAGE CXX C_STANDARD 11)
            add_executable(m3_session_contract tests/session_contract.cpp)
            target_link_libraries(m3_session_contract PRIVATE micius_session ${_parser_targets})
            add_test(NAME persistent_session_contract COMMAND "${Python3_EXECUTABLE}" -B
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/session_tests.py" $<TARGET_FILE:m3_session_probe>
                $<TARGET_FILE:m3_session_contract> $<TARGET_FILE:micius_cli>)
            add_test(NAME persistent_session_runner COMMAND "${Python3_EXECUTABLE}" -B
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/m3_runner_tests.py" $<TARGET_FILE:m3_session_probe>)
            add_executable(m2_chat_contract tests/chat_state_contract.cpp)
            target_link_libraries(m2_chat_contract PRIVATE micius_text ${_parser_targets})
            add_test(NAME chat_text_contract COMMAND "${Python3_EXECUTABLE}" -B
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/chat_text_tests.py" $<TARGET_FILE:m2_token_probe>
                $<TARGET_FILE:m2_chat_contract> $<TARGET_FILE:micius_cli>)
        endif()
    endif()
    add_executable(abi_v07_client tests/abi_v07_client.c)
    target_include_directories(abi_v07_client BEFORE PRIVATE tests/reference/v07)
    target_link_libraries(abi_v07_client PRIVATE micius_core)
    set_target_properties(abi_v07_client PROPERTIES LINKER_LANGUAGE CXX C_STANDARD 11)
    add_test(NAME abi_v07_client COMMAND abi_v07_client)
    if(_parser_targets)
        add_executable(m1_model_probe examples/file-consumer/model_probe.c)
        target_link_libraries(m1_model_probe PRIVATE micius_gguf ${_parser_targets})
        foreach(arch LLAMA QWEN2 QWEN3)
            if(MICIUS_PARSER_${arch})
                target_compile_definitions(m1_model_probe PRIVATE LINK_${arch})
            endif()
        endforeach()
        set_target_properties(m1_model_probe PROPERTIES LINKER_LANGUAGE CXX C_STANDARD 11)
    endif()
    if(MICIUS_PARSER_QWEN2 AND _release_simd)
        find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter)
        if(MICIUS_BUILD_CLI)
            add_executable(m1_file_session_contract tests/file_session_contract.cpp)
            target_link_libraries(m1_file_session_contract PRIVATE micius_gguf micius_parser_qwen2)
            add_test(NAME file_session_contract COMMAND "${Python3_EXECUTABLE}" -B
                "${CMAKE_CURRENT_SOURCE_DIR}/tests/module_session_tests.py"
                $<TARGET_FILE:m1_file_session_contract> $<TARGET_FILE:micius_cli>)
        endif()
        add_test(NAME qwen2_forward_contract COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/qwen2_forward_tests.py" $<TARGET_FILE:m1_model_probe> "${MICIUS_BACKEND}")
    endif()
endif()

# Fail closed on snapshots from another numerical build. This is an identity
# boundary, not a portable cross-compiler snapshot compatibility promise.
file(GLOB_RECURSE _state_files CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/src/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h")
list(APPEND _state_files "${CMAKE_CURRENT_SOURCE_DIR}/src/micius_runtime.cpp" "${CMAKE_CURRENT_SOURCE_DIR}/CMakeLists.txt")
list(SORT _state_files)
string(TOUPPER "${CMAKE_BUILD_TYPE}" _state_config)
set(_state_identity "M3-state-v1;${CMAKE_CXX_COMPILER_ID};${CMAKE_CXX_COMPILER_VERSION};${CMAKE_SYSTEM_NAME};${CMAKE_SYSTEM_PROCESSOR};${CMAKE_SIZEOF_VOID_P};${CMAKE_BUILD_TYPE};${CMAKE_CXX_FLAGS};${CMAKE_CXX_FLAGS_${_state_config}}")
foreach(f IN LISTS _state_files)
    file(SHA256 "${f}" h)
    string(APPEND _state_identity ";${h}")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${f}")
endforeach()
get_cmake_property(_state_variables VARIABLES)
list(SORT _state_variables)
foreach(v IN LISTS _state_variables)
    if(v MATCHES "^MICIUS_(BACKEND|REFERENCE_|I[0-9]|PW2_|ENABLE_|DECODE_PROFILE)" OR v STREQUAL "ARCH" OR v STREQUAL "ALIGN")
        string(APPEND _state_identity ";${v}=${${v}}")
    endif()
endforeach()
if(ARCH STREQUAL "native")
    cmake_host_system_information(RESULT _cpu QUERY PROCESSOR_DESCRIPTION)
    string(APPEND _state_identity ";${_cpu}")
endif()
string(SHA256 _state_hash "${_state_identity}")
target_compile_definitions(micius_core PRIVATE MICIUS_STATE_CONTRACT_ID="${_state_hash}")
if(MSVC AND MICIUS_STATIC_RUNTIME)
    set_property(TARGET ${_release_targets} PROPERTY MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
endif()

if(BUILD_TESTING AND MICIUS_PARSER_LLAMA AND MICIUS_PARSER_QWEN2 AND MICIUS_PARSER_QWEN3)
    find_package(Python3 3.8 REQUIRED COMPONENTS Interpreter)
    add_executable(m4_weight_probe tests/weight_format_probe.cpp)
    target_include_directories(m4_weight_probe PRIVATE src)
    target_link_libraries(m4_weight_probe PRIVATE micius_quant)
    add_test(NAME dense_weight_forward_contract COMMAND "${Python3_EXECUTABLE}" -B
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/dense_forward_tests.py" $<TARGET_FILE:m1_model_probe> $<TARGET_FILE:m4_weight_probe>)
    if(TARGET m3_session_probe)
        add_test(NAME dense_delivery_runner COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/m4_runner_tests.py" $<TARGET_FILE:m3_session_probe> $<TARGET_FILE:m1_model_probe>
            $<TARGET_FILE:m2_token_probe> $<TARGET_FILE:m2_chat_contract>)
        add_test(NAME dense_text_session_contract COMMAND "${Python3_EXECUTABLE}" -B
            "${CMAKE_CURRENT_SOURCE_DIR}/tests/dense_text_tests.py" $<TARGET_FILE:m2_token_probe>
            $<TARGET_FILE:m2_chat_contract> $<TARGET_FILE:m3_session_probe>)
    endif()
endif()

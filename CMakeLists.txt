cmake_minimum_required(VERSION 3.16)

project(MICIUS VERSION 0.6.0 LANGUAGES CXX)

include(CTest)
include(CheckCXXCompilerFlag)
include(CheckIPOSupported)

# The rebuilt runtime uses std::construct_at/std::destroy_at and therefore
# cannot be compiled as C++17 without changing source-level lifetime handling.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

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
       "Route reference-mode GEMV calls through the experimental checked Q8_32 AVX2 backend" OFF)
option(MICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2
       "Use the experimental two-row shared-input/direct-weight-load Q8_32 AVX2 backend" OFF)
option(MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES
       "Use the prevalidated GEMV entry only for models owned by the private V1 binding boundary" OFF)
option(MICIUS_REFERENCE_GEMV_PREPARED_F64
       "Prepare one validated F64 activation snapshot for reference row-pair GEMV" OFF)
set(MICIUS_PW2_SELECTION "NONE" CACHE STRING "PW2 experimental weight snapshot: NONE, QKV, O")
set_property(CACHE MICIUS_PW2_SELECTION PROPERTY STRINGS NONE QKV O)
set(MICIUS_PW2_BUDGET_MIB "512" CACHE STRING "Maximum extra raw F32 weight payload per binding")
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
       "Route reference attention V aggregation through the order-preserving AVX2 backend" OFF)
option(MICIUS_REFERENCE_ATTENTION_QUERY_F64_CACHE
       "Convert each reference attention query head to exact F64 once before its key scan" OFF)
option(MICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD
       "Evaluate four ordered reference attention key rows while sharing each query conversion" OFF)

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

    # Retain the original Windows/Clang CodeView behavior.
    if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
        target_compile_options(micius_build_options INTERFACE -gcodeview)
    endif()
else()
    message(WARNING
        "Compiler '${CMAKE_CXX_COMPILER_ID}' has no MICIUS tuning profile; "
        "only the required C++20 feature level will be applied")
endif()

# Preserve CONFIGURE_DEPENDS source discovery while separating the one program
# entry point from the reusable static archive. Unlike the original file, this
# does not compile Shell/Parser twice into both the archive and the CLI.
file(GLOB_RECURSE ALL_SRC_FILES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

set(MICIUS_CLI_MAIN_SOURCES ${ALL_SRC_FILES})
list(FILTER MICIUS_CLI_MAIN_SOURCES INCLUDE REGEX "MiciusCLIMain\\.cpp$")

set(MICIUS_CORE_SOURCES ${ALL_SRC_FILES})
list(FILTER MICIUS_CORE_SOURCES EXCLUDE REGEX "MiciusCLIMain\\.cpp$")

if(NOT MICIUS_CORE_SOURCES)
    message(FATAL_ERROR "No MICIUS core sources were found below src/")
endif()

# Isolate the reference GEMV FP contract from legacy draft arithmetic. Both
# prepared OFF/ON compile the same no-contraction baseline. The frozen oracle
# uses separate rounding too, including scalar tails. Strict options are source
# options and therefore follow target/global optimization options.
set(MICIUS_GEMV_STRICT_SOURCES
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
target_compile_definitions(micius_core PRIVATE
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
    target_include_directories(micius_ittnotify SYSTEM PUBLIC "${MICIUS_ITT_ROOT}/include")
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
    message(WARNING
        "MICIUS_REFERENCE_GEMV_AVX2 is ON: reference-mode GEMV uses the experimental "
        "AVX2 backend and is no longer the scalar-double oracle")
endif()

if(MICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2)
    target_compile_definitions(micius_core PRIVATE
        MICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2=1)
    message(WARNING
        "MICIUS_REFERENCE_GEMV_ROW_PAIR_AVX2 is ON: reference-mode GEMV uses "
        "the experimental two-row shared-input/direct-weight-load AVX2 backend")
endif()

if(MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES)
    target_compile_definitions(micius_core PRIVATE
        MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES=1)
    message(WARNING
        "MICIUS_REFERENCE_GEMV_PREVALIDATED_SCALES is ON: only models accepted "
        "by the private V1 validated-binding boundary use the fast scale path; "
        "all other engine states retain checked GEMV")
endif()

if(MICIUS_REFERENCE_GEMV_PREPARED_F64)
    target_compile_definitions(micius_core PRIVATE MICIUS_REFERENCE_GEMV_PREPARED_F64=1)
    message(WARNING "Prepared F64 GEMV input ON: invocation-scoped snapshot; Q8 weights retained")
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
    message(WARNING
        "MICIUS_REFERENCE_ATTENTION_VALUE_AVX2 is ON: reference-mode attention "
        "V aggregation uses the experimental order-preserving AVX2 backend")
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
    message(WARNING
        "MICIUS_REFERENCE_ATTENTION_SCORE_TIME_QUAD is ON: reference attention "
        "scores evaluate four chronological key rows together while sharing "
        "each exact F32-to-F64 query conversion")
endif()

target_include_directories(micius_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/src>
    # Original include roots remain valid when the historical tree is restored.
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/AI>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include/app>
    $<INSTALL_INTERFACE:include>
)

target_link_libraries(micius_core
    PUBLIC
        Threads::Threads
    PRIVATE
        micius_build_options
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

if(MICIUS_BUILD_CLI)
    if(NOT MICIUS_CLI_MAIN_SOURCES)
        message(FATAL_ERROR "MICIUS_BUILD_CLI is ON, but MiciusCLIMain.cpp was not found")
    endif()

    add_executable(micius_cli ${MICIUS_CLI_MAIN_SOURCES})
    target_link_libraries(micius_cli PRIVATE micius_core Threads::Threads)
    micius_configure_executable(micius_cli)
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
        message(WARNING "Skipping ${target_name}: source not found at ${source_file}")
        return()
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

if(BUILD_TESTING)
    micius_add_contract_test(runtime_contract_tests
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime_contract_tests.cpp")
    micius_add_contract_test(gguf_parser_tests
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/gguf_parser_tests.cpp")
    micius_add_contract_test(tensor_codec_tests
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/tensor_codec_tests.cpp")
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
    micius_add_contract_test(benchmark_itt_roi_tests
        "${CMAKE_CURRENT_SOURCE_DIR}/tests/benchmark_itt_roi_tests.cpp")
    if(MICIUS_BENCHMARK_ITT)
        target_compile_definitions(benchmark_itt_roi_tests PRIVATE MICIUS_BENCHMARK_ITT=1)
        target_link_libraries(benchmark_itt_roi_tests PRIVATE micius_ittnotify)
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

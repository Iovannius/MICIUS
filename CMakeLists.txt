cmake_minimum_required(VERSION 3.16)
project(MICIUS LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(ARCH "native" CACHE STRING "Target architecture optimization (e.g., native, haswell, znver3)")
set(ALIGN "32" CACHE STRING "Loop alignment boundary")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

find_package(Threads REQUIRED)

file(GLOB_RECURSE ALL_SRC_FILES CONFIGURE_DEPENDS
    "src/*.cpp"
)

list(FILTER ALL_SRC_FILES EXCLUDE REGEX "MiciusCLIMain\\.cpp$")

add_library(micius_core STATIC ${ALL_SRC_FILES})

target_include_directories(micius_core PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/AI
    ${CMAKE_CURRENT_SOURCE_DIR}/include/app
)

target_compile_options(micius_core PRIVATE
    -O3
    -g
    -fno-omit-frame-pointer
    -march=${ARCH}
    -ffast-math
    -flto=thin
    -funroll-loops
    -falign-functions=32
    -falign-loops=${ALIGN}
    -mprefer-vector-width=256
    -fno-stack-protector
)

if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(micius_core PRIVATE -gcodeview)
endif()

add_executable(micius_cli src/app/MiciusCLIMain.cpp)
target_link_libraries(micius_cli PRIVATE micius_core Threads::Threads)

target_link_options(micius_cli PRIVATE
    -flto=thin
    -g
    -static-libgcc
    -static-libstdc++
    -fuse-ld=lld
)

if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_link_options(micius_cli PRIVATE
        -Wl,--pdb=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/micius_cli.pdb
    )
endif()

add_executable(micius_sandbox tests/stress/AIEmuSandbox.cpp)
target_link_libraries(micius_sandbox PRIVATE micius_core Threads::Threads)

target_link_options(micius_sandbox PRIVATE
    -flto=thin
    -g
    -static-libgcc
    -static-libstdc++
    -fuse-ld=lld
)

if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_link_options(micius_sandbox PRIVATE
        -Wl,--pdb=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/micius_sandbox.pdb
    )
endif()
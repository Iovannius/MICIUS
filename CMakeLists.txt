cmake_minimum_required(VERSION 3.16)
project(MICIUS LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(ARCH "native" CACHE STRING "")
set(ALIGN "32" CACHE STRING "")

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

file(GLOB SRC_FILES CONFIGURE_DEPENDS
    "tests/AIEmuMain.cpp"
    "src/AI/*.cpp"
)

find_package(Threads REQUIRED)

add_executable(my_target ${SRC_FILES})

target_include_directories(my_target PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/include/AI)

target_compile_options(my_target PRIVATE
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
    -pthread
)

target_link_options(my_target PRIVATE
    -flto=thin
    -g
    -static-libgcc
    -static-libstdc++
    -fuse-ld=lld
    -pthread
)

target_link_libraries(my_target PRIVATE Threads::Threads)

if(WIN32 AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    target_compile_options(my_target PRIVATE -gcodeview)
    target_link_options(my_target PRIVATE
        -Wl,--pdb=${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/my_target.pdb
    )
endif()
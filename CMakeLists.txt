cmake_minimum_required(VERSION 3.16)
project(MyClangProject LANGUAGES CXX)

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
    -march=${ARCH}
    -ffast-math
    -flto=thin
    -funroll-loops
    -falign-loops=${ALIGN}
    -mprefer-vector-width=256
    -fno-stack-protector
    -pthread
)

target_link_options(my_target PRIVATE
    -flto=thin
    -static
    -pthread
    -fuse-ld=lld
)

target_link_libraries(my_target PRIVATE Threads::Threads)
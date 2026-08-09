cmake_minimum_required(VERSION 3.8)

project(ModernFastInverseSqareRoot LANGUAGES CXX)

# Build the core library with the fast inverse‑sqrt implementations.
add_library(fisq STATIC fast_inverse_sqrt.cpp)
target_include_directories(fisq PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Benchmark executable
add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PRIVATE fisq)

# Use C++20 for all targets and enable optimisations.
if (CMAKE_VERSION VERSION_GREATER 3.12)
    set_property(TARGET fisq benchmark PROPERTY CXX_STANDARD 20)
endif()
target_compile_options(fisq
    PRIVATE "$<$<CXX_COMPILER_ID:MSVC>:/O2>$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O2>"
)


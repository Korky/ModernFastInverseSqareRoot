cmake_minimum_required(VERSION 3.11)

project(ModernFastInverseSqareRoot LANGUAGES CXX)

# Enable generation of compile_commands.json for clangd support.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# No external test framework is required; a lightweight custom test harness
# will be used instead.

# Build the core library with the fast inverse‑sqrt implementations.
add_library(fisq STATIC fast_inverse_sqrt.cpp)
target_include_directories(fisq PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Benchmark executable
add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PRIVATE fisq)

# Unit tests using a lightweight custom harness
enable_testing()
add_executable(fisq_tests tests/test_fast_inverse_sqrt.cpp)
target_include_directories(fisq_tests PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(fisq_tests PRIVATE fisq)
set_property(TARGET fisq_tests PROPERTY CXX_STANDARD 20)
add_test(NAME fisq_tests COMMAND fisq_tests)

# Use C++20 for all targets and enable optimisations.
if (CMAKE_VERSION VERSION_GREATER 3.12)
    set_property(TARGET fisq benchmark PROPERTY CXX_STANDARD 20)
endif()

# MSVC enables /RTC1 in Debug mode which conflicts with /O2 optimisation.
# The custom compile options below add /O2 for all configurations, so we
# do not override the default debug flags here.

# MSVC enables /RTC1 in Debug mode which conflicts with /O2 optimisation.
# The following compile options enable optimisations only for Release builds
# and disable runtime checks for Debug builds, avoiding the conflict.
target_compile_options(fisq
    PRIVATE
        # Enable /O2 optimisation for MSVC in Release configuration
        "$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Release>>:/O2>"
        # Disable runtime checks (/RTC0) for MSVC in Debug configuration
        "$<$<AND:$<CXX_COMPILER_ID:MSVC>,$<CONFIG:Debug>>:/RTC0>"
        # For non-MSVC compilers, use -O2 in all configurations
        "$<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-O2>"
        # Ensure at least SSE3 support for the RSQRTSS intrinsic used by the
        # SIMD implementation.  GCC/Clang honour this flag on x86_64.
        "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-msse3>"
        # Enable AVX2 only if the compiler supports it; GCC/Clang honour
        # this flag on x86_64.
        "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-mavx2>"
        # Enable native architecture optimisations for GCC/Clang
        $<$<CXX_COMPILER_ID:GNU,Clang>:-march=native>
        # MSVC
        $<$<CXX_COMPILER_ID:MSVC>:/arch:AVX2>
)


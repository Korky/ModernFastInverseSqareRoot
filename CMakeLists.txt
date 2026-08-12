cmake_minimum_required(VERSION 3.12)

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
add_test(NAME fisq_tests COMMAND fisq_tests)

# MSVC enables /RTC1 in Debug mode which conflicts with /O2 optimisation.
# The following compile options enable optimisations only for Release builds
# and disable runtime checks for Debug builds, avoiding the conflict.
target_compile_options(fisq
    PRIVATE
         # --- MSVC (cl.exe or clang-cl) ------------------------------------
        # Use AVX‑512 + FMA3 for *any* MSVC toolset.  If your clang‑cl
        # version is older than ~15.x, you can fall back to /QxHost.
        "$<$<CXX_COMPILER_ID:MSVC>:/arch:AVX512>"
        # (Optional) if you want the fallback:
        "$<$<AND:$<CXX_COMPILER_ID:MSVC>, $<NOT:$<TARGET_PROPERTY:HAS_AVX512>>>:/QxHost>"

        # --- GCC / Clang on Linux/macOS ----------------------------------
        "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-march=native>"
        # If you’re happy to relax FP semantics:
        # "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-ffast-math>"

        # --- Remove old SSE/AVX2 flags – they are implied by AVX512
        # (But if you need to keep them for other targets, just comment this out.)
        # "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-msse3>"
        # "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-mavx2>"
)


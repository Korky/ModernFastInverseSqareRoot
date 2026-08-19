cmake_minimum_required(VERSION 3.12)

project(ModernFastInverseSquareRoot LANGUAGES CXX)

# No external test framework is required; a lightweight custom test harness
# will be used instead.

# Build the core library with the fast inverse‑sqrt implementations.
add_library(fisq STATIC fast_inverse_sqrt.cpp)
target_include_directories(fisq PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_compile_features(fisq PUBLIC cxx_std_20)
# Benchmark executable
add_executable(benchmark benchmark.cpp)
target_link_libraries(benchmark PRIVATE fisq)

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
        # "$<$<OR:$<CXX_COMPILER_ID:GNU>,$<CXX_COMPILER_ID:Clang>>:-mavx512f>"
)

# -------------------------------------------------------------
# Enable CTest support
# -------------------------------------------------------------
enable_testing()
# -------------------------------------------------------------
# Compile the test executable
# -------------------------------------------------------------
add_executable(test_fast_inverse_sqrt
    tests/test_fast_inverse_sqrt.cpp   # <-- your file
)

# Link the library that defines the functions you call in the test.
target_link_libraries(test_fast_inverse_sqrt PRIVATE fisq)

# -------------------------------------------------------------
# Register the executable as a CTest unit
# -------------------------------------------------------------
add_test(
    NAME FastInverseSqrtTest          # any unique name you like
    COMMAND $<TARGET_FILE:test_fast_inverse_sqrt>
)

# Optional: make CTest stop on first failure or print output on failure.
# set_tests_properties(FastInverseSqrtTest PROPERTIES FAIL_REGULAR_EXPRESSION "Some tests failed.")

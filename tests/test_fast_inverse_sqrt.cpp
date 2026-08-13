// tests/test_fast_inverse_sqrt_new.cpp
//
// A lightweight test suite that validates the following functions:
//
//   • fisq::FastInverseSqrt(float)
//   • fisq::InverseSqrtSIMD(float)
//   • fisq::FastInverseSqrtSIMDBatch(const float*,float*,size_t)
//   • (conditionally) fisq::FastInverseSqrtAVX2Batch
//   • (conditionally) fisq::FastInverseSqrtAVX512Batch
//
// The suite compares every result against the exact value `1.0f / std::sqrt(x)`
// using a relative‑error tolerance of 0.2 % (2e-3).  This is generous enough
// for all three implementations while still catching gross inaccuracies.
#include "fast_inverse_sqrt.hpp"

#include <array>
#include <cmath>
#include <cstddef>   // size_t
#include <iostream>
#include <limits>
#include <vector>

int main()
{
    constexpr float tolerance = 2e-3f;      // 0.2 % relative error
    bool ok = true;

    /* ------------------------------------------------------------------ */
    /* 1️⃣  Single‑value tests – scalar implementations                  */
    /* ------------------------------------------------------------------ */

    const std::array<float,12> test_values = {
        1e-30f,
        1e-10f,
        1e-5f,
        0.001f,
        0.01f,
        0.1f,
        1.f,
        10.f,
        100.f,
        1'000.f,
        1'000'000.f,
        std::numeric_limits<float>::max()
    };

    for (float val : test_values) {
        const float exact = 1.0f / std::sqrt(val);

        /* FastInverseSqrt */
        const float q3   = fisq::FastInverseSqrt(val);
        if (std::abs(q3 - exact) / std::abs(exact) > tolerance) {
            std::cerr << "FastInverseSqrt failed for value=" << val
                      << ", got=" << q3 << ", expected=" << exact << '\n';
            ok = false;
        }

        /* InverseSqrtSIMD */
        const float simd = fisq::InverseSqrtSIMD(val);
        if (std::abs(simd - exact) / std::abs(exact) > tolerance) {
            std::cerr << "InverseSqrtSIMD failed for value=" << val
                      << ", got=" << simd << ", expected=" << exact << '\n';
            ok = false;
        }
    }

    /* ------------------------------------------------------------------ */
    /* 2️⃣  Batch tests – SIMD vectorised implementations                */
    /* ------------------------------------------------------------------ */

    // Helper to run a batch test and compare every element.
    auto run_batch_test = [&](auto batch_func, std::size_t n,
                              const char* func_name) {
        std::vector<float> src(n), dst(n);
        for (std::size_t i = 0; i < n; ++i)
            src[i] = test_values[i % test_values.size()];

        // Call the batch function
        batch_func(src.data(), dst.data(), n);

        // Validate each result
        for (std::size_t i = 0; i < n; ++i) {
            const float exact = 1.0f / std::sqrt(src[i]);
            if (std::abs(dst[i] - exact) / std::abs(exact) > tolerance) {
                std::cerr << func_name << " failed for index=" << i
                          << ", got=" << dst[i] << ", expected=" << exact << '\n';
                ok = false;
            }
        }
    };

    // SSE (4‑wide)
    run_batch_test(fisq::FastInverseSqrtSIMDBatch, 12,
                   "FastInverseSqrtSIMDBatch");

#if defined(__AVX__) && defined(__AVX2__)
    // AVX2 (8‑wide) – only available when the compiler targets AVX2
    run_batch_test(fisq::FastInverseSqrtAVX2Batch, 16,
                   "FastInverseSqrtAVX2Batch");
#endif

#if defined(__AVX512F__)
    // AVX‑512 (16‑wide) – only available with AVX‑512 support
    run_batch_test(fisq::FastInverseSqrtAVX512Batch, 32,
                   "FastInverseSqrtAVX512Batch");
#endif

    /* ------------------------------------------------------------------ */
    /* 3️⃣  Summary                                                       */
    /* ------------------------------------------------------------------ */

    if (ok) {
        std::cout << "All tests passed.\n";
        return 0;
    } else {
        std::cerr << "Some tests failed.\n";
        return 1;
    }
}

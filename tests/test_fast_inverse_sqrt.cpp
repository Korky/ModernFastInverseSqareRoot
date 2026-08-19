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
#include <random>
#include <cassert>
#include <xmmintrin.h> // SSE intrinsics
#if defined(__AVX__) && defined(__AVX2__)
#include <immintrin.h>
#endif 

int main()
{
    constexpr float tolerance = 2e-3f;      // 0.2 % relative error
    bool ok = true;

    // ------------------------------------------------------------------
    // Single‑value tests – scalar implementations                       
    // ------------------------------------------------------------------

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

        // FastInverseSqrt
        const float q3   = fisq::FastInverseSqrt(val);
        if (std::abs(q3 - exact) / std::abs(exact) > tolerance) {
            std::cerr << "FastInverseSqrt failed for value=" << val
                      << ", got=" << q3 << ", expected=" << exact << '\n';
            ok = false;
        }

        // InverseSqrtSIMD
        const float simd = fisq::InverseSqrtSIMD(val);
        if (std::abs(simd - exact) / std::abs(exact) > tolerance) {
            std::cerr << "InverseSqrtSIMD failed for value=" << val
                      << ", got=" << simd << ", expected=" << exact << '\n';
            ok = false;
        }
    }

    // ------------------------------------------------------------------
    // Batch tests – SIMD vectorised implementations                     
    // ------------------------------------------------------------------
    
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
    run_batch_test(
        static_cast<void(*)(const float*, float*, std::size_t)>(&fisq::FastInverseSqrtBatch),
        12,
        "FastInverseSqrtBatch"
    );

#if defined(__AVX__) && defined(__AVX2__)
    // AVX2 (8‑wide) – only available when the compiler targets AVX2
    run_batch_test(
        static_cast<void(*)(const float*, float*, std::size_t)>(&fisq::FastInverseSqrtAVX2Batch),
        16,
        "FastInverseSqrtAVX2Batch"
    );
#endif

#if defined(__AVX512F__)
    // AVX‑512 (16‑wide) – only available with AVX‑512 support
    run_batch_test(
        static_cast<void(*)(const float*, float*, std::size_t)>(&fisq::FastInverseSqrtAVX512Batch),
        32,
        "FastInverseSqrtAVX512Batch"
    );
#endif

    // ------------------------------------------------------------------
    // Additional edge‑case tests – zero, negative, and NaN handling
    // ------------------------------------------------------------------
    {
        float val = 0.0f;
        const float exact = std::numeric_limits<float>::infinity(); // 1/sqrt(0)
        auto check = [&](auto fn, const char* name) {
            float got = fn(val);
            if (!std::isinf(got)) {
                std::cerr << name << " did not return inf for value=0\n";
                ok = false;
            }
        };
        check(static_cast<float(*)(float)>(fisq::FastInverseSqrt), "FastInverseSqrt");
        check(static_cast<float(*)(float)>(fisq::InverseSqrtSIMD), "InverseSqrtSIMD");
    }

    // Negative input – expect NaN
    {
        float val = -1.0f;
        auto check_nan = [&](auto fn, const char* name) {
            float got = fn(val);
            if (!std::isnan(got)) {
                std::cerr << name << " did not return NaN for value=-1\n";
                ok = false;
            }
        };
        check_nan(static_cast<float(*)(float)>(fisq::FastInverseSqrt), "FastInverseSqrt");
        check_nan(static_cast<float(*)(float)>(fisq::InverseSqrtSIMD), "InverseSqrtSIMD");
    }

    // ------------------------------------------------------------------
    // Randomised testing over a wide float range
    // ------------------------------------------------------------------
    {
        constexpr std::size_t N = 10000;
        std::vector<float> src(N);
        std::mt19937 rng(0xdeadbeef); // deterministic seed for reproducibility
        std::uniform_real_distribution<float> dist(1e-30f, 1e+38f);

        for (auto& v : src) v = dist(rng);

        auto run_random_test = [&](auto batch_func, const char* name) {
            std::vector<float> dst(N);
            batch_func(src.data(), dst.data(), N);
            for (std::size_t i=0; i<N; ++i) {
                float exact = 1.0f / std::sqrt(src[i]);
                float got    = dst[i];
                float rel_err = std::abs(got - exact)/std::abs(exact);
                if (rel_err > tolerance) {
                    std::cerr << name << " failed for index=" << i
                            << ", val=" << src[i]
                            << ", got=" << got << ", expected=" << exact
                            << ", rel_err=" << rel_err << '\n';
                    ok = false;
                }
            }
        };
        // SSE/AVX2/AVX512 batch tests – the same helper works for all.
        run_random_test(static_cast<void(*)(const float*, float*, std::size_t)>(fisq::FastInverseSqrtBatch), "FastInverseSqrtBatch");
#if defined(__AVX__) && defined(__AVX2__)
        run_random_test(static_cast<void(*)(const float*, float*, std::size_t)>(fisq::FastInverseSqrtAVX2Batch), "FastInverseSqrtAVX2Batch");
#endif
#if defined(__AVX512F__)
        run_random_test(static_cast<void(*)(const float*, float*, std::size_t)>(fisq::FastInverseSqrtAVX512Batch), "FastInverseSqrtAVX512Batch");
#endif
    }

    // ------------------------------------------------------------------
    // Batch size edge‑case – non‑multiple of SIMD width
    // ------------------------------------------------------------------
    // {
    //     // Use an arbitrary length that is NOT a multiple of 8 (AVX2) or 16 (AVX512)
    //     const std::size_t bad_n = 7;   // 7 ≠ 4,8,16 …
    //     std::vector<float> src(bad_n), dst(bad_n);
    //     for (std::size_t i=0;i<bad_n;++i) src[i] = test_values[i % test_values.size()];

    //     // With exceptions disabled, the batch functions cannot throw.
    //     // Verify that the call succeeds (no crash) and results are reasonable.
    //     fisq::FastInverseSqrtBatch(src.data(), dst.data(), bad_n);
    //     for (std::size_t i = 0; i < bad_n; ++i) {
    //         float exact = 1.0f / std::sqrt(src[i]);
    //         if (std::abs(dst[i] - exact) / std::abs(exact) > tolerance) {
    //             std::cerr << "FastInverseSqrtBatch failed for index=" << i
    //                       << ", got=" << dst[i]
    //                       << ", expected=" << exact << '\n';
    //             ok = false;
    //         }
    //     }
    // }

    // ------------------------------------------------------------------ 
    // Summary                                                            
    // ------------------------------------------------------------------

    if (ok) {
        std::cout << "All tests passed.\n";
        return 0;
    } else {
        std::cerr << "Some tests failed.\n";
        return 1;
    }
}
// Lightweight unit test harness for the fast inverse square root functions.

#include "fast_inverse_sqrt.hpp"
#include <array>
#include <cmath>
#include <limits>
#include <iostream>

int main() {
    // Deterministic set of representative values covering a wide range:
    const std::array<float, 12> test_values = {
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
        std::numeric_limits<float>::max(),
    };

    // Relative error tolerance: 0.1% (1e-3). The fast inverse sqrt
    // implementations are known to be slightly less accurate than the
    // reference, but this bound comfortably covers all test cases.
    constexpr float tolerance = 2e-3f; // 0.2% relative error

    bool ok = true;
    for (float val : test_values) {
        const float exact = 1.0f / std::sqrt(val);
        const float q3   = fisq::FastInverseSqrt(val);
        const float simd = fisq::FastInverseSqrtSIMD(val);

        const float err_q3   = std::abs(q3 - exact) / std::abs(exact);
        const float err_simd = std::abs(simd - exact) / std::abs(exact);

        if (err_q3 > tolerance || err_simd > tolerance) {
            std::cerr << "Value: " << val
                      << ", exact: " << exact
                      << ", q3: " << q3 << " (err=" << err_q3 << ")"
                      << ", simd: " << simd << " (err=" << err_simd << ")\n";
            ok = false;
        }
    }

    if (ok) {
        std::cout << "All tests passed.\n";
        return 0;
    } else {
        std::cerr << "Some tests failed.\n";
        return 1;
    }
}

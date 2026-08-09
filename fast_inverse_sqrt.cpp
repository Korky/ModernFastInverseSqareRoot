// fast_inverse_sqrt.cpp
// Implementation of the fast inverse‑sqrt algorithms.

#include "fast_inverse_sqrt.hpp"
#include <xmmintrin.h> // SSE intrinsics

namespace fisq {

template <std::floating_point T>
constexpr T FastInverseSqrt(T number) {
    static_assert(sizeof(T) == 4, "FastInverseSqrt only supports 32‑bit floats");
    constexpr T threehalfs = static_cast<T>(1.5);

    T x2 = number * static_cast<T>(0.5);
    T y = number;

    // Reinterpret float bits as int
    uint32_t i = std::bit_cast<uint32_t>(y);
    i = 0x5f3759df - (i >> 1);  // magic number and bit shift

    y = std::bit_cast<T>(i);
    // One iteration of Newton‑Raphson
    y = y * (threehalfs - (x2 * y * y));
    return y;
}

// Explicit instantiation for float
template constexpr float FastInverseSqrt<float>(float);

[[nodiscard]] inline float FastInverseSqrtSIMD(float number) {
    __m128 input = _mm_set_ss(number);              // Set the input as a single float
    __m128 approx = _mm_rsqrt_ss(input);            // Fast approximate inverse sqrt

    // Newton‑Raphson refinement for improved precision
    const __m128 half  = _mm_set_ss(0.5f);
    const __m128 three = _mm_set_ss(1.5f);
    __m128 number_half = _mm_mul_ss(input, half);
    __m128 approx_sq   = _mm_mul_ss(approx, approx);
    __m128 mult        = _mm_mul_ss(number_half, approx_sq);
    __m128 nr          = _mm_sub_ss(three, mult);
    __m128 refined     = _mm_mul_ss(approx, nr);

    float result;
    _mm_store_ss(&result, refined);
    return result;
}

} // namespace fisq

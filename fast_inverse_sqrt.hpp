// fast_inverse_sqrt.hpp
// Header for the Quake‑3 fast inverse‑square‑root and its SIMD counterpart.

#pragma once

#include <bit>
#include <cstdint>
#include <concepts>

namespace fisq {

/// 
/// Fast inverse square root for 32‑bit floating point numbers.
///
/// @tparam T must be a floating‑point type with size 4 bytes (i.e. `float`).
/// @param number The value to compute the inverse sqrt of.
/// @return Approximate 1/sqrt(number).
///
template <std::floating_point T>
inline constexpr T FastInverseSqrt(T number) {
    static_assert(sizeof(T) == 4, "FastInverseSqrt only supports 32-bit floats");
    constexpr T threehalfs = static_cast<T>(1.5);

    T x2 = number * static_cast<T>(0.5);
    T y = number;

    // Reinterpret float bits as int
    uint32_t i = std::bit_cast<uint32_t>(y);
    i = 0x5f3759df - (i >> 1);  // magic number and bit shift

    y = std::bit_cast<T>(i);
    // One iteration of Newton–Raphson
    y = y * (threehalfs - (x2 * y * y));
    return y;
}

/// SIMD accelerated fast inverse square root using SSE intrinsics.
/// 
[[nodiscard]] float FastInverseSqrtSIMD(float number);
// The AVX2 implementation is only available when the compiler supports
// the corresponding instruction set.  Guard the declaration to avoid
// unresolved symbols on targets that cannot compile it.
#if defined(__AVX__) && defined(__AVX2__)
[[nodiscard]] float FastInverseSqrtAVX2(float number);
#endif

} // namespace fisq

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
constexpr T FastInverseSqrt(T number);

/// SIMD accelerated fast inverse square root using SSE intrinsics.
/// 
[[nodiscard]] inline float FastInverseSqrtSIMD(float number);

} // namespace fisq

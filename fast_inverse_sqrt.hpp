// fast_inverse_sqrt.hpp
// Header for the Quake‑3 fast inverse‑square‑root and its SIMD counterpart.

#pragma once

#include <bit>
#include <cstdint>
#include <concepts>

// Namespace for fast inverse square root implementations.
// The namespace encapsulates the functions to avoid polluting the global namespace.
// The functions are designed to be efficient and leverage SIMD instructions when available.
// The implementations are based on the original Quake III Arena algorithm, with improvements
// for modern compilers and architectures.
namespace fisq {

    /// Fast inverse square root using Newton–Raphson method for 32‑bit floating point numbers.
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

    /// SIMD accelerated inverse square root using architecture-specific SIMD intrinsics directly.
    ///
    /// @param number The value to compute the inverse sqrt of.
    /// @return Approximate 1/sqrt(number).
    [[nodiscard]] float InverseSqrtSIMD(float number);

    /// SIMD accelerated fast inverse square root using Newton–Raphson method with SSE intrinsics.
    ///
    /// @param src Pointer to input array of floats.
    /// @param dst Pointer to output array of floats (must be preallocated).
    /// @param n Number of elements to process (must be a multiple of 4).
    void FastInverseSqrtSIMDBatch(const float* src, float* dst, size_t n);
    
    // The AVX2 & AVX512 implementation is only available when the compiler supports
    // the corresponding instruction set.  Guard the declaration to avoid
    // unresolved symbols on targets that cannot compile it.
#if defined(__AVX__) && defined(__AVX2__)
    /// AVX2 accelerated fast inverse square root for batch processing using Newton–Raphson method with AVX2 intrinsics.
    ///
    /// @param src Pointer to input array of floats.
    /// @param dst Pointer to output array of floats (must be preallocated).
    /// @param n Number of elements to process (must be a multiple of 8).
    void FastInverseSqrtAVX2Batch(const float* src, float* dst, size_t n);
#endif
#if defined(__AVX512F__)
    /// AVX512 accelerated fast inverse square root for batch processing using Newton–Raphson method with AVX512 intrinsics.
    ///
    /// @param src Pointer to input array of floats.
    /// @param dst Pointer to output array of floats (must be preallocated).
    /// @param n Number of elements to process (must be a multiple of 16).
    void FastInverseSqrtAVX512Batch(const float* src, float* dst, size_t n);
#endif

} // namespace fisq

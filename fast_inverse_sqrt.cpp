// fast_inverse_sqrt.cpp
// Implementation of the fast inverse‑sqrt algorithms.

#include "fast_inverse_sqrt.hpp"
// SSE and AVX intrinsics are used conditionally. The header is always
// available on x86_64 compilers, but the implementation will only use it
// when the corresponding compiler flags enable the instruction set.
// Intrinsic headers – included before the namespace to avoid namespaced
// declarations of global intrinsics.
#include <xmmintrin.h> // SSE intrinsics (SSE3)
#if defined(__AVX__) && defined(__AVX2__)
#include <immintrin.h>
#endif

namespace fisq {

    [[nodiscard]] float FastInverseSqrtSIMD(float number) {
        __m128 input = _mm_set_ss(number);              // Set the input as a single float
        __m128 approx = _mm_rsqrt_ss(input);            // Fast approximate inverse sqrt

        // Newton‑Raphson refinement for improved precision
        static const __m128 half  = _mm_set_ss(0.5f);
        static const __m128 three = _mm_set_ss(1.5f);
        __m128 number_half = _mm_mul_ss(input, half);
        __m128 approx_sq   = _mm_mul_ss(approx, approx);

        // Newton‑Raphson: y * (1.5 - 0.5*n*y^2)
        __m128 mult        = _mm_mul_ss(number_half, approx_sq);
        __m128 nr          = _mm_sub_ss(three, mult);
        __m128 refined     = _mm_mul_ss(approx, nr);

        float result = _mm_cvtss_f32(refined);
        return result;
    }

    [[nodiscard]] float InverseSqrtSIMD(float number) {
        __m128 reg = _mm_set_ss(number);
        reg = _mm_rsqrt_ss(reg); // Executes hardware-level 1/sqrt
        return _mm_cvtss_f32(reg);
    }

} // namespace fisq

    // ------------------------------------------------------------------
    // AVX2 implementation – only compiled when the target supports AVX2.
    // ------------------------------------------------------------------
#if defined(__AVX__) && defined(__AVX2__)

namespace fisq {

    [[nodiscard]] float FastInverseSqrtAVX2(float number) {
        // Broadcast the input to all lanes of a 256‑bit vector.
        __m256 input = _mm256_set1_ps(number);
        __m256 approx = _mm256_rsqrt_ps(input);

        static const __m256 half  = _mm256_set1_ps(0.5f);
        static const __m256 three = _mm256_set1_ps(1.5f);
        __m256 number_half = _mm256_mul_ps(input, half);
        __m256 approx_sq   = _mm256_mul_ps(approx, approx);
        __m256 mult        = _mm256_mul_ps(number_half, approx_sq);
        __m256 nr          = _mm256_sub_ps(three, mult);
        __m256 refined     = _mm256_mul_ps(approx, nr);

        // Store the result to a stack array and return the first element.
        alignas(32) float res[8];
        _mm256_store_ps(res, refined);
        return res[0];
    }

} // namespace fisq
#endif

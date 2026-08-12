// fast_inverse_sqrt.cpp – Fast inverse square root implementations using SIMD intrinsics.

#include "fast_inverse_sqrt.hpp"
// Include SIMD intrinsic headers; conditionally compiled based on target architecture.
#include <xmmintrin.h> // SSE intrinsics
#if defined(__AVX__) && defined(__AVX2__)
#include <immintrin.h>
#endif

namespace fisq {

    [[nodiscard]] float FastInverseSqrtSIMD(float number) {
        const __m128 input = _mm_set_ss(number);              // Broadcast scalar to XMM register
        const __m128 approx = _mm_rsqrt_ss(input);            // Approximate 1/√x using hardware instruction

        static const __m128 half  = _mm_set_ss(0.5f);      // Half value for Newton–Raphson
        static const __m128 three = _mm_set_ss(1.5f);      // Three‑half constant

        // Newton–Raphson refinement: y * (3 – n*y²) / 2
        __m128 number_half = _mm_mul_ss(input, half);
        __m128 approx_sq   = _mm_mul_ss(approx, approx);
        __m128 mult        = _mm_mul_ss(number_half, approx_sq);
        __m128 nr          = _mm_sub_ss(three, mult);
        __m128 refined     = _mm_mul_ss(approx, nr);

        float result = _mm_cvtss_f32(refined);
        return result;
    }
    [[nodiscard]] inline __m128 FastInverseSqrtSIMDBatch(__m128 number) {

        const __m128 approx = _mm_rsqrt_ss(number); // Approximate 1/√x using hardware instruction

        static const __m128 half  = _mm_set_ss(0.5f);      // Half value for Newton–Raphson
        static const __m128 three = _mm_set_ss(1.5f);      // Three‑half constant

        // Newton–Raphson refinement: y * (3 – n*y²) / 2
        __m128 number_half = _mm_mul_ss(number, half);
        __m128 approx_sq   = _mm_mul_ss(approx, approx);
        __m128 mult        = _mm_mul_ss(number_half, approx_sq);
        __m128 nr          = _mm_sub_ss(three, mult);
        return _mm_mul_ss(approx, nr);
    }
    inline void FastInverseSqrtSIMDBatch(const float* src, float* dst, size_t n) { // n must be a multiple of 4
        for (size_t i = 0; i < n; i += 4) {
            const __m128 input  = _mm_loadu_ps(src + i); // Load 4 floats
            const __m128 result = FastInverseSqrtSIMDBatch(input);
            _mm_storeu_ps(dst + i, result); // Store 4 results
        }
    }
    [[nodiscard]] float InverseSqrtSIMD(float number) {
        __m128 reg = _mm_set_ss(number);
        reg = _mm_rsqrt_ss(reg);   // Hardware reciprocal sqrt (approximate)
        return _mm_cvtss_f32(reg);
    }

} // namespace fisq

    // ------------------------------------------------------------------
    // AVX2 implementation – only compiled when the target supports AVX2.
    // ------------------------------------------------------------------
#if defined(__AVX__) && defined(__AVX2__)

namespace fisq {

    [[nodiscard]] float FastInverseSqrtAVX2(float number) {
        // Broadcast scalar to all lanes of an AVX2 (256‑bit) vector.
        __m256 input = _mm256_set1_ps(number);
        __m256 approx = _mm256_rsqrt_ps(input);

        static const __m256 half  = _mm256_set1_ps(0.5f);
        static const __m256 three = _mm256_set1_ps(1.5f);
        __m256 number_half = _mm256_mul_ps(input, half);
        __m256 approx_sq   = _mm256_mul_ps(approx, approx);
        __m256 mult        = _mm256_mul_ps(number_half, approx_sq);
        __m256 nr          = _mm256_sub_ps(three, mult);
        __m256 refined     = _mm256_mul_ps(approx, nr);

        // Store result in local array; return first lane.
        alignas(32) float res[8];
        _mm256_store_ps(res, refined);
        return res[0];
    }

    // Vectorised batch processing  elements with AVX2.
    [[nodiscard]] inline __m256 FastInverseSqrtAVX2Batch(__m256 input) {
        // Approximate inverse sqrt using rsqrt14.
        const __m256 approx = _mm256_rsqrt14_ps(input);

        static const __m256 half  = _mm256_set1_ps(0.5f);
        static const __m256 three = _mm256_set1_ps(1.5f);

        const __m256 approx_sq = _mm256_mul_ps(approx, approx); // y²
        const __m256 mult      = _mm256_fmadd_ps(input, approx_sq, half); // 0.5*x*y² (fused)
        const __m256 nr        = _mm256_sub_ps(three, mult); // 1.5 – 0.5*x*y²
        return _mm256_mul_ps(approx, nr); // y * (1.5 – 0.5*x*y²)
    }

    inline void FastInverseSqrtAVX2Batch(const float* src, float* dst, size_t n) // n must be a multiple of 8
    {
        for (size_t i = 0; i < n; i += 8)
        {
            const __m256 input  = _mm256_loadu_ps(src + i); // Load 8 floats
            const __m256 result = FastInverseSqrtAVX2Batch(input);
            _mm256_storeu_ps(dst + i, result); // Store 8 results
        }
    }

    
#if defined(__AVX512F__)
    [[nodiscard]] float FastInverseSqrtAVX512(float number) {
        // Broadcast scalar to a 512‑bit vector.
        const __m512 input = _mm512_set1_ps(number);

        // Rough estimate using rsqrt14 (≈28‑bit accuracy).
        const __m512 approx = _mm512_rsqrt14_ps(input); // approximate 1/√x

        static const __m512 half  = _mm512_set1_ps(0.5f);
        static const __m512 three = _mm512_set1_ps(1.5f);

        // Newton–Raphson refinement using fused multiply‑add.
        const __m512 approx_sq  = _mm512_mul_ps(approx, approx); // y²
        const __m512 mult       = _mm512_fmadd_ps(input, approx_sq, half); // 0.5*x*y²
        const __m512 nr         = _mm512_sub_ps(three, mult); // 1.5 – 0.5*x*y²
        const __m512 refined    = _mm512_mul_ps(approx, nr); // y * (1.5 – 0.5*x*y²)

        // Extract first lane of result.
        return _mm_cvtss_f32(_mm512_extractf32x4_ps(refined, 0));
    }

    // Vectorised batch processing 16 elements with AVX‑512.
    [[nodiscard]] inline __m512 FastInverseSqrtAVX512Batch(__m512 input) {
        // Approximate inverse sqrt using rsqrt14.
        const __m512 approx = _mm512_rsqrt14_ps(input);

        static const __m512 half  = _mm512_set1_ps(0.5f);
        static const __m512 three = _mm512_set1_ps(1.5f);

        const __m512 approx_sq = _mm512_mul_ps(approx, approx); // y²
        const __m512 mult      = _mm512_fmadd_ps(input, approx_sq, half); // 0.5*x*y² (fused)
        const __m512 nr        = _mm512_sub_ps(three, mult); // 1.5 – 0.5*x*y²
        return _mm512_mul_ps(approx, nr); // y * (1.5 – 0.5*x*y²)
    }

    inline void FastInverseSqrtAVX512Batch(const float* src, float* dst, size_t n) // n must be a multiple of 16
    {
        for (size_t i = 0; i < n; i += 16)
        {
            const __m512 input  = _mm512_loadu_ps(src + i); // Load 16 floats
            const __m512 result = FastInverseSqrtAVX512Batch(input);
            _mm512_storeu_ps(dst + i, result); // Store 16 results
        }
    }
#endif // __AVX512F__
} // namespace fisq
#endif // __AVX__ && __AVX2__

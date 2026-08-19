// fast_inverse_sqrt.cpp – Fast inverse square root implementations using SIMD intrinsics.

#include "fast_inverse_sqrt.hpp"
// Include SIMD intrinsic headers; conditionally compiled based on target architecture.
#include <xmmintrin.h> // SSE intrinsics
#if defined(__AVX__) && defined(__AVX2__)
#include <immintrin.h>
#endif

namespace fisq {
    
    [[nodiscard]] float InverseSqrtSIMD(float number) {
        __m128 reg = _mm_set_ss(number);
        reg = _mm_rsqrt_ss(reg);   // Hardware reciprocal sqrt (approximate)
        return _mm_cvtss_f32(reg);
    }

    [[nodiscard]] inline __m128 InverseSqrtSIMDBatch(__m128 number) {
        return _mm_rsqrt_ps(number); // Approximate 1/√x using hardware instruction
    }

    [[nodiscard]] inline __m128 FastInverseSqrtSIMDBatch(__m128 number) {

        const __m128 approx = _mm_rsqrt_ps(number); // Approximate 1/√x using hardware instruction

        static const __m128 half  = _mm_set1_ps(0.5f);      // Half value for Newton–Raphson
        static const __m128 three = _mm_set1_ps(1.5f);      // Three‑half constant

        // Newton–Raphson refinement: y * (3 – n*y²) / 2
        const __m128 number_half = _mm_mul_ps(number, half);
        const __m128 approx_sq   = _mm_mul_ps(approx, approx);
        const __m128 newt_raph = _mm_fnmadd_ps(number_half, approx_sq, three);
        return _mm_mul_ps(approx, newt_raph);
    }

    template <> struct SIMDTraits<__m128> {
      static constexpr size_t width = 4;
      static __m128 load(const float *ptr) { return _mm_loadu_ps(ptr); }
      static void store(float *ptr, __m128 value) { _mm_storeu_ps(ptr, value); }
      static __m128 inverse_sqrt(__m128 value) {
        return FastInverseSqrtSIMDBatch(value);
      }
    };


    // Batch processing of inverse square root using SIMD intrinsics.
    // This function processes an array of floats in batches, leveraging SIMD instructions
    // for improved performance. The implementation is specialized for different SIMD widths
    // (SSE, AVX, AVX512) based on the target architecture.
    void FastInverseSqrtBatch(const float *src, float *dst, size_t n) { // n must be a multiple of 4
#if defined(__AVX512F__)
      FastInverseSqrtBatch<__m512>(src, dst, n);
#elif defined(__AVX__) && defined(__AVX2__)
      FastInverseSqrtBatch<__m256>(src, dst, n);
#else
      FastInverseSqrtBatch<__m128>(src, dst, n);
#endif
    }
} // namespace fisq


// ------------------------------------------------------------------
// AVX2 implementation – only compiled when the target supports AVX2.
// ------------------------------------------------------------------
#if defined(__AVX__) && defined(__AVX2__)

namespace fisq {

    // Vectorised batch processing  elements with AVX2.
    [[nodiscard]] inline __m256 FastInverseSqrtAVX2Batch(__m256 input) {
        // Approximate inverse sqrt using rsqrt14.
        const __m256 approx = _mm256_rsqrt14_ps(input);

        static const __m256 half  = _mm256_set1_ps(0.5f);
        static const __m256 three = _mm256_set1_ps(1.5f);

        const __m256 input_half  = _mm256_mul_ps(input, half);
        const __m256 approx_sq   = _mm256_mul_ps(approx, approx);
        const __m256 newt_raph = _mm256_fnmadd_ps(input_half, approx_sq, three);
        return _mm256_mul_ps(approx, newt_raph);
    }
    // Batch processing of inverse square root using AVX2 intrinsics.
    void FastInverseSqrtAVX2Batch(const float* src, float* dst, size_t n) // n must be a multiple of 8
    {
        for (size_t i = 0; i < n; i += 8)
        {
            const __m256 input  = _mm256_loadu_ps(src + i); // Load 8 floats
            const __m256 result = FastInverseSqrtAVX2Batch(input);
            _mm256_storeu_ps(dst + i, result); // Store 8 results
        }
    }
    // SIMD traits specialization for AVX2 (__m256).
    template <> struct SIMDTraits<__m256> {
      static constexpr size_t width = 8;
      static __m256 load(const float *ptr) { return _mm256_loadu_ps(ptr); }
      static void store(float *ptr, __m256 value) {
        _mm256_storeu_ps(ptr, value);
      }
      static __m256 inverse_sqrt(__m256 value) {
        return FastInverseSqrtAVX2Batch(value);
      }
    };

#if defined(__AVX512F__)
    // Vectorised batch processing 16 elements with AVX‑512.
    [[nodiscard]] inline __m512 FastInverseSqrtAVX512Batch(__m512 input) {
        // Approximate inverse sqrt using rsqrt14.
        const __m512 approx = _mm512_rsqrt14_ps(input);

        static const __m512 half  = _mm512_set1_ps(0.5f);
        static const __m512 three = _mm512_set1_ps(1.5f);

        const __m512 input_half = _mm512_mul_ps(input, half);
        const __m512 approx_sq  = _mm512_mul_ps(approx, approx); 
        const __m512 newt_raph = _mm512_fnmadd_ps(input_half, approx_sq, three);
        return _mm512_mul_ps(approx, newt_raph);
    }

    void FastInverseSqrtAVX512Batch(const float* src, float* dst, size_t n) // n must be a multiple of 16
    {
        for (size_t i = 0; i < n; i += 16)
        {
            const __m512 input  = _mm512_loadu_ps(src + i); // Load 16 floats
            const __m512 result = FastInverseSqrtAVX512Batch(input);
            _mm512_storeu_ps(dst + i, result); // Store 16 results
        }
    }
    template <> struct SIMDTraits<__m512> {
      static constexpr size_t width = 16;
      static __m512 load(const float *ptr) { return _mm512_loadu_ps(ptr); }
      static void store(float *ptr, __m512 value) {
        _mm512_storeu_ps(ptr, value);
      }
      static __m512 inverse_sqrt(__m512 value) {
        return FastInverseSqrtAVX512Batch(value);
      }
    };
#endif // __AVX512F__
} // namespace fisq
#endif // __AVX__ && __AVX2__


namespace fisq {


    

    

    

    
}
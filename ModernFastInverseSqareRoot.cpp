// ModernFastInverseSqareRoot.cpp : Defines the entry point for the application.
//
#include <bit>
#include <cstdint>
#include <concepts>
#include <type_traits>
#include <iostream>
#include <xmmintrin.h>  // SSE intrinsics
#include <iomanip>  // for precision
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

// Modern C++20 Quake 3 version
template<std::floating_point T>
T FastInverseSqrt(T number) {
	static_assert(sizeof(T) == 4, "FastInverseSqrt only supports 32-bit floats");

	constexpr T threehalfs = static_cast<T>(1.5);
	T x2 = number * static_cast<T>(0.5);
	T y = number;

	// Reinterpret float bits as int
	uint32_t i = std::bit_cast<uint32_t>(y);
	i = 0x5f3759df - (i >> 1);  // magic number and bit shift

	// Convert bits back to float
	y = std::bit_cast<T>(i);

	// One iteration of Newton-Raphson
	y = y * (threehalfs - (x2 * y * y));

	return y;
}

// SIMD Version
[[nodiscard]] inline float FastInvSqrtSIMD(float number) {
	__m128 input = _mm_set_ss(number);              // Set the input as a single float
	__m128 approx = _mm_rsqrt_ss(input);       // Fast approximate inverse sqrt

	// Optional: Newton-Raphson refinement for improved precision
	// y = y * (1.5 - (x * 0.5 * y * y));
	__m128 half = _mm_set_ss(0.5f);
	__m128 three = _mm_set_ss(1.5f);
	__m128 number_half = _mm_mul_ss(input, half);
	__m128 approx_sq = _mm_mul_ss(approx, approx);
	__m128 mult = _mm_mul_ss(number_half, approx_sq);
	__m128 nr = _mm_sub_ss(three, mult);
	__m128 refined = _mm_mul_ss(approx, nr);

	float result;
	_mm_store_ss(&result, refined);
	return result;
}

// std version
inline float InvSqrtStd(float x) {
	return 1.0f / std::sqrtf(x);
}

int main() {

	// Precision Test
	float number = 25.0f;
	float invSqrt = FastInverseSqrt(number);
	float invSqrtSMID = FastInvSqrtSIMD(number);
	float invSqrtStd = InvSqrtStd(number);

	std::cout << "Fast InvSqrt(" << number << ") approx " << invSqrt << "\n";
	std::cout << "SMID Fast InvSqrt(" << number << ") approx " << invSqrtSMID << "\n";
	std::cout << "Check: 1/sqrt(" << number << ") = " << invSqrtStd << "\n";


// ---------------------------------------------------------------------
// Benchmark each algorithm on the same input set.  The goal is to
// give every implementation a fair comparison by:
//   * Using the same input vector for all runs.
//   * Reusing pre‑allocated result buffers so allocation overhead is
//     not counted in timings.
//   * Measuring with steady_clock and keeping the loop body identical.
// ---------------------------------------------------------------------

constexpr size_t numSamples = 1'000'000; // number of random values per run
std::vector<float> inputs(numSamples);

// Fill with random positive floats – seed is fixed for reproducibility.
std::mt19937 rng(42);
std::uniform_real_distribution<float> dist(0.1f, 1000.0f);
for (auto& val : inputs)
    val = dist(rng);

// Allocate result buffers once; each algorithm writes to its own buffer.
std::vector<float> resultsQ3(numSamples), resultsSIMD(numSamples), resultsSTD(numSamples);

// Helper lambda that times a single algorithm and writes into the
// provided output vector.  The function pointer is kept simple for
// readability; we could template it but the small number of
// implementations keeps the code straightforward.
auto benchmark = [&](const char* /*name*/, float (*func)(float), std::vector<float>& out) {
    auto start = std::chrono::steady_clock::now();
    for (size_t i = 0; i < numSamples; ++i)
        out[i] = func(inputs[i]);
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
};

// Run each algorithm separately and collect times.
double simdTime = benchmark("SIMD", FastInvSqrtSIMD, resultsSIMD);
double q3Time   = benchmark("Quake3", FastInverseSqrt<float>, resultsQ3);
double stdTime  = benchmark("Std", InvSqrtStd, resultsSTD);

// Prevent compiler from eliminating the loops by using a checksum.
auto checksum = [](const std::vector<float>& v) {
    double sum = 0.0;
    for (float f : v)
        sum += static_cast<double>(f);
    return sum;
};
double simdSum = checksum(resultsSIMD);
double q3Sum   = checksum(resultsQ3);
double stdSum  = checksum(resultsSTD);

	std::cout << std::fixed << std::setprecision(4);
	std::cout << "SIMD Time:\t\t" << simdTime << " ms\n";
	std::cout << "Quake3 Time:\t\t" << q3Time << " ms\n";
    std::cout << "std::sqrt Time:\t\t" << stdTime << " ms\n";

    // Print checksums to keep the compiler from optimizing away the loops
    std::cout << "Checksums -> SIMD: " << simdSum
              << ", Quake3: " << q3Sum
              << ", Std: " << stdSum << '\n';

    // Compute error metrics relative to the standard implementation.
    auto compute_error = [&](const std::vector<float>& approx) {
        double max_err = 0.0;
        double sum_err = 0.0;
        for (size_t i = 0; i < numSamples; ++i) {
            double err = std::abs(approx[i] - resultsSTD[i]);
            sum_err += err;
            if (err > max_err)
                max_err = err;
        }
        return std::make_pair(sum_err / numSamples, max_err);
    };

    auto [simdMeanErr, simdMaxErr] = compute_error(resultsSIMD);
    auto [q3MeanErr, q3MaxErr]     = compute_error(resultsQ3);
    auto [stdMeanErr, stdMaxErr]   = compute_error(resultsSTD); // zero

    std::cout << "Avg error -> SIMD: " << simdMeanErr
              << ", Quake3: " << q3MeanErr
              << ", Std: " << stdMeanErr << '\n';
    std::cout << "Max error -> SIMD: " << simdMaxErr
              << ", Quake3: " << q3MaxErr
              << ", Std: " << stdMaxErr << '\n';

    return 0;
}


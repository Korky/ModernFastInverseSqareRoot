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


	// Speed Test
	constexpr size_t numSamples = 1'000'000;
	std::vector<float> inputs(numSamples);

	// Fill with random positive floats
	std::mt19937 rng(42);
	std::uniform_real_distribution<float> dist(0.1f, 1000.0f);
	for (auto& val : inputs)
		val = dist(rng);

	std::vector<float> resultsQ3(numSamples);
	std::vector<float> resultsSIMD(numSamples);
	std::vector<float> resultsSTD(numSamples);


	// Measure SIMD time
	auto t1 = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < numSamples; ++i) {
		resultsSIMD[i] = FastInvSqrtSIMD(inputs[i]);
	}
	auto t2 = std::chrono::high_resolution_clock::now();

	// Measure q3 sqrt time
	auto t3 = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < numSamples; ++i) {
		resultsQ3[i] = FastInverseSqrt(inputs[i]);
	}
	auto t4 = std::chrono::high_resolution_clock::now();
	
	// Measure std sqrt time
	auto t5 = std::chrono::high_resolution_clock::now();
	for (size_t i = 0; i < numSamples; ++i) {
		resultsSTD[i] = InvSqrtStd(inputs[i]);
	}
	auto t6 = std::chrono::high_resolution_clock::now();


	auto simdTime = std::chrono::duration<double, std::milli>(t2 - t1).count();
	auto q3Time = std::chrono::duration<double, std::milli>(t4 - t3).count();
	auto stdTime = std::chrono::duration<double, std::milli>(t6 - t5).count();

	std::cout << std::fixed << std::setprecision(4);
	std::cout << "SIMD Time:\t\t" << simdTime << " ms\n";
	std::cout << "Quake3 Time:\t\t" << q3Time << " ms\n";
	std::cout << "std::sqrt Time:\t\t" << stdTime << " ms\n";


	return 0;
}


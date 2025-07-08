// ModernFastInverseSqareRoot.cpp : Defines the entry point for the application.
//

#include "ModernFastInverseSqareRoot.h"
#include <bit>
#include <cstdint>
#include <concepts>
#include <type_traits>
#include <iostream>

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

int main() {
	float number = 25.0f;
	float invSqrt = FastInverseSqrt(number);

	std::cout << "Fast InvSqrt(" << number << ") ≈ " << invSqrt << "\n";
	std::cout << "Check: 1/sqrt(" << number << ") = " << 1.0f / std::sqrt(number) << "\n";
	return 0;
}


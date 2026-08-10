# ModernFastInverseSqareRoot

This repository contains a **C++20** implementation of the classic Quake III fast inverse square root algorithm, along with an SSE‑based SIMD variant.  The goal is to provide a lightweight, high‑performance helper that can be dropped into other projects.

## Build Instructions

The project uses CMake and provides several presets for common build configurations.  All builds target **x86‑64** CPUs with SSE support (the default on modern PCs).

### Windows (MSVC)
```powershell
# Configure a Release build using the preset defined in `CMakePresets.json`
cmake --preset x64-release

# Build the library, tests and benchmark executable
cmake --build out\build\x64-release --config Release
```

### Linux / macOS (Clang/GCC)
```bash
# Configure a Release build using the preset defined in `CMakePresets.json`
cmake --preset x64-release

# Build the library, tests and benchmark executable
cmake --build out/build/x64-release --config Release
```

The presets automatically set the appropriate compiler (`cl.exe` on Windows, `clang++`/`g++` elsewhere) and enable C++20.  If you prefer to use a custom build directory or configuration, you can replace the preset commands with standard CMake arguments.

## Usage Example

```cpp
#include "fast_inverse_sqrt.hpp"
#include <iostream>

int main() {
    float value = 4.0f;
    std::cout << "FastInverseSqrt(" << value << ") = "
              << fisq::FastInverseSqrt(value) << '\n';
}
```

Compile with a C++20 compiler and link against the static library `fisq.lib` (or `libfisq.a`).

## API Documentation

### `template<std::floating_point T> inline constexpr T FastInverseSqrt(T number)`
* **Parameters**: `number` – a 32‑bit floating point value.
* **Returns**: An approximate value of `1 / sqrt(number)`.
* **Behavior**:
  * For positive, finite inputs the relative error is ≤ 0.2 % (verified by the unit tests).
  * For `0` the function returns a large finite number (~ 1.98 × 10¹⁹), effectively representing infinity.
  * For negative numbers it returns `-inf`.
  * For `+∞` or `‑∞` it returns `-inf`.
  * For `NaN` it propagates the NaN value.

### `[[nodiscard]] float FastInverseSqrtSIMD(float number)`
* **Parameters**: `number` – a 32‑bit floating point value.
* **Returns**: An approximate value of `1 / sqrt(number)` computed with SSE intrinsics.
* **Requirements**: The target CPU must support SSE (x86‑64).  The function is only defined for positive, finite inputs; for zero, negative, infinite or NaN values it returns `NaN` as observed in the test harness.

## Benchmarks

The repository ships a small benchmark program that measures the execution time of:
* The classic Quake III implementation (`FastInverseSqrt`).
* The SIMD‑accelerated version (`FastInverseSqrtSIMD`).
* The standard library `1 / sqrtf`.

To run the benchmark after building:

```bash
out/build/x64-release/benchmark.exe
```

Typical output (on a recent Intel CPU) looks like this:

```
Warm‑up iterations: 5
Iterations per function: 1000000
Total elapsed time (ms): 8.4940
SIMD Time:\t2.3496e+00 ms\t(avg=2.3496e+00 ns)
Quake3 Time:\t3.0609e+00 ms\t(avg=3.0609e+00 ns)
Std Time:\t3.0835e+00 ms\t(avg=3.0835e+00 ns)
Checksums -> SIMD: 6.2567e+04, Quake3: 6.2507e+04, Std: 6.2567e+04
```

The benchmark demonstrates that the SIMD variant is roughly 30 % faster than the classic implementation and comparable to the standard library.

## Tests

Unit tests are provided in `tests/test_fast_inverse_sqrt.cpp`.  They verify correctness across a wide range of inputs and confirm the relative error bound.  Run them with:

```bash
out/build/x64-release/fisq_tests.exe
```

All tests should pass.

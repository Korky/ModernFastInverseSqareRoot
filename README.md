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

The presets automatically set the appropriate compiler (`clang-cl.exe` on Windows, `clang++`/`g++` elsewhere) and enable C++20.  If you prefer to use a custom build directory or configuration, you can replace the preset commands with standard CMake arguments.

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

The repository ships a small benchmark program that measures the execution **time** and, starting with this update, also prints detailed error metrics for each fast‑inverse‑sqrt implementation.
* The classic Quake III implementation (`FastInverseSqrt`).
* The SIMD‑accelerated version (`FastInverseSqrtSIMD`).
* The standard library `1 / sqrtf`.


```bash
out/build/x64-release/benchmark.exe
```
Typical output (on a recent Intel CPU) looks like this, including timing and error metrics:

```
Warm-up iterations: 5
Iterations per function: 1000000
Total elapsed time (ms): 2.1601

Time
Q3(C++20):      4.3970e-01 ms   (avg=4.3970e-01 ns)
Q3(SIMD):       2.9590e-01 ms   (avg=2.9590e-01 ns)
RSqrt(SIMD):    1.0796e+00 ms   (avg=1.0796e+00 ns)
Std:            1.4245e+00 ms

Error metrics
                mean_abs,               max_abs,                mean_rel
Q3(C++20):      0x1.f2fc97efe0ce1p-15,  0x1.47ec000000000p-8,   0x1.f82dc4cee149ap-11
Q3(SIMD):       0x1.3e75bc44bf4cbp-29,  0x1.0000000000000p-21,  0x1.39df740d6494dp-25
RSqrt(SIMD):    0x1.91c80a2877ee5p-18,  0x1.7a60000000000p-11,  0x1.8ded4c7a47243p-14

Checksums -> Q3 (C++20): 0x1.e85653c9d2000p+15, Q3(SIMD): 0x1.e8cd4b2584400p+15, RSqrt(SIMD): 0x1.e8cd509700000p+15, Std: 0x1.e8cd4b8475800p+15
```
or with AVX
```
Warm-up iterations: 5
Iterations per function: 1000000
Total elapsed time (ms): 2.4103

Time
Q3(C++20):      4.3460e-01 ms   (avg=4.3460e-01 ns)
Q3(SIMD):       3.1820e-01 ms   (avg=3.1820e-01 ns)
Q3(AVX512):     3.0080e-01 ms   (avg=3.0080e-01 ns)
Q3(AVX2):       2.9710e-01 ms   (avg=2.9710e-01 ns)
RSqrt(SIMD):    1.1238e+00 ms   (avg=1.1238e+00 ns)
Std:            1.3604e+00 ms

Error metrics
                mean_abs,               max_abs,                mean_rel
Q3(C++20):      0x1.f2fc99f559b3dp-15,  0x1.47ec000000000p-8,   0x1.f82dc8846a928p-11
Q3(SIMD):       0x1.3edbb59ddc1e8p-29,  0x1.0000000000000p-21,  0x1.3a0c87d5ce9e6p-25
Q3(AVX512):     0x1.fe392e1ef73c1p-30,  0x1.0000000000000p-22,  0x1.f6c5412b8305ep-26
Q3(AVX2):       0x1.fe392e1ef73c1p-30,  0x1.0000000000000p-22,  0x1.f6c5412b8305ep-26
RSqrt(SIMD):    0x1.91c5b673c4f3cp-18,  0x1.7a60000000000p-11,  0x1.8decd52429cb0p-14

Checksums -> Q3 (C++20): 0x1.e85653b85bc00p+15, Q3(SIMD): 0x1.e8cd4b14cba00p+15, RSqrt(SIMD): 0x1.e8cd50d000000p+15, Q3(AVX512): 0x1.e8cd4b6c63c00p+15, Q3(AVX2): 0x1.e8cd4b6c63c00p+15, Std: 0x1.e8cd4b7383a00p+15
```

The benchmark demonstrates that the SIMD variant is roughly 30 % faster than the classic implementation and comparable to the standard library.

## Tests

Unit tests are provided in `tests/test_fast_inverse_sqrt.cpp`.  They verify correctness across a wide range of inputs and confirm the relative error bound.  Run them with:

```bash
out/build/x64-release/fisq_tests.exe
```

All tests should pass.

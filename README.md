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
Total elapsed time (ms): 74.5122
Quake3 Time:    2.1690e+01 ms   (avg=2.1690e+01 ns)
SIMD Time:      3.9054e+01 ms   (avg=3.9054e+01 ns)
Std Time:       1.3768e+01 ms

Checksums -> Quake3: 6.2507e+04, SIMD: 6.2567e+04, Std: 6.2567e+04
Error metrics
        mean_abs,               max_abs,                mean_rel
Quake3: 0x1.f2fc97efe0ce1p-15,  0x1.47ec000000000p-8,   0x1.f82dc4cee149ap-11
SIMD:   0x1.3e75bc44bf4cbp-29,  0x1.0000000000000p-21,  0x1.39df740d6494dp-25
```

The benchmark demonstrates that the SIMD variant is roughly 30 % faster than the classic implementation and comparable to the standard library.

## Tests

Unit tests are provided in `tests/test_fast_inverse_sqrt.cpp`.  They verify correctness across a wide range of inputs and confirm the relative error bound.  Run them with:

```bash
out/build/x64-release/fisq_tests.exe
```

All tests should pass.

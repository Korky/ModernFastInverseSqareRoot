#include "fast_inverse_sqrt.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <tuple>
#include <chrono>

int main() {
    constexpr size_t numSamples = 1'000'000;
    constexpr int warmupCount = 5;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.1f, 1000.f);

    // Generate inputs
    std::vector<float> inputs(numSamples);
    for (auto &v : inputs) v = dist(rng);

    std::vector<float> q3(numSamples), simd(numSamples), avx(numSamples), stdv(numSamples);

    auto bench = [&](float (*func)(float), std::vector<float> &out) {
        volatile float dummy = 0.f;
        for (int w = 0; w < warmupCount; ++w)
            for (size_t i = 0; i < numSamples; ++i) dummy += func(inputs[i]);

        auto start = std::chrono::steady_clock::now();
        for (size_t i = 0; i < numSamples; ++i) out[i] = func(inputs[i]);
        return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
    };

    double tQ3   = bench(fisq::FastInverseSqrt<float>, q3);
    double tSIMD = bench(fisq::FastInverseSqrtSIMD, simd);

#if defined(__AVX__) && defined(__AVX2__)
    double tAVX  = bench(fisq::FastInverseSqrtAVX2, avx);
#endif

    // Reference std::sqrt
    double tStd   = bench([](float x) { return 1.0f / std::sqrtf(x); }, stdv);

    auto checksum = [](const std::vector<float> &v) {
        double s = 0;
        for (auto f : v)
            s += f;
        return s;
    };

    // Error calculation lambda
    auto compute_error = [&](const std::vector<float> &fast, const std::vector<float> &ref) {
        double abs_err_sum = 0.0, rel_err_sum = 0.0;
        float max_abs = 0.0f;
        for (size_t i = 0; i < fast.size(); ++i) {
            float diff = fast[i] - ref[i];
            float abs_diff = std::abs(diff);
            abs_err_sum += abs_diff;
            if (abs_diff > max_abs)
                max_abs = abs_diff;
            rel_err_sum += std::abs(diff / ref[i]);
        }
        return std::make_tuple(abs_err_sum / fast.size(), max_abs, rel_err_sum / fast.size());
    };

    auto [q3_mean_abs, q3_max_abs, q3_mean_rel] = compute_error(q3, stdv);
    auto [simd_mean_abs, simd_max_abs, simd_mean_rel] = compute_error(simd, stdv);
#if defined(__AVX__) && defined(__AVX2__)
    auto [avx_mean_abs, avx_max_abs, avx_mean_rel] = compute_error(avx, stdv);
#endif

    std::cout.setf(std::ios::fixed); std::cout << std::setprecision(4);
    std::cout << "Warm-up iterations: " << warmupCount << '\n';
    std::cout << "Iterations per function: " << numSamples << '\n';

    double totalTime = tQ3 + tSIMD;
#if defined(__AVX__) && defined(__AVX2__)
    totalTime += tAVX;
#endif
    totalTime += tStd;

    std::cout << "Total elapsed time (ms): " << totalTime << '\n';
    std::cout << std::scientific;
    double avgQ3_ns = tQ3 * 1e6 / numSamples;
    double avgSIMD_ns = tSIMD * 1e6 / numSamples;
#if defined(__AVX__) && defined(__AVX2__)
    double avgAVX_ns = tAVX * 1e6 / numSamples;
#endif
    std::cout << "Quake3 Time:\t" << tQ3 << " ms\t(avg=" << avgQ3_ns << " ns)\n";
    std::cout << "SIMD Time:\t" << tSIMD << " ms\t(avg=" << avgSIMD_ns << " ns)\n";
#if defined(__AVX__) && defined(__AVX2__)
    std::cout << "AVX2 Time:\t" << tAVX << " ms\t(avg=" << avgAVX_ns << " ns)\n";
#endif
    std::cout << "Std Time:\t" << tStd << " ms\n";

    std::cout << '\n' << "Checksums -> Quake3: " << checksum(q3)
              << ", SIMD: " << checksum(simd);
#if defined(__AVX__) && defined(__AVX2__)
    std::cout << ", AVX2: " << checksum(avx);
#endif
    std::cout << ", Std: " << checksum(stdv) << '\n';

    std::cout.setf(std::ios::fixed); std::cout << std::setprecision(6);
    std::cout << "Error metrics\n\tmean_abs,\t\tmax_abs,\t\tmean_rel\n";
    std::cout << "Quake3:\t" << q3_mean_abs << ",\t" << q3_max_abs << ",\t" << q3_mean_rel << '\n';
    std::cout << "SIMD:\t" << simd_mean_abs << ",\t" << simd_max_abs << ",\t" << simd_mean_rel << '\n';
#if defined(__AVX__) && defined(__AVX2__)
    std::cout << "AVX2:\t" << avx_mean_abs << ",\t" << avx_max_abs << ",\t" << avx_mean_rel << '\n';
#endif

    return 0;
}

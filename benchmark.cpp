// Benchmark harness for the fast inverse‑sqrt implementations.

#include "fast_inverse_sqrt.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

int main() {
    // Number of samples per function. A large number reduces the impact of
    // loop overhead on the measured time.
    constexpr size_t numSamples = 1'000'000;
    // Warm‑up iterations to allow CPU caches, branch predictors and any
    // compiler optimisations that depend on repeated execution to settle.
    constexpr int warmupCount = 5;
    std::vector<float> inputs(numSamples);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.1f, 1000.f);
    for (auto& v : inputs) v = dist(rng);

    std::vector<float> q3(numSamples), simd(numSamples), stdv(numSamples);

    // Helper that performs warm‑up iterations followed by a timed run.
    auto bench=[&](float (*func)(float), std::vector<float>& out){
        // Warm‑up: execute the function and accumulate a checksum to
        // prevent the compiler from eliminating the loop. The checksum is
        // discarded after warm‑up.
        volatile float dummy = 0.f;
        for(int w=0; w<warmupCount; ++w) {
            for(size_t i=0;i<numSamples;++i){
                dummy += func(inputs[i]);
            }
        }

        auto start = std::chrono::steady_clock::now();
        for(size_t i=0;i<numSamples;++i) out[i] = func(inputs[i]);
        return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    };

    double tQ3   = bench(fisq::FastInverseSqrt<float>, q3);
    double tSIMD = bench(fisq::FastInverseSqrtSIMD, simd);
    double tStd  = bench([](float x){return 1.0f/std::sqrtf(x);}, stdv);

    auto checksum=[&](const std::vector<float>& v){double s=0;for(auto f:v)s+=f;return s;};
    double sumQ3   = checksum(q3), sumSIMD = checksum(simd), sumStd = checksum(stdv);

    std::cout.setf(std::ios::fixed);std::cout<<std::setprecision(4);
    // Report raw times and derived metrics.
    // Convert average per call to nanoseconds for readability.
    const double avgSIMD_ns = tSIMD * 1e6 / static_cast<double>(numSamples);
    const double avgQ3_ns   = tQ3   * 1e6 / static_cast<double>(numSamples);
    const double avgStd_ns  = tStd  * 1e6 / static_cast<double>(numSamples);

    std::cout << "Warm‑up iterations: " << warmupCount << '\n';
    std::cout << "Iterations per function: " << numSamples << '\n';
    std::cout << "Total elapsed time (ms): " << tSIMD + tQ3 + tStd << '\n';
    // Use scientific notation for very small averages.
    std::cout.setf(std::ios::scientific, std::ios::floatfield);
    std::cout << "SIMD Time:\t" << tSIMD << " ms\t(avg=" << avgSIMD_ns << " ns)\n";
    std::cout << "Quake3 Time:\t" << tQ3   << " ms\t(avg=" << avgQ3_ns   << " ns)\n";
    std::cout << "Std Time:\t"  << tStd  << " ms\t(avg=" << avgStd_ns  << " ns)\n";
    std::cout << "Checksums -> SIMD: "<<sumSIMD<<", Quake3: "<<sumQ3<<", Std: "<<sumStd<<'\n';

    return 0;
}


// Benchmark harness for the fast inverse‑sqrt implementations.

#include "fast_inverse_sqrt.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

int main() {
    constexpr size_t numSamples = 1'000'000;
    std::vector<float> inputs(numSamples);
    std::mt19937 rng(42);
    std::uniform_real_distribution<float> dist(0.1f, 1000.f);
    for (auto& v : inputs) v = dist(rng);

    std::vector<float> q3(numSamples), simd(numSamples), stdv(numSamples);

    auto bench=[&](float (*func)(float), std::vector<float>& out){
        auto start=std::chrono::steady_clock::now();
        for(size_t i=0;i<numSamples;++i) out[i]=func(inputs[i]);
        return std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
    };

    double tQ3   = bench(fisq::FastInverseSqrt<float>, q3);
    double tSIMD = bench(fisq::FastInverseSqrtSIMD, simd);
    double tStd  = bench([](float x){return 1.0f/std::sqrtf(x);}, stdv);

    auto checksum=[&](const std::vector<float>& v){double s=0;for(auto f:v)s+=f;return s;};
    double sumQ3   = checksum(q3), sumSIMD = checksum(simd), sumStd = checksum(stdv);

    std::cout.setf(std::ios::fixed);std::cout<<std::setprecision(4);
    std::cout << "SIMD Time:\t" << tSIMD << " ms\n";
    std::cout << "Quake3 Time:\t" << tQ3   << " ms\n";
    std::cout << "Std Time:\t"  << tStd  << " ms\n";
    std::cout << "Checksums -> SIMD: "<<sumSIMD<<", Quake3: "<<sumQ3<<", Std: "<<sumStd<<'\n';

    return 0;
}


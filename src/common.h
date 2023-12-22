#include <iostream>
#include <chrono>
#include <vector>
#include <utility>
#include <random>
#include <algorithm>

struct ProblemSize {
    long M;
    long N;
    long K;

    long ASize() const {
        return M * K;
    }

    long BSize() const {
        return K * N;
    }

    long CSize() const {
        return M * N;
    }
};

struct BenchmarkScope {
    BenchmarkScope(const ProblemSize& problem_size) : problem_size(problem_size) {
        num_iter = 100. / 2 / problem_size.M / problem_size.N / problem_size.K * 1e9;
    }

    void Start() {
        start = std::chrono::high_resolution_clock::now();
    }

    void End() {
        long M = problem_size.M;
        long N = problem_size.N;
        long K = problem_size.K;
        auto end = std::chrono::high_resolution_clock::now();
        auto diff = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        float gflops = ((2. * static_cast<double>(M) * N * K * num_iter) * 1e6 / diff.count()) / 1e9;
        std::cout.precision(2);
        std::cout << M << 'x' << N << 'x' << K << ' ' << std::fixed << gflops << std::endl;
    }

    long NumIter() {
        return num_iter;
    }
private:
    const ProblemSize& problem_size;
    std::chrono::time_point<std::chrono::high_resolution_clock> start;
    long num_iter;
};

inline void InitMat(float *A, float *B, float *C, const ProblemSize& s) {
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    std::generate(A, A + s.ASize(), [&]() { return dis(gen); });
    std::generate(B, B + s.BSize(), [&]() { return dis(gen); });
    std::fill(C, C + s.CSize(), 0.f);
}

const std::vector<ProblemSize> kProblemSizes = {
    {128, 128, 128},
    {256, 256, 256},
    {256, 1024, 4096},
    {256, 4096, 1024},
    {256, 1024, 1024},
    {128, 1024, 4096},
    {128, 4096, 1024},
    {128, 1024, 1024},
    {256, 768, 768},
    {128, 768, 768},
    {128, 3072, 768},
    {128, 768, 3072},
    {256, 3072, 768},
    {256, 768, 3072},
    {128, 768, 2304},
    {1024, 2560, 1024},
    {1024, 1024, 512},
    {1024, 352, 512},
    {1024, 512, 256}
};
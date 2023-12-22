#include "common.h"

void MatMul(float *A, float *B, float *C, int M, int N, int K) {
    for (int m = 0; m < M; ++m) {
        for (int k = 0; k < K; ++k) {
            for (int n = 0; n < N; ++n) {
                C[m * N + n] += A[m * K + k] * B[k * N + n];
            }
        }
    }
}

int main() {
    for (const auto& s : kProblemSizes) {
        BenchmarkScope scope(s);
        
        auto A = new float[s.ASize()];
        auto B = new float[s.BSize()];
        auto C = new float[s.CSize()];

        InitMat(A, B, C, s);
        
        scope.Start();
        for (int i = 0; i < scope.NumIter(); ++i) {
            MatMul(A, B, C, s.M, s.N, s.K);
        }
        scope.End();
    }

    return 0;
}
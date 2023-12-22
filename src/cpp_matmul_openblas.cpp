#include <cblas.h>

#include "common.h"

void MatMul(float *A, float *B, float *C, int M, int N, int K) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, K, B, N, 0, C, N);
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
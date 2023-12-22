#include <mkl.h>
#include <mkl_cblas.h>

#include "common.h"

void MatMul(float *A, float *B, float *C, int M, int N, int K) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, M, N, K, 1, A, K, B, N, 0, C, N);
}

int main() {
    for (const auto& s : kProblemSizes) {
        BenchmarkScope scope(s);
        
        float *A = (float*)mkl_malloc(sizeof(float) * s.ASize(), 64);
        float *B = (float*)mkl_malloc(sizeof(float) * s.BSize(), 64);
        float *C = (float*)mkl_malloc(sizeof(float) * s.CSize(), 64);

        InitMat(A, B, C, s);
        
        scope.Start();
        for (int i = 0; i < scope.NumIter(); ++i) {
            MatMul(A, B, C, s.M, s.N, s.K);
        }
        scope.End();
    }

    return 0;
}
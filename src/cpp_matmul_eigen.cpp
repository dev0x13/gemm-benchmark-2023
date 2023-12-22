#include <Eigen/Dense>

#include "common.h"

using MatT = Eigen::MatrixXd;

void InitMat(MatT& A, MatT& B, MatT& C) {
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(0.0, 1.0);
    for (size_t r = 0; r < A.rows(); ++r) {
        for (size_t c = 0; c < A.cols(); ++c) {
            A(r, c) = dis(gen);
        }
    }
    for (size_t r = 0; r < B.rows(); ++r) {
        for (size_t c = 0; c < B.cols(); ++c) {
            B(r, c) = dis(gen);
        }
    }
    C = MatT::Constant(C.rows(), C.cols(), 0);
}

void MatMul(MatT& A, MatT& B, MatT& C) {
    C.noalias() = A * B;
}

int main() {
    for (const auto& s : kProblemSizes) {
        BenchmarkScope scope(s);
        
        MatT A = MatT(s.M, s.K);
        MatT B = MatT(s.K, s.N);
        MatT C = MatT(s.M, s.N);

        InitMat(A, B, C);
        
        scope.Start();
        for (int i = 0; i < scope.NumIter(); ++i) {
            MatMul(A, B, C);
        }
        scope.End();
    }

    return 0;
}
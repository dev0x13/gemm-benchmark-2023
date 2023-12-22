#/bin/bash

set -e

make all

echo "Single-thread C++ benchmarks"

export OMP_NUM_THREADS=1

echo "cpp_matmul_naive"
./cpp_matmul_naive
echo "cpp_matmul_eigen"
./cpp_matmul_eigen
echo "cpp_matmul_openblas"
./cpp_matmul_openblas
echo "cpp_matmul_mkl"
./cpp_matmul_mkl

echo "Multi-thread C++ benchmarks"

unset OMP_NUM_THREADS

echo "cpp_matmul_eigen"
./cpp_matmul_eigen
echo "cpp_matmul_openblas"
./cpp_matmul_openblas
echo "cpp_matmul_mkl"
./cpp_matmul_mkl

echo "Mojo benchmark"

if command -v mojo &> /dev/null; then
  mojo matmul.mojo;
else
  echo "Mojo is not installed, skipping benchmark";
fi

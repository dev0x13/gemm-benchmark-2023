cpp_matmul_naive: cpp_matmul_naive.cpp common.h
	clang++ -O3 cpp_matmul_naive.cpp -march=native -o cpp_matmul_naive

cpp_matmul_eigen: cpp_matmul_eigen.cpp common.h
	clang++ -O3 cpp_matmul_eigen.cpp -march=native -I ${EIGENROOT} -fopenmp -o cpp_matmul_eigen

cpp_matmul_mkl: cpp_matmul_mkl.cpp common.h
	clang++ -O3 cpp_matmul_mkl.cpp -march=native -I ${MKLROOT}/include -L ${MKLROOT}/lib -Wl,--rpath ${MKLROOT}/lib -liomp5 -lmkl_core -lmkl_intel_thread -lmkl_intel_lp64 -lm -o cpp_matmul_mkl

cpp_matmul_openblas: cpp_matmul_openblas.cpp common.h
	clang++ -O3 cpp_matmul_openblas.cpp -march=native -I ${OPENBLASROOT}/include -L ${OPENBLASROOT}/lib -Wl,--rpath ${OPENBLASROOT}/lib -fopenmp -lopenblas -lm  -o cpp_matmul_openblas

clean:
	rm cpp_matmul_eigen cpp_matmul_mkl cpp_matmul_naive cpp_matmul_openblas

all: cpp_matmul_eigen cpp_matmul_mkl cpp_matmul_naive cpp_matmul_openblas

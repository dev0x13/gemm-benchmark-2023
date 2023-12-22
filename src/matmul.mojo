# ===----------------------------------------------------------------------=== #
# Copyright (c) 2023, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #

# This sample demonstrates how various systems optimizations can be
# applied to a naive matmul implementation in Mojo to gain significant
# performance speedups

import benchmark
from memory import memset_zero, stack_allocation
from random import rand
from algorithm import vectorize, parallelize, vectorize_unroll
from algorithm import Static2DTileUnitFunc as Tile2DFunc
from python import Python
from tensor import Tensor
from utils.index import Index
from memory.buffer import NDBuffer

alias type = DType.float32


struct Matrix:
    var data: DTypePointer[type]
    var rows: Int
    var cols: Int

    # Initialize zeroeing all values
    fn __init__(inout self, rows: Int, cols: Int):
        self.data = DTypePointer[type].alloc(rows * cols)
        memset_zero(self.data, rows * cols)
        self.rows = rows
        self.cols = cols

    # Initialize taking a pointer, don't set any elements
    fn __init__(inout self, rows: Int, cols: Int, data: DTypePointer[DType.float32]):
        self.data = data
        self.rows = rows
        self.cols = cols

    ## Initialize with random values
    @staticmethod
    fn rand(rows: Int, cols: Int) -> Self:
        let data = DTypePointer[type].alloc(rows * cols)
        rand(data, rows * cols)
        return Self(rows, cols, data)

    fn __getitem__(self, y: Int, x: Int) -> Float32:
        return self.load[1](y, x)

    fn __setitem__(self, y: Int, x: Int, val: Float32):
        return self.store[1](y, x, val)

    fn load[nelts: Int](self, y: Int, x: Int) -> SIMD[DType.float32, nelts]:
        return self.data.simd_load[nelts](y * self.cols + x)

    fn store[nelts: Int](self, y: Int, x: Int, val: SIMD[DType.float32, nelts]):
        return self.data.simd_store[nelts](y * self.cols + x, val)


# Mojo has SIMD vector types, we can vectorize the Matmul code as follows.
alias nelts = simdwidthof[type]()  # The SIMD vector width.


# Using stdlib vectorize function
fn vectorized(inout C: Matrix, A: Matrix, B: Matrix):
    for m in range(C.rows):
        for k in range(A.cols):

            @parameter
            fn dot[nelts: Int](n: Int):
                C.store[nelts](
                    m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n)
                )

            vectorize[nelts, dot](C.cols)


# Parallelize the code by using the builtin parallelize function
fn parallelized(inout C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        for k in range(A.cols):

            @parameter
            fn dot[nelts: Int](n: Int):
                C.store[nelts](
                    m, n, C.load[nelts](m, n) + A[m, k] * B.load[nelts](k, n)
                )

            vectorize[nelts, dot](C.cols)

    parallelize[calc_row](C.rows, C.rows)


# Perform 2D tiling on the iteration space defined by end_x and end_y.
fn tile[tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int](end_x: Int, end_y: Int):
    # Note: this assumes that ends are multiples of the tiles.
    for y in range(0, end_y, tile_y):
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)


# Use the above tile function to perform tiled matmul.
fn tiled(inout C: Matrix, A: Matrix, B: Matrix):
    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):

                @parameter
                fn dot[
                    nelts: Int,
                ](n: Int):
                    C.store[nelts](
                        m,
                        n + x,
                        C.load[nelts](m, n + x) + A[m, k] * B.load[nelts](k, n + x),
                    )

                vectorize[nelts, dot](tile_x)

        # We hardcode the tile factor to be 4.
        alias tile_size = 4
        tile[calc_tile, nelts * tile_size, tile_size](C.cols, B.rows)

    parallelize[calc_row](C.rows, C.rows)


# Unroll the vectorized loop by a constant factor.
# from Functional import vectorize_unroll
fn unrolled(inout C: Matrix, A: Matrix, B: Matrix):
    alias tile_size = 4

    @parameter
    fn calc_row(m: Int):
        @parameter
        fn calc_tile[tile_x: Int, tile_y: Int](x: Int, y: Int):
            for k in range(y, y + tile_y):

                @parameter
                fn dot[
                    nelts: Int,
                ](n: Int):
                    C.store[nelts](
                        m,
                        n + x,
                        C.load[nelts](m, n + x) + A[m, k] * B.load[nelts](k, n + x),
                    )

                # Vectorize by nelts and unroll by tile_x/nelts
                # Here unroll factor is 4
                vectorize_unroll[nelts, tile_x // nelts, dot](tile_x)

        tile[calc_tile, nelts * tile_size, tile_size](C.cols, B.rows)

    parallelize[calc_row](C.rows, C.rows)


# Perform 2D tiling on the iteration space defined by end_x and end_y, parallelizing over y.
fn tile_parallel[
    tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int
](end_x: Int, end_y: Int, M: Int):
    # Note: this assumes that ends are multiples of the tiles.
    @parameter
    fn row(yo: Int):
        let y = tile_y * yo
        for x in range(0, end_x, tile_x):
            tiled_fn[tile_x, tile_y](x, y)

    parallelize[row](end_y // tile_y, M)


# Use stack allocation for tiles to accumulate values efficiently,
# avoiding repeated reads and writes to memory. Also reorder the loops
# and do not fully unroll the loop over the reduction dimension.
fn reordered(inout C: Matrix, A: Matrix, B: Matrix):
    alias tile_k = 8
    alias tile_k_unroll = 8
    alias tile_i = 32
    alias tile_j = nelts * 4

    @parameter
    fn calc_tile[tile_j: Int, tile_i: Int](jo: Int, io: Int):
        # Allocate the tile of accumulators on the stack.
        var accumulators = Matrix(
            tile_i, tile_j, stack_allocation[tile_i * tile_j, DType.float32]()
        )

        for ko in range(0, A.cols, tile_k * tile_k_unroll):
            for _ in range(tile_i):
                for i in range(tile_k):

                    @unroll
                    for k in range(tile_k_unroll):

                        @parameter
                        fn calc_tile_cols[nelts: Int](j: Int):
                            accumulators.store[nelts](
                                i,
                                j,
                                accumulators.load[nelts](i, j)
                                + A[io + i, ko + k] * B.load[nelts](ko + k, jo + j),
                            )

                        vectorize_unroll[nelts, tile_j // nelts, calc_tile_cols](tile_j)

        # Copy the local tile to the output
        for i in range(tile_i):
            for j in range(tile_j):
                C[io + i, jo + j] = accumulators[i, j]

    tile_parallel[calc_tile, tile_j, tile_i](C.cols, C.rows, C.rows)


# Perform 2D tiling on the iteration space defined by end_x and end_y, parallelizing
# over x and y, and iterating in an order that has better L3 cache locality
fn tile_parallel_swizzled[
    tiled_fn: Tile2DFunc, tile_x: Int, tile_y: Int
](end_x: Int, end_y: Int, M: Int, N: Int):
    # Note: this assumes that ends are multiples of the tiles.
    alias tile_outer = 8
    alias group_size = tile_outer * 4

    # L3 cache swizzling
    @parameter
    fn row(swizzled: Int):
        let group_id = swizzled // group_size
        let group_offset_x = (group_id * tile_outer) % (N // tile_y)
        let yo = (swizzled % group_size) // tile_outer
        let xo = group_offset_x + (swizzled % tile_outer)
        let y = tile_y * yo
        let x = tile_x * xo
        tiled_fn[tile_x, tile_y](x, y)

    parallelize[row]((end_y // tile_y * end_x // tile_x), M * 2)


# Same as previous example but utilisizing tile swizzling for better L3 cache locality.
fn swizzled(inout C: Matrix, A: Matrix, B: Matrix):
    alias tile_k = 8
    alias tile_k_unroll = 8
    alias tile_i = 32
    alias tile_j = nelts * 4

    @parameter
    fn calc_tile[tile_j: Int, tile_i: Int](jo: Int, io: Int):
        # Allocate the tile of accumulators on the stack.
        var accumulators = Matrix(
            tile_i, tile_j, stack_allocation[tile_i * tile_j, DType.float32]()
        )

        for ko in range(0, A.cols, tile_k * tile_k_unroll):
            for _ in range(tile_i):
                for i in range(tile_k):

                    @unroll
                    for k in range(tile_k_unroll):

                        @parameter
                        fn calc_tile_cols[nelts: Int](j: Int):
                            accumulators.store[nelts](
                                i,
                                j,
                                accumulators.load[nelts](i, j)
                                + A[io + i, ko + k] * B.load[nelts](ko + k, jo + j),
                            )

                        vectorize_unroll[nelts, tile_j // nelts, calc_tile_cols](tile_j)

        # Copy the local tile to the output
        for i in range(tile_i):
            for j in range(tile_j):
                C[io + i, jo + j] = accumulators[i, j]

    tile_parallel_swizzled[calc_tile, tile_j, tile_i](C.cols, C.rows, C.rows, C.cols)


@always_inline
fn bench[
    func: fn (inout Matrix, Matrix, Matrix) -> None, name: StringLiteral
](M: Int, N: Int, K: Int) raises:
    var A = Matrix.rand(M, K)
    var B = Matrix.rand(K, N)
    var C = Matrix(M, N)

    @always_inline
    @parameter
    fn test_fn():
        _ = func(C, A, B)

    let secs = benchmark.run[test_fn](max_runtime_secs=0.5).mean()
    # Prevent the matrices from being freed before the benchmark run
    A.data.free()
    B.data.free()
    C.data.free()
    let gflops = ((2 * M * N * K) / secs) / 1e9

    let py = Python.import_module("builtins")
    _ = py.print(
        py.str("{:<13}{:>8.3f} GFLOPS").format(
            name, gflops
        )
    )

@always_inline
fn test[
    func: fn (inout Matrix, Matrix, Matrix) -> None
](A: Matrix, B: Matrix, M: Int, N: Int, K: Int) raises -> SIMD[type, 1]:
    var C = Matrix(M, N)
    _ = func(C, A, B)
    var result = SIMD[type, 1]()
    for i in range(C.rows):
        for j in range(C.cols):
            result += C[i, j]
    return result


fn test_all(M: Int, N: Int, K: Int) raises:
    let A = Matrix.rand(M, K)
    let B = Matrix.rand(K, N)

    let result = test[vectorized](A, B, M, N, K)

    if test[vectorized](A, B, M, N, K) != result:
        raise Error("Vectorize output does not match")
    if test[parallelized](A, B, M, N, K) != result:
        raise Error("Parallelize output incorrect")
    if test[tiled](A, B, M, N, K) != result:
        raise Error("Tiled output incorrect")
    if test[unrolled](A, B, M, N, K) != result:
        raise Error("Unroll output incorrect")
    if test[reordered](A, B, M, N, K) != result:
        raise Error("Loop reorder output incorrect")
    if test[swizzled](A, B, M, N, K) != result:
        raise Error("Loop reorder output incorrect")

    A.data.free()
    B.data.free()


fn bench2(M: Int, N: Int, K: Int) raises:
    print(M, N, K)
    test_all(M, N, K)
    bench[vectorized, "Vectorized:"](M, N, K)
    bench[parallelized, "Parallelized:"](M, N, K)
    bench[tiled, "Tiled:"](M, N, K)
    bench[unrolled, "Unrolled:"](M, N, K)
    bench[reordered, "Reordered:"](M, N, K)
    bench[swizzled, "Swizzled:"](M, N, K)

fn main() raises:
    bench2(128, 128, 128)
    bench2(256, 256, 256)
    bench2(256, 1024, 4096)
    bench2(256, 4096, 1024)
    bench2(256, 1024, 1024)
    bench2(128, 1024, 4096)
    bench2(128, 4096, 1024)
    bench2(128, 1024, 1024)
    bench2(256, 768, 768)
    bench2(128, 768, 768)
    bench2(128, 3072, 768)
    bench2(128, 768, 3072)
    bench2(256, 3072, 768)
    bench2(256, 768, 3072)
    bench2(128, 768, 2304)
    bench2(1024, 2560, 1024)
    bench2(1024, 1024, 512)
    # This one crashes with a segfault:
    # https://github.com/modularml/mojo/issues/1426
    # bench2(1024, 352, 512)
    bench2(1024, 512, 256)
# gemm-benchmark-2023

This repository hosts a set of becnhmarks for modern (2023) high-performance floating-point GEMM implementations.

Implementaitons covered:
* Naive C++
* Intel MKL 2020.4.304
* Eigen 3.4.0
* OpenBLAS 0.3.25
* Mojo 0.6.0

## Running benchmarks

1.  Build the Docker image:
```shell
docker build --build-arg="MODULAR_AUTH_TOKEN=<token>" -t gemm-benchmark-2023 .
```
The `MODULAR_AUTH_TOKEN` argument is optional. If set, the Docker image will also include benchmark for the Mojo language. The Modular auth token can be obtained from [Modular's website](https://developer.modular.com/download).

2. Run the Docker container:
```shell
docker run -it gemm-benchmark-2023
```
The container will output benchmark results in GFLOPS to the console.

## Reference numbers

Here are some reference results obtained on Intel Xeon Platinum 8124M (16 cores, `c5.4xlarge` AWS EC2 instance) running Ubuntu 22.04:

### Multithread

| Problem size   | Mojo ("Swizzled") | Eigen | MKL    | OpenBLAS |
|----------------|-------------------|-------|--------|----------|
| 128x128x128    |              24.4 | 109.0 |  543.8 |    143.6 |
| 256x256x256    |             129.6 | 190.3 |  939.5 |    380.9 |
| 256x1024x4096  |             835.4 | 378.8 | 1063.6 |    860.6 |
| 256x4096x1024  |             818.0 | 428.0 | 1001.6 |    770.9 |
| 256x1024x1024  |             690.8 | 387.8 | 1037.7 |    806.8 |
| 128x1024x4096  |             820.5 | 390.8 | 1078.6 |    784.4 |
| 128x4096x1024  |             795.0 | 404.6 | 1044.2 |    688.9 |
| 128x1024x1024  |             679.6 | 380.9 | 1028.9 |    707.4 |
| 256x768x768    |             582.2 | 351.1 | 1051.7 |    818.5 |
| 128x768x768    |             579.0 | 342.1 |  893.4 |    707.5 |
| 128x3072x768   |             783.4 | 396.7 |  990.2 |    755.0 |
| 128x768x3072   |             814.3 | 381.2 | 1085.7 |    784.2 |
| 256x3072x768   |             794.7 | 424.3 | 1096.7 |    846.4 |
| 256x768x3072   |             819.6 | 356.1 | 1116.0 |    865.4 |
| 128x768x2304   |             797.8 | 381.0 | 1089.7 |    826.5 |
| 1024x2560x1024 |             808.8 | 437.3 | 1176.1 |   1126.2 |
| 1024x1024x512  |             556.7 | 399.0 | 1227.9 |    933.6 |
| 1024x352x512   |               0.0 | 206.3 | 1185.6 |    575.1 |
| 1024x512x256   |             245.4 | 313.5 | 1231.7 |    749.2 |

### Single thread

| Problem size   | Mojo ("Vectorized") | Eigen | MKL   | OpenBLAS | Naive |
|----------------|---------------------|-------|-------|----------|-------|
| 128x128x128    |                18.6 |  67.8 | 166.9 |     98.7 |  21.1 |
| 256x256x256    |                20.5 |  57.1 | 149.6 |    133.5 |  19.5 |
| 256x1024x4096  |                10.0 |  56.0 | 136.4 |    125.4 |  11.0 |
| 256x4096x1024  |                 8.3 |  56.5 | 138.6 |    128.0 |  11.0 |
| 256x1024x1024  |                11.9 |  57.7 | 152.0 |    140.5 |  13.2 |
| 128x1024x4096  |                10.3 |  51.0 |  80.5 |    112.3 |  11.1 |
| 128x4096x1024  |                 8.5 |  51.0 |  77.4 |    112.3 |  10.8 |
| 128x1024x1024  |                11.5 |  53.4 |  91.4 |    127.9 |  13.2 |
| 256x768x768    |                26.0 |  58.2 | 155.0 |    151.0 |  13.1 |
| 128x768x768    |                12.5 |  54.6 | 163.3 |    137.7 |  13.2 |
| 128x3072x768   |                 9.6 |  52.1 |  87.7 |    124.5 |  12.5 |
| 128x768x3072   |                11.2 |  51.9 | 157.2 |    134.6 |  12.6 |
| 256x3072x768   |                 9.7 |  58.6 | 147.4 |    137.0 |  12.4 |
| 256x768x3072   |                12.0 |  59.6 | 148.4 |    143.3 |  12.6 |
| 128x768x2304   |                12.4 |  52.4 | 162.7 |    135.2 |  12.7 |
| 1024x2560x1024 |                10.7 |  58.8 | 160.3 |    151.9 |  12.6 |
| 1024x1024x512  |                11.7 |  58.1 | 163.5 |    141.8 |  13.2 |
| 1024x352x512   |                 0.0 |  56.6 | 162.3 |    159.4 |  17.1 |
| 1024x512x256   |                25.6 |  57.5 | 167.7 |    143.8 |  20.3 |

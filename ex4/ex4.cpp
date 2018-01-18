#include <cassert>
#include <iostream>
#include <chrono>

#include <cuda_runtime.h>

#include "ex4.h"

int main() {
    int test_size = 1000;

    double *d_A, *d_x, *d_y;

    cudaError_t cudaErr;
    cudaErr = cudaMalloc(&d_A, test_size * test_size * sizeof(double));
    assert(cudaErr == cudaSuccess);
    cudaErr = cudaMalloc(&d_x, test_size * sizeof(double));
    assert(cudaErr == cudaSuccess);
    cudaErr = cudaMalloc(&d_y, test_size * sizeof(double));
    assert(cudaErr == cudaSuccess);

    auto start = std::chrono::high_resolution_clock::now();
    axpy(test_size, 1.0, d_x, 1, d_y, 1);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << diff.count() << std::endl;

    start = std::chrono::high_resolution_clock::now();
    gemv(test_size, test_size, d_A, test_size, d_x, 1, d_y, 1);
    end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> diff = end - start;
    std::cout << diff.count() << std::endl;

    return 0;
}

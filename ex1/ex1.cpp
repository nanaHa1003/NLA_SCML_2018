#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <chrono>

template <typename T>
T nrm2(int n, const T* x, int incx) noexcept {
    T sum = 0.0;
    for (int i = 0; i < n; ++i) {
        sum = std::fma(x[i * incx], x[i * incx], sum);
    }
    return std::sqrt(sum);
}

template <typename T>
void gemv(
    int m, int n,
    const T* A, int lda,
    const T* x, int incx,
    T* y, int incy) noexcept {
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            y[j * incy] = std::fma(A[i * lda + j], x[j * incx], y[j * incy]);
        }
    }
}

template <typename T>
void gemm(
    int m, int n, int k,
    const T* A, int lda,
    const T* B, int ldb,
    T* C, int ldc) noexcept {
    for (int i = 0; i < m; ++i) {
        for (int l = 0; l < n; ++l) {
            for (int j = 0; j < k; ++j) {
                C[i * ldc + j] = std::fma(A[i * lda + l], B[l * ldb + j], C[i * ldc + j]);
            }
        }
    }
}

int main(int argc, char **argv) {
    std::vector<int>    test_sizes = {{ 32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048 }};
    std::vector<double> elap_times;

    elap_times.reserve(test_sizes.size());

    std::cout << std::setprecision(16);
    std::cout << std::fixed;

    // Print problem sizes
    for (size_t i = 0; i < test_sizes.size() - 1; ++i) {
        std::cout << test_sizes[i] << ",";
    }
    std::cout << test_sizes.back() << std::endl;

    // Problem 1. norm
    for (int &size: test_sizes) {
        std::vector<double> x(size);

        // Run 10 times then calculate average time
        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 10; ++i) {
            double tmp = nrm2(size, x.data(), 1);
        }
        auto end = std::chrono::steady_clock::now();

        std::chrono::duration<double> diff_time = end - start;
        elap_times.push_back(diff_time.count() / 10.0);
    }

    // Summary of problem 1
    for (size_t i = 0; i < elap_times.size() - 1; ++i) {
        std::cout << elap_times[i] << ",";
    }
    std::cout << elap_times.back() << std::endl;

    elap_times.resize(0);

    // Problem 2. gemv
    for (int &size: test_sizes) {
        std::vector<double> A(size * size), x(size), y(size);

        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 10; ++i) {
            gemv(size, size, A.data(), size, x.data(), 1, y.data(), 1);
        }
        auto end = std::chrono::steady_clock::now();

        std::chrono::duration<double> diff_time = end - start;
        elap_times.push_back(diff_time.count() / 10.0);
    }

    // Summary of problem 2
    for (size_t i = 0; i < elap_times.size() - 1; ++i) {
        std::cout << elap_times[i] << ",";
    }
    std::cout << elap_times.back() << std::endl;

    elap_times.resize(0);

    // Problem 3. gemm
    for (int &size : test_sizes) {
        std::vector<double> A(size * size), B(size * size), C(size * size);

        auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 10; ++i) {
            gemm(size, size, size, A.data(), size, B.data(), size, C.data(), size);
        }
        auto end = std::chrono::steady_clock::now();

        std::chrono::duration<double> diff_time = end - start;
        elap_times.push_back(diff_time.count() / 10.0);
    }

    // Summary of problem 3
    for (size_t i = 0; i < elap_times.size() - 1; ++i) {
        std::cout << elap_times[i] << ",";
    }
    std::cout << elap_times.back() << std::endl;

    return 0;
}

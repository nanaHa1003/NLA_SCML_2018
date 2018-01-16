#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>

template <typename T>
void getrf(int n, T* A, int lda) noexcept {
    for (int i = 0; i < n; ++i) {
        T inv_pivot = 1.0 / A[i * lda + i];
        for (int j = i + 1; j < n; ++j) {
            A[i * lda + j] *= inv_pivot;
        }

        for (int j = i + 1; j < n; ++j) {
            for (int k = i + 1; k < n; ++k) {
                A[j * lda + k] -= A[j * lda + i] * A[i * lda + k];
            }
        }
    }
}

template <typename T>
void getrf_pivot(int n, T* A, int lda, int *ipiv) noexcept {
    std::iota(ipiv, ipiv + n, 0);

    for (int i = 0; i < n; ++i) {
        // Find pivot first
        int pivot_idx = i;
        T   pivot_val = A[i * lda + i];
        for (int j = i + 1; j < n; ++j) {
            if (A[i * lda + j] > pivot_val) {
                pivot_val = A[i * lda + j];
                pivot_idx = j;
            }
        }

        // Perform pivoting if necessary
        if (pivot_idx != i) {
            std::swap(ipiv[i], ipiv[pivot_idx]);
            for (int j = i; j < n; ++j) {
                std::swap(A[i * lda + j], A[pivot_idx * lda + j]);
            }
        }

        T inv_pivot = 1.0 / A[i * lda + i];
        for (int j = i + 1; j < n; ++j) {
            A[i * lda + j] *= inv_pivot;
        }

        for (int j = i + 1; j < n; ++j) {
            for (int k = i + 1; k < n; ++k) {
                A[j * lda + k] -= A[j * lda + i] * A[i * lda + k];
            }
        }
    }
}

int main(int argc, char **argv) {
    std::vector<int> test_sizes = {{ 8, 16, 32, 64, 128, 256, 512, 1024, 2048 }};

    // LU without pivoting
    for (int size: test_sizes) {
        std::vector<double> A(size * size);

        // Generate random matrix A
        std::generate(A.begin(), A.end(), [](){
            return static_cast<double>(std::rand()) / RAND_MAX;
        });

        auto start = std::chrono::system_clock::now();
        getrf(size, A.data(), size);
        auto end = std::chrono::system_clock::now();

        std::chrono::duration<double> diff = end - start;
        std::cout << diff.count() << std::endl;
    }

    // LU with pivoting
    for (int size: test_sizes) {
        std::vector<double> A(size * size);
        std::vector<int>    ipiv(size);

        // Generate random matrix A
        std::generate(A.begin(), A.end(), [](){
            return static_cast<double>(std::rand()) / RAND_MAX;
        });

        auto start = std::chrono::system_clock::now();
        getrf_pivot(size, A.data(), size, ipiv.data());
        auto end = std::chrono::system_clock::now();

        std::chrono::duration<double> diff = end - start;
        std::cout << diff.count() << std::endl;
    }

    return 0;
}

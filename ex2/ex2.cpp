#include <cmath>
#include <cassert>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <numeric>
#include <limits>
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
        // First find column pivot
        int pivot_idx = i;
        T   pivot_val = A[i * lda + i];
        for (int j = i + 1; j < n; ++j) {
            if (A[j * lda + i] > pivot_val) {
                pivot_val = A[j * lda + i];
                pivot_idx = j;
            }
        }

        // Perform pivoting if necessary
        if (pivot_idx != i) {
            std::swap(ipiv[i], ipiv[pivot_idx]);
            std::swap_ranges(A + i * lda + i, A + (i + 1) * lda, A + pivot_idx * lda + i);
        }

        T inv_pivot = 1.0 / pivot_val;
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

void test(int size) noexcept {
    std::vector<double> A(size * size);
    std::generate(A.begin(), A.end(), [](){
        return 0.01 * static_cast<double>(std::rand()) / RAND_MAX;
    });

    for (int i = 0; i < size; ++i) {
        A[i * size + i] += 1.0;
    }

    std::vector<double> x(size, 1.0), y(size, 0.0);
    for (int i = 0; i < size; ++i) {
        for (int j = 0; j < size; ++j) {
            y[j] += x[i] * A[i * size + j];
        }
    }

    getrf(size, A.data(), size);

    for (int i = 0; i < size; ++i) {
        for (int j = i + 1; j < size; ++j) {
            y[j] -= A[i * size + j] * y[i];
        }
    }

    for (int i = size - 1; i >= 0; --i) {
        y[i] /= A[i * size + i];
        for (int j = i - 1; j >= 0; --j) {
            y[j] -= A[i * size + j] * y[i];
        }
    }

    int nrm2 = 0.0;
    for (int i = 0; i < size; ++i) {
        nrm2 += (x[i] - y[i]) * (x[i] - y[i]);
    }
    nrm2 = sqrt(nrm2);
    assert(nrm2 < std::numeric_limits<double>::epsilon());
}

int main(int argc, char **argv) {
    test(256);

    std::vector<int>    test_sizes = {{ 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 }};
    std::vector<double> elap_times;

    elap_times.reserve(test_sizes.size());

    std::cout << std::setprecision(8);
    std::cout.setf(std::ios_base::fixed);

    for (size_t i = 0; i < test_sizes.size() - 1; ++i) {
        std::cout << test_sizes[i] << ",";
    }
    std::cout << test_sizes.back() << std::endl;

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
        elap_times.push_back(diff.count());
    }

    // Summarize LU performance
    for (size_t i = 0; i < test_sizes.size() - 1; ++i) {
        std::cout << elap_times[i] << ",";
    }
    std::cout << elap_times.back() << std::endl;

    elap_times.resize(0);

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
        elap_times.push_back(diff.count());
    }

    // Summarize LU w/ pivoting performance
    for (size_t i = 0; i < test_sizes.size() - 1; ++i) {
        std::cout << elap_times[i] << ",";
    }
    std::cout << elap_times.back() << std::endl;

    return 0;
}

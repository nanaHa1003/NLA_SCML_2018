#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <chrono>

// non block getrf
template <typename T>
void getrf2(int n, T *A, int lda) noexcept {
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

// trsm for side = left, uplo = lower, trans = n, diag = u
// Use for computing L_11^-1 * A_12
template <typename T>
void trsm_llnu(int m, int n, T *A, int lda, T *B, int ldb) noexcept {
    for (int i = 0; i < m; ++i) {
        for (int j = i + 1; j < m; ++j) {
            for (int k = 0; k < n; ++k) {
                B[k * ldb + j] -= A[i * lda + j] * B[k * ldb + i];
            }
        }
    }
}

// trsm for side = right, uplo = upper, trans = n, diag = n
// Use for computing A_21 * U_11^-1
template <typename T>
void trsm_runn(int m, int n, T *A, int lda, T *B, int ldb) noexcept {
    for (int i = 0; i < n; ++i) {
        T inv_scale = 1.0 / A[i * lda + i];
        for (int j = 0; j < m; ++j) {
            B[i * ldb + j] *= inv_scale;
        }

        for (int j = i + 1; j < n; ++j) {
            for (int k = 0; k < m; ++k) {
                B[j * ldb + k] -= A[j * lda + i] * B[i * ldb + k];
            }
        }
    }
}

// Use to construct the Schur's complement
template <typename T>
void gemm(
    int m, int n, int k,
    T alpha,
    T *A, int lda,
    T *B, int ldb,
    T beta,
    T *C, int ldc) noexcept {
    for (int j = 0; j < n; ++j) {
        T *C_ptr = C + j * ldc;
        for (int i = 0; i < m; ++i) {
            C_ptr[i] *= beta;
        }
    }

    for (int i = 0; i < m; ++i) {
        T *B_ptr = B + i * ldb;
        for (int j = 0; j < n; ++j) {
            T *C_ptr = C + j * ldc + i;
            for (int l = 0; l < k; ++l) {
                *C_ptr += alpha * A[l * lda + j] * B_ptr[l];
            }
        }
    }
}

// block version of getrf
template <typename T, int bs = 64>
void getrf(int n, T *A, int lda) noexcept {
    if (n < bs) {
        getrf2(n, A, lda);
        return;
    }

    for (int i = 0; i < n; i += bs) {
        if (n - i > bs) {
            // Alias blocks
            // [ A11 A12 ]
            // [ A21 A22 ]
            T *A11 = A + i * lda + i;
            T *A12 = A11 + bs * lda;
            T *A21 = A11 + bs;
            T *A22 = A12 + bs;

            getrf2(bs, A11, lda);
            trsm_llnu(bs, n - i - bs, A11, lda, A12, lda);
            trsm_runn(n - i - bs, bs, A11, lda, A21, lda);
            gemm(n - i - bs, n - i - bs, bs, T(-1.0), A21, lda, A12, lda, T(1.0), A22, lda);
        } else {
            getrf2(n - i, A + i * lda + i, lda);
        }
    }
}

void test() noexcept {
    std::vector<double> A = {{
        4.0, 3.0, 2.0, 1.0,
        3.0, 4.0, 3.0, 2.0,
        2.0, 3.0, 4.0, 3.0,
        1.0, 2.0, 3.0, 4.0
    }};

    getrf<double, 2>(4, A.data(), 4);

    std::cout << std::setprecision(3);
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) {
            std::cout << A[j * 4 + i] << "\t";
        }
        std::cout << std::endl;
    }
}

int main(int argc, char **argv) {
    test();

    std::vector<int>    test_sizes = {{ 32, 64, 128, 256, 512, 1024, 2048 }};
    std::vector<double> elap_times;

    elap_times.reserve(test_sizes.size());

    std::cout << std::setprecision(8);
    std::cout.setf(std::ios_base::fixed);

    for (size_t i = 0; i < test_sizes.size() - 1; ++i) {
        std::cout << test_sizes[i] << ",";
    }
    std::cout << test_sizes.back() << std::endl;

    for (int &size: test_sizes) {
        std::vector<double> A(size * size);

        std::generate(A.begin(), A.end(), [](){
            return double(std::rand()) / RAND_MAX;
        });

        auto start = std::chrono::high_resolution_clock::now();
        getrf(size, A.data(), size);
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double> diff = end - start;
        elap_times.push_back(diff.count());
    }

    for (size_t i = 0; i < test_sizes.size() - 1; ++i) {
        std::cout << elap_times[i] << ",";
    }
    std::cout << elap_times.back() << std::endl;

    return 0;
}

#include <cassert>
#include <iostream>
#include <iomanip>
#include <vector>
#include <algorithm>
#include <limits>
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
    if (std::abs(beta - T(1.0)) > std::numeric_limits<T>::epsilon()) {
        for (int i = 0; i < m * n; ++i) {
            C[i] *= beta;
        }
    }

    for (int j = 0; j < n; ++j) {
        for (int l = 0; l < k; ++l) {
            for (int i = 0; i < m; ++i) {
                C[j * ldc + i] += alpha * B[j * ldb + l] * A[l * lda + i];
            }
        }
    }
}

// block version of getrf
template <typename T, int bs = 128>
void getrf(int n, T *A, int lda) noexcept {
    if (n < bs) {
        getrf2(n, A, lda);
        return;
    }

    for (int i = 0; i < n; i += bs) {
        if (n - i > bs) {
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

inline int min(int a, int b) noexcept {
    return std::min(a, b);
}

template <typename T, int bs = 64>
void getrf_omp_task(int n, T *A, int lda) {
    if (n < bs) {
        getrf2(n, A, lda);
        return;
    }

    #pragma omp parallel
    {
        #pragma omp single
        for (int i = 0; i < n; i += bs) {
            if (n - i > bs) {
                // Divide into blocks
                T *A11 = A + i * lda + i;

                #pragma omp task depend(inout: A[i * lda + i])
                getrf2(bs, A11, lda);

                for (int j = i + bs; j < n; j += bs) {
                    int rs = min(bs, n - j);

                    T *A1j = A + j * lda + i;
                    T *Aj1 = A + i * lda + j;

                    #pragma omp task depend(in: A[i * lda + i]) depend(inout: A[j * lda + i])
                    trsm_llnu(bs, rs, A11, lda, A1j, lda);

                    #pragma omp task depend(in: A[i * lda + i]) depend(inout: A[i * lda + j])
                    trsm_runn(rs, bs, A11, lda, Aj1, lda);
                }

                for (int j = i + bs; j < n; j += bs) {
                    for (int k = i + bs; k < n; k += bs) {
                        T *Aj1 = A + i * lda + j;
                        T *A1k = A + k * lda + i;
                        T *Ajk = A + k * lda + j;

                        int ms = min(bs, n - j);
                        int ns = min(bs, n - k);

                        #pragma omp task depend(in: A[i * lda + j], A[k * lda + i]) depend(inout: A[k * lda + j])
                        gemm(ms, ns, bs, T(-1.0), Aj1, lda, A1k, lda, T(1.0), Ajk, lda);
                    }
                }

            } else {
                #pragma omp task depend(inout: A[i * lda + i])
                getrf2(n - i, A + i * lda + i, lda);
            }
            #pragma omp taskwait
        }
    }
}

template <int size = 16, int bs = 8>
void test() noexcept {
    std::vector<double> A(size * size);
    std::generate(A.begin(), A.end(), [](){
        return 0.1 * static_cast<double>(rand()) / RAND_MAX;
    });

    for (int i = 0; i < size; ++i) {
        A[i * size + i] += 1.0;
    }
    std::vector<double> B = A;

    getrf2(size, A.data(), size);
    getrf_omp_task<double, bs>(size, B.data(), size);

    double norm = 0.0;
    for (int i = 0; i < size * size; ++i) {
        norm += (A[i] - B[i]) * (A[i] - B[i]);
    }
    assert(norm < std::numeric_limits<double>::epsilon());
}

int main(int argc, char **argv) {
    test<1024, 32>();

    std::vector<int>    test_sizes = {{ 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192 }};
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
        getrf_omp_task<double, 64>(size, A.data(), size);
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

#include <cassert>
#include <iostream>
#include <algorithm>
#include <chrono>

#include <cuda_runtime.h>

#include "ex4.h"

void full_to_csr_ref(
    int m, int n,
    double *A, int lda,
    int **rowptr, int **colidx, double **values) {
    *rowptr = new int[m + 1];

    int zero = 0;
    std::fill(*rowptr, *rowptr + m + 1, zero);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            rowptr[0][j + 1] += (A[i * lda + j]) ?1 :0;
        }
    }

    for (int i = 0; i < m; ++i) {
        rowptr[0][i + 1] += rowptr[0][i];
    }

    *colidx = new int[rowptr[0][m]];
    *values = new double[rowptr[0][m]];

    int pos = 0;
    for (int i = 0; i < m; ++i) {
        for (int j = 0; j < n; ++j) {
            if (A[i * lda + j]) {
                colidx[0][pos] = j;
                values[0][pos] = A[i * lda + j];
                ++pos;
            }
        }
    }
}

int test(int size) {
    if (size <= 0) return 0;

    double *A = new double[size * size];
    std::fill(A, A + size * size, 0.0);

    if (size > 1) {
      A[0] =  2.0;
      A[1] = -1.0;
      for (int i = 1; i < size - 1; ++i) {
        A[i * size + i - 1] = -1.0;
        A[i * size + i    ] =  2.0;
        A[i * size + i + 1] = -1.0;
      }
      A[size * size - 2] = -1.0;
      A[size * size - 1] =  2.0;
    } else if (size == 1) {
      A[0] = 2.0;
    }

    int *rowptr, *colidx;
    double *values;
    full_to_csr_ref(size, size, A, size, &rowptr, &colidx, &values);

    double *d_A;
    cudaError_t cudaErr = cudaMalloc(reinterpret_cast<void **>(&d_A), size * size * sizeof(double));
    assert(cudaErr == cudaSuccess);

    cudaErr = cudaMemcpy(d_A, A, size * size * sizeof(double), cudaMemcpyHostToDevice);
    assert(cudaErr == cudaSuccess);

    int *d_rowptr, *d_colidx;
    double *d_values;
    full_to_csr(size, size, d_A, size, &d_rowptr, &d_colidx, &d_values);

    // Verify results
    int    *h_rowptr = new int[size + 1];
    int    *h_colidx = new int[rowptr[size]];
    double *h_values = new double[rowptr[size]];

    cudaErr = cudaMemcpy(h_rowptr, d_rowptr, (size + 1) * sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaErr == cudaSuccess);
    cudaErr = cudaMemcpy(h_colidx, d_colidx, rowptr[size] * sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaErr == cudaSuccess);
    cudaErr = cudaMemcpy(h_values, d_values, rowptr[size] * sizeof(double), cudaMemcpyDeviceToHost);
    assert(cudaErr == cudaSuccess);

    int errcnt = 0;
    for (int i = 0; i < size + 1; ++i) {
      if (rowptr[i] != h_rowptr[i]) errcnt += 1;
    }
    for (int i = 0; i < rowptr[size]; ++i) {
      if (colidx[i] != h_colidx[i]) errcnt += 1;
    }
    for (int i = 0; i < rowptr[size]; ++i) {
      if (values[i] != h_values[i]) errcnt += 1;
    }

    cudaFree(d_rowptr);
    cudaFree(d_colidx);
    cudaFree(d_values);

    delete[] rowptr;
    delete[] colidx;
    delete[] values;
    delete[] h_rowptr;
    delete[] h_colidx;
    delete[] h_values;

    return errcnt;
}

int main() {
    assert(test(256) == 0);
    return 0;
}

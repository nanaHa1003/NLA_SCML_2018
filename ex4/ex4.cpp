#include <cassert>
#include <iostream>
#include <algorithm>
#include <chrono>

#include <cuda_runtime.h>

#include "ex4.h"

template <typename T>
void full_to_csr_ref(
    int m, int n,
    double *A, int lda,
    int **rowptr, int **colidx, double **values) {
    *rowptr = new double[m + 1];

    int zero = 0;
    std::fill(*rowptr, *rowptr + m + 1, zero);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            rowptr[0][j + 1] += (A[i * lda + j]) ?1 :0;
        }
    }

    for (i = 0; i < m; ++i) {
        rowptr[0][i + 1] += rowptr[0][i];
    }


}

template <typename T>
void test() {

}

int main() {
    return 0;
}

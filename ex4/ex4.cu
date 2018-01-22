#include <cassert>
#include <cuda_runtime.h>

__global__ void count_row_nnz(int m, int n, double *A, int lda, int *rownnz) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < m) {
        int     count = 0;
        double *value = A + tid;

        for (int i = 0; i < n; ++i) {
            count += (*value) ?1 :0;
            value += lda;
        }
        rownnz[tid] = count;
    }
}

__global__ void build_rowptr(int n, int *rowptr) {
    int tid = threadIdx.x + blockIdx.x + blockDim.x;
    // do something here
}

__global__ void fill_csr_values(
    int m, int n,
    double *A, int lda,
    int *rowptr, int *colidx, double *values) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < m) {
        int    idx = rowptr[tid];
        double ptr = A + tid;

        for (int i = 0; i < n; ++i) {
            if (*ptr) {
                colidx[idx] = i;
                values[idx] = *ptr;

                ++idx;
            }
            ptr += lda;
        }
    }
}

void full_to_csr(
    int m, int n,
    double *A, int lda,
    int **rowptr, int **colidx, double **values) {

    cudaError_t cudaErr;
    cudaErr = cudaMalloc(rowptr, (m + 1) * sizeof(int));
    assert(cudaErr == cudaSuccess);

    // Launch kernel to get number of nnz
    int bs = 1024, gs = (m - 1) / bs + 1;
    count_row_nnz<<<bs, gs>>>(m, n, A, lda, *rowptr);
    assert(cudaGetLastError() == cudaSuccess);

    // Perform prefix sum to build rowptr
    build_rowptr<<<bs, gs>>>(m, *rowptr);
    assert(cudaGetLastError() == cudaSuccess);

    int nnz = 0;
    cudaErr = cudaMemcpy(&nnz, *rowptr + m, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaErr == cudaSuccess);

    cudaErr = cudaMalloc(colidx, nnz * sizeof(int));
    assert(cudaErr == cudaSuccess);

    cudaErr = cudaMalloc(values, nnz * sizeof(double));
    assert(cudaErr == cudaSuccess);

    fill_csr_values<<<bs, gs>>>();
    assert(cudaGetLastError() == cudaSuccess);
}

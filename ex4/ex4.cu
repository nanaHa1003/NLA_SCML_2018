#include <cassert>
#include <iostream>
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

__global__ void block_scan(int n, int *array, int *buffer) {
    extern __shared__ int shm[]; // size of shm: 2 * blockDim.x * sizeof(int)

    int tid = threadIdx.x + (blockIdx.x << 1) * blockDim.x;
    if (tid < n) {
        // For block 0, 2, 4, 6, 8, ...
        shm[threadIdx.x] = array[tid];
    }
    if (tid + blockDim.x < n) {
        // For block 1, 3, 5, 7, 9, ...
        shm[threadIdx.x + blockDim.x] = array[tid + blockDim.x];
    }

    __syncthreads();

    int num_elements = (blockIdx.x != gridDim.x - 1) ?(blockDim.x << 1) :(n - blockDim.x * blockIdx.x);

    // Start up-sweep phase
    int idx = threadIdx.x << 1, shift = 1;
    while (idx + shift < num_elements) {
        shm[idx + shift] += shm[idx];
        idx   <<= 1;
        shift <<= 1;
    }

    __syncthreads();

    // Start down-sweep phase
    int sum = shm[num_elements - 1];
    if (threadIdx.x == 0) {
        shm[num_elements  - 1] = shm[num_elements << 1];
        shm[num_elements << 1] = 0;
    }
    idx   >>= 1;
    shift >>= 1;
    while (idx + shift > threadIdx.x) {
        int tmp = shm[idx + shift];
        shm[idx + shift] += shm[idx];
        shm[idx] = tmp;
    }

    __syncthreads();

    if (tid < n) {
        array[tid] = shm[threadIdx.x];
    }
    if (tid + blockDim.x < n) {
        array[tid + blockDim.x] = shm[threadIdx.x + blockDim.x];
    }

    if (threadIdx.x == 0) {
        buffer[blockIdx.x] = sum;
    }
}

__global__ void block_add(int n, int *array, int *buffer) {
    int tid = blockIdx.x * blockDim.x + (threadIdx.x << 1);

    int sum = buffer[blockIdx.x];
    if (tid < n) {
        array[tid] += sum;
    }
    if (tid + blockDim.x < n) {
        array[tid + blockDim.x] += sum;
    }
}

template <typename T>
inline T min(T a, T b) {
    return (a < b) ?a :b;
}

void build_rowptr(int n, int *rowptr) {
    int bs = min(1024, 32 * ((n - 1) / 64 + 1));
    int gs = (n - 1) / (bs << 1) + 1;

    int *buffer;
    cudaError_t cudaErr = cudaMalloc(reinterpret_cast<void **>(&buffer), gs * sizeof(int));
    assert(cudaErr == cudaSuccess);

    block_scan<<<gs , bs, 2 * bs * sizeof(int)>>>(n, rowptr, buffer);
    assert(cudaGetLastError() == cudaSuccess);

    // Need some modification
    block_add<<<gs, bs>>>(n + 1 - bs, rowptr + bs, buffer);
    assert(cudaGetLastError() == cudaSuccess);

    cudaFree(buffer);
}

__global__ void fill_csr_values(
    int m, int n,
    double *A, int lda,
    int *rowptr, int *colidx, double *values) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < m) {
        int     idx = rowptr[tid];
        double *ptr = A + tid;

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
    count_row_nnz<<<gs, bs>>>(m, n, A, lda, *rowptr);
    assert(cudaGetLastError() == cudaSuccess);

    build_rowptr(m, *rowptr);

    int nnz = 0;
    cudaErr = cudaMemcpy(&nnz, *rowptr + m, sizeof(int), cudaMemcpyDeviceToHost);
    assert(cudaErr == cudaSuccess);

    cudaErr = cudaMalloc(colidx, nnz * sizeof(int));
    assert(cudaErr == cudaSuccess);

    cudaErr = cudaMalloc(values, nnz * sizeof(double));
    assert(cudaErr == cudaSuccess);

    fill_csr_values<<<gs, bs>>>(m, n, A, lda, *rowptr, *colidx, *values);
    assert(cudaGetLastError() == cudaSuccess);
}

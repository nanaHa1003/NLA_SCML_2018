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

  // Initialize shm
  int tid = threadIdx.x + (blockIdx.x << 1) * blockDim.x;
  if (tid < n) {
    shm[threadIdx.x] = array[tid];
  } else {
    shm[threadIdx.x] = 0;
  }
  if (tid + blockDim.x < n) {
    shm[threadIdx.x + blockDim.x] = array[tid + blockDim.x];
  } else {
    shm[threadIdx.x + blockDim.x] = 0;
  }

  __syncthreads();

  int num_elements = blockDim.x << 1;

  // Start up-sweep phase
  int idx = threadIdx.x << 1, shift = 1;
  while (shift < num_elements) {
    if (idx + shift < num_elements) {
      shm[idx + shift] += shm[idx];
    }
    idx += idx + 1;
    shift <<= 1;
  }
  __syncthreads();

  // Start down-sweep phase
  int sum = shm[num_elements - 1];
  if (threadIdx.x == 0) {
    shm[num_elements - 1] = 0;
  }
  idx = (idx - 1) >> 1;
  shift >>= 1;
  while (shift > 0) {
    if (idx + shift < num_elements) {
      int tmp = shm[idx + shift];
      shm[idx + shift] += shm[idx];
      shm[idx] = tmp;
    }
    idx = (idx - 1) >> 1;
    shift >>= 1;
    __syncthreads();
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

__global__ void block_add(int n, int *array, int *buf) {
  int tid = blockIdx.x * (blockDim.x << 1) + threadIdx.x;

  int val = buf[blockIdx.x];
  if (tid < n) {
    array[tid] += val;
  }
  if (tid + blockDim.x < n) {
    array[tid + blockDim.x] += val;
  }
}

void exclusive_scan(int n, int *array) {
    int bs = 1024;
    int gs = (n - 1) / (bs << 1) + 1;

    int *buffer;
    cudaError_t cudaErr = cudaMalloc(reinterpret_cast<void **>(&buffer), gs * sizeof(int));
    assert(cudaErr == cudaSuccess);

    block_scan<<<gs , bs, 2 * bs * sizeof(int)>>>(n, array, buffer);
    assert(cudaGetLastError() == cudaSuccess);

    if (gs > 1) {
      block_add<<<gs - 1, bs>>>(n - (bs << 1), array + (bs << 1), buffer);
      assert(cudaGetLastError() == cudaSuccess);
    }

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

    exclusive_scan(m + 1, *rowptr);

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

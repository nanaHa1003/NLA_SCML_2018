// x += ay
__global__ void _axpy(int n, double a, double *x, int incx, double *y, int incy) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < n) {
        x[tid * incx] += a * y[tid * incy];
    }
}

__global__ void _gemv(int m, int n, double *A, int lda, double *x, int incx, double *y, int incy) {
    int tid = threadIdx.x + blockIdx.x * blockDim.x;
    if (tid < m) {
        double yi = 0;
        for (int i = 0; i < n; ++i) {
            yi += x[i * incx] * A[i * lda + tid];
        }
        y[tid * incy] = yi;
    }
}

void axpy(int n, double a, double *x, int incx, double *y, int incy) {
    int bs = 1024;
    int gs = (n - 1) / bs + 1;
    _axpy<<<gs, bs>>>(n, a, x, incx, y, incy);
}

void gemv(int m, int n, double *A, int lda, double *x, int incx, double *y, int incy) {
    int bs = 1024;
    int gs = (n - 1) / bs + 1;
    _gemv<<<gs, bs>>>(m, n, A, lda, x, incx, y, incy);
}

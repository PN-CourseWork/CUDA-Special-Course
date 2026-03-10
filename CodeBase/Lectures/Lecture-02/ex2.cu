// Exercise 2: Warp shuffle reduction + atomicAdd (slides 22-23)
// Instead of every thread doing atomicAdd, we first reduce within each warp (32 threads)
// using register-level shuffles, then only 1 thread per warp does atomicAdd — 32x fewer atomics
#include <stdio.h>
#define N (1 << 24)       // 16M elements
#define BLOCK_SIZE 256    // threads per block (= 8 warps per block)

__global__ void reduction_warp(double *a, int n, double *res) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // Out-of-bounds threads get 0.0 so they don't affect the sum
    double value = (idx < n) ? a[idx] : 0.0;

    // Phase 1 (slide 19): warp-level reduction using __shfl_down_sync (slide 22)
    // Each iteration halves the distance: 16, 8, 4, 2, 1
    // After 5 steps, lane 0 holds the sum of all 32 lanes
    // -1 mask = 0xffffffff = all 32 lanes participate
    for (int i = 16; i > 0; i /= 2)
        value += __shfl_down_sync(-1, value, i);

    // Phase 3 (slide 19): lane 0 of each warp writes to global result
    // threadIdx.x & 31 extracts the lane ID within the warp (0-31)
    if ((threadIdx.x & 31) == 0)
        atomicAdd(res, value);
}

int main() {
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate and initialize host array
    double *h_array = new double[N];
    for (int i = 0; i < N; i++) h_array[i] = 1.0;

    // Allocate GPU memory and copy input
    double *d_input, *d_res;
    cudaMalloc(&d_input, N * sizeof(double));
    cudaMalloc(&d_res, sizeof(double));
    cudaMemcpy(d_input, h_array, N * sizeof(double), cudaMemcpyHostToDevice);
    double zero = 0.0;
    cudaMemcpy(d_res, &zero, sizeof(double), cudaMemcpyHostToDevice);

    // GPU timing (slide 3)
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    reduction_warp<<<num_blocks, BLOCK_SIZE>>>(d_input, N, d_res);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    double sum;
    cudaMemcpy(&sum, d_res, sizeof(double), cudaMemcpyDeviceToHost);

    printf("sum  = %.0f (expected %d)\n", sum, N);
    printf("time = %.3f ms\n", ms);
    printf("BW   = %.1f GB/s\n", N * sizeof(double) / ms / 1e6);

    cudaFree(d_input); cudaFree(d_res);
    delete[] h_array;
}

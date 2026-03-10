// Exercise 4: Three-way benchmark — baseline vs warp shuffle vs warp+smem (slides 10-11, 21-22)
// Runs all three kernels back-to-back and prints a speed-up table
#include <stdio.h>
#define N (1 << 24)           // 16M elements
#define BLOCK_SIZE 256        // threads per block (= 8 warps per block)
#define WARPS_PER_BLOCK (BLOCK_SIZE / 32)

// --- Kernel 1: baseline — every thread atomicAdds (slide 21) ---
__global__ void reduction_baseline(double *a, int n, double *res) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) atomicAdd(res, a[idx]);
}

// --- Kernel 2: warp shuffle only — lane 0 of each warp atomicAdds (slide 22) ---
__global__ void reduction_warp(double *a, int n, double *res) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    double value = (idx < n) ? a[idx] : 0.0;

    for (int i = 16; i > 0; i /= 2)
        value += __shfl_down_sync(-1, value, i);

    if ((threadIdx.x & 31) == 0)
        atomicAdd(res, value);
}

// --- Kernel 3: warp shuffle + shared memory — thread 0 of each block atomicAdds (slides 10-11) ---
__global__ void reduction_smem(double *a, int n, double *res) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    double value = (idx < n) ? a[idx] : 0.0;

    // Phase 1: warp-level reduction in registers
    for (int i = 16; i > 0; i /= 2)
        value += __shfl_down_sync(-1, value, i);

    // Phase 2: collect one partial sum per warp into shared memory (slide 11)
    __shared__ double smem[WARPS_PER_BLOCK];
    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;

    if (lane == 0)
        smem[warp] = value;
    __syncthreads();

    // Phase 2b: first warp reduces the block's partial sums
    value = (threadIdx.x < WARPS_PER_BLOCK) ? smem[threadIdx.x] : 0.0;
    if (warp == 0) {
        for (int i = 16; i > 0; i /= 2)
            value += __shfl_down_sync(-1, value, i);

        // Phase 3: one atomicAdd per block
        if (lane == 0)
            atomicAdd(res, value);
    }
}

int main() {
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    double *h_array = new double[N];
    for (int i = 0; i < N; i++) h_array[i] = 1.0;

    double *d_input, *d_res;
    cudaMalloc(&d_input, N * sizeof(double));
    cudaMalloc(&d_res, sizeof(double));
    cudaMemcpy(d_input, h_array, N * sizeof(double), cudaMemcpyHostToDevice);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    double sum, zero = 0.0;
    float ms_baseline, ms_warp, ms_smem;

    // --- Run baseline ---
    cudaMemcpy(d_res, &zero, sizeof(double), cudaMemcpyHostToDevice);
    cudaEventRecord(t0);
    reduction_baseline<<<num_blocks, BLOCK_SIZE>>>(d_input, N, d_res);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms_baseline, t0, t1);
    cudaMemcpy(&sum, d_res, sizeof(double), cudaMemcpyDeviceToHost);
    printf("baseline  : sum = %.0f  time = %8.3f ms  BW = %6.1f GB/s\n",
           sum, ms_baseline, N * sizeof(double) / ms_baseline / 1e6);

    // --- Run warp shuffle ---
    cudaMemcpy(d_res, &zero, sizeof(double), cudaMemcpyHostToDevice);
    cudaEventRecord(t0);
    reduction_warp<<<num_blocks, BLOCK_SIZE>>>(d_input, N, d_res);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms_warp, t0, t1);
    cudaMemcpy(&sum, d_res, sizeof(double), cudaMemcpyDeviceToHost);
    printf("warp shfl : sum = %.0f  time = %8.3f ms  BW = %6.1f GB/s\n",
           sum, ms_warp, N * sizeof(double) / ms_warp / 1e6);

    // --- Run warp + smem ---
    cudaMemcpy(d_res, &zero, sizeof(double), cudaMemcpyHostToDevice);
    cudaEventRecord(t0);
    reduction_smem<<<num_blocks, BLOCK_SIZE>>>(d_input, N, d_res);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms_smem, t0, t1);
    cudaMemcpy(&sum, d_res, sizeof(double), cudaMemcpyDeviceToHost);
    printf("warp+smem : sum = %.0f  time = %8.3f ms  BW = %6.1f GB/s\n",
           sum, ms_smem, N * sizeof(double) / ms_smem / 1e6);

    printf("\nspeed-up warp vs baseline      : %.1fx\n", ms_baseline / ms_warp);
    printf("speed-up smem vs baseline      : %.1fx\n", ms_baseline / ms_smem);
    printf("speed-up smem vs warp-only     : %.1fx\n", ms_warp / ms_smem);

    cudaFree(d_input); cudaFree(d_res);
    delete[] h_array;
}

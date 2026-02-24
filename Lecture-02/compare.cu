// Compare: baseline atomicAdd (slide 21) vs warp shuffle (slides 22-23)
#include <stdio.h>
#define N (1 << 24)       // 16M elements
#define BLOCK_SIZE 256    // threads per block

// Baseline: every thread atomicAdds — extreme serialization (slide 21)
__global__ void reduction_baseline(double *a, int n, double *res) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) atomicAdd(res, a[idx]);
}

// Warp shuffle: reduce 32 lanes in registers, then lane 0 atomicAdds (slides 22, 19)
__global__ void reduction_warp(double *a, int n, double *res) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    double value = (idx < n) ? a[idx] : 0.0;

    // 5-step warp reduction: offsets 16, 8, 4, 2, 1 (slide 22)
    for (int i = 16; i > 0; i /= 2)
        value += __shfl_down_sync(-1, value, i);

    // Only lane 0 has the full warp sum — 32x fewer atomics than baseline
    if ((threadIdx.x & 31) == 0)
        atomicAdd(res, value);
}

int main() {
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    double *h_array = new double[N];
    for (int i = 0; i < N; i++) h_array[i] = 1.0;

    // Allocate GPU memory and copy input
    double *d_input, *d_res;
    cudaMalloc(&d_input, N * sizeof(double));
    cudaMalloc(&d_res, sizeof(double));
    cudaMemcpy(d_input, h_array, N * sizeof(double), cudaMemcpyHostToDevice);

    // Setup timing
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    double sum, zero = 0.0;
    float ms_baseline, ms_warp;

    // --- Run baseline ---
    cudaMemcpy(d_res, &zero, sizeof(double), cudaMemcpyHostToDevice);  // reset result to 0
    cudaEventRecord(t0);
    reduction_baseline<<<num_blocks, BLOCK_SIZE>>>(d_input, N, d_res);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms_baseline, t0, t1);
    cudaMemcpy(&sum, d_res, sizeof(double), cudaMemcpyDeviceToHost);
    printf("baseline     : sum = %.0f  time = %8.3f ms  BW = %6.1f GB/s\n",
           sum, ms_baseline, N * sizeof(double) / ms_baseline / 1e6);

    // --- Run warp shuffle ---
    cudaMemcpy(d_res, &zero, sizeof(double), cudaMemcpyHostToDevice);  // reset result to 0
    cudaEventRecord(t0);
    reduction_warp<<<num_blocks, BLOCK_SIZE>>>(d_input, N, d_res);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);
    cudaEventElapsedTime(&ms_warp, t0, t1);
    cudaMemcpy(&sum, d_res, sizeof(double), cudaMemcpyDeviceToHost);
    printf("warp shuffle : sum = %.0f  time = %8.3f ms  BW = %6.1f GB/s\n",
           sum, ms_warp, N * sizeof(double) / ms_warp / 1e6);

    printf("\nspeed-up: %.1fx\n", ms_baseline / ms_warp);

    cudaFree(d_input); cudaFree(d_res);
    delete[] h_array;
}

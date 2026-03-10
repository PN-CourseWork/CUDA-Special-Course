// Exercise 3: Warp shuffle + shared memory block reduction (slides 10-11)
// Phase 1: each warp reduces 32 lanes to 1 value in registers (__shfl_down_sync)
// Phase 2: lane 0 of each warp writes to shared memory; first warp reduces those partial sums
// Phase 3: thread 0 of the block does a single atomicAdd — 1 atomic per block (not per warp)
#include <stdio.h>
#define N (1 << 24)           // 16M elements
#define BLOCK_SIZE 256        // threads per block (= 8 warps per block)
#define WARPS_PER_BLOCK (BLOCK_SIZE / 32)

__global__ void reduction_smem(double *a, int n, double *res) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // Out-of-bounds threads contribute 0 so they don't affect the sum
    double value = (idx < n) ? a[idx] : 0.0;

    // Phase 1 (slide 10): warp-level reduction using __shfl_down_sync
    // After 5 steps lane 0 of each warp holds the sum of all 32 lanes
    for (int i = 16; i > 0; i /= 2)
        value += __shfl_down_sync(-1, value, i);

    // Phase 2 (slide 11): collect one partial sum per warp into shared memory
    // Shared memory has one slot per warp in this block
    __shared__ double smem[WARPS_PER_BLOCK];

    int lane = threadIdx.x & 31;          // lane ID within warp (0-31)
    int warp = threadIdx.x >> 5;          // warp ID within block  (0..WARPS_PER_BLOCK-1)

    if (lane == 0)
        smem[warp] = value;               // store warp partial sum (slide 11)

    __syncthreads();                      // wait until all warps have written

    // Phase 2b: the first warp reduces the WARPS_PER_BLOCK partial sums
    // Only the first warp participates; pad with 0 for unused slots
    if (threadIdx.x < WARPS_PER_BLOCK) {
        value = smem[threadIdx.x];
    } else {
        value = 0.0;
    }

    if (warp == 0) {
        // Reduce WARPS_PER_BLOCK values (≤ 32) using shuffle — no extra smem needed
        for (int i = 16; i > 0; i /= 2)
            value += __shfl_down_sync(-1, value, i);

        // Phase 3 (slide 10): only thread 0 writes the block result — 1 atomic per block
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
    double zero = 0.0;
    cudaMemcpy(d_res, &zero, sizeof(double), cudaMemcpyHostToDevice);

    // GPU timing (slide 3)
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    reduction_smem<<<num_blocks, BLOCK_SIZE>>>(d_input, N, d_res);
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

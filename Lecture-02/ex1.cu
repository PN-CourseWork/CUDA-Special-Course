// Exercise 1: Baseline reduction — every thread atomicAdds to one global address (slide 21)
#include <stdio.h>
#define N (1 << 24)       // 16M elements
#define BLOCK_SIZE 256    // threads per block

// __global__ means this runs on the GPU, launched from CPU
__global__ void reduction_baseline(double *a, int n, double *res) {
    // Each thread computes its unique index across all blocks
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // atomicAdd: thread-safe += to a single memory address (slide 20)
    // Every thread writes to the same address — causes extreme serialization
    if (idx < n) atomicAdd(res, a[idx]);
}

int main() {
    // Ceiling division to cover all N elements
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    // Allocate and initialize host (CPU) array — like np.ones(N)
    double *h_array = new double[N];
    for (int i = 0; i < N; i++) h_array[i] = 1.0;

    // Allocate GPU memory (like moving data to GPU in PyTorch: .cuda())
    double *d_input, *d_res;
    cudaMalloc(&d_input, N * sizeof(double));   // GPU array for input
    cudaMalloc(&d_res, sizeof(double));          // GPU scalar for result

    // Copy input from CPU -> GPU
    cudaMemcpy(d_input, h_array, N * sizeof(double), cudaMemcpyHostToDevice);
    // Initialize result to 0 on GPU
    double zero = 0.0;
    cudaMemcpy(d_res, &zero, sizeof(double), cudaMemcpyHostToDevice);

    // GPU timing with CUDA events (slide 3) — more accurate than CPU timers
    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);                        // start timer
    // Launch kernel: <<<num_blocks, threads_per_block>>>
    reduction_baseline<<<num_blocks, BLOCK_SIZE>>>(d_input, N, d_res);
    cudaEventRecord(t1);                        // stop timer
    cudaEventSynchronize(t1);                   // wait for GPU to finish

    // Get elapsed time in milliseconds
    float ms;
    cudaEventElapsedTime(&ms, t0, t1);
    // Copy result back from GPU -> CPU
    double sum;
    cudaMemcpy(&sum, d_res, sizeof(double), cudaMemcpyDeviceToHost);

    printf("sum  = %.0f (expected %d)\n", sum, N);
    printf("time = %.3f ms\n", ms);
    printf("BW   = %.1f GB/s\n", N * sizeof(double) / ms / 1e6);

    // Free GPU and CPU memory
    cudaFree(d_input); cudaFree(d_res);
    delete[] h_array;
}

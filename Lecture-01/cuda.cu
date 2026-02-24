// Reduction - Basic CUDA version
#include <stdio.h>

#define N (1 << 24)
#define BLOCK_SIZE 256

__global__ void reduce(double *input, double *output, int n) {
    __shared__ double sdata[BLOCK_SIZE];

    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    sdata[tid] = (i < n) ? input[i] : 0.0;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }

    if (tid == 0) output[blockIdx.x] = sdata[0];
}

int main() {
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    double *h_array = new double[N];
    for (int i = 0; i < N; i++) h_array[i] = 1.0;

    double *d_input, *d_output;
    cudaMalloc(&d_input, N * sizeof(double));
    cudaMalloc(&d_output, num_blocks * sizeof(double));
    cudaMemcpy(d_input, h_array, N * sizeof(double), cudaMemcpyHostToDevice);

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0);
    cudaEventCreate(&t1);

    cudaEventRecord(t0);

    // Reduce until single value
    int n = N;
    int blocks = num_blocks;
    reduce<<<blocks, BLOCK_SIZE>>>(d_input, d_output, n);
    while (blocks > 1) {
        n = blocks;
        blocks = (blocks + BLOCK_SIZE - 1) / BLOCK_SIZE;
        reduce<<<blocks, BLOCK_SIZE>>>(d_output, d_output, n);
    }

    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    double sum;
    cudaMemcpy(&sum, d_output, sizeof(double), cudaMemcpyDeviceToHost);

    float ms;
    cudaEventElapsedTime(&ms, t0, t1);

    printf("sum = %f (expected %d)\n", sum, N);
    printf("time = %f ms\n", ms);

    cudaFree(d_input);
    cudaFree(d_output);
    delete[] h_array;
    return 0;
}

// Reduction - OpenMP GPU Offload version
#include <stdio.h>
#include <omp.h>

#define N (1 << 24)

int main() {
    double *array = new double[N];
    for (int i = 0; i < N; i++) array[i] = 1.0;

    double sum = 0.0;

    // Time data transfer to GPU
    double t_transfer_start = omp_get_wtime();
    #pragma omp target enter data map(to:array[0:N])
    double t_transfer_end = omp_get_wtime();

    // Time kernel execution only
    double t_kernel_start = omp_get_wtime();
    #pragma omp target teams distribute parallel for reduction(+:sum)
    for (int i = 0; i < N; i++) {
        sum += array[i];
    }
    double t_kernel_end = omp_get_wtime();

    // Cleanup GPU memory
    #pragma omp target exit data map(delete:array[0:N])

    double t_transfer = (t_transfer_end - t_transfer_start) * 1000.0;
    double t_kernel = (t_kernel_end - t_kernel_start) * 1000.0;
    double t_total = t_transfer + t_kernel;

    printf("sum = %f (expected %d)\n", sum, N);
    printf("\nTiming breakdown:\n");
    printf("  transfer = %.3f ms\n", t_transfer);
    printf("  kernel   = %.3f ms\n", t_kernel);
    printf("  total    = %.3f ms\n", t_total);
    printf("\nBandwidth (kernel only) = %.2f GB/s\n", (N * sizeof(double)) / (t_kernel * 1e6));

    delete[] array;
    return 0;
}




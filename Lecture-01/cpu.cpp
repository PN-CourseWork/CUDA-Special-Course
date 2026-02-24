// Reduction - CPU version
#include <stdio.h>
#include <omp.h>

#define N (1 << 24)

int main() {
    double *array = new double[N];
    for (int i = 0; i < N; i++) array[i] = 1.0;

    double sum = 0.0;
    double t0 = omp_get_wtime();

    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++) {
        sum += array[i];
    }

    double t1 = omp_get_wtime();

    printf("sum = %f (expected %d)\n", sum, N);
    printf("time = %f ms\n", (t1 - t0) * 1000.0);

    delete[] array;
    return 0;
}

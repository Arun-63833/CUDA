#include <stdio.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

// Kernel for vector addition on GPU
__global__ void vectoradd(float *a, float *b, float *c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

// CPU implementation of vector addition
void vectoradd_cpu(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    FILE *fp = fopen("timings.txt", "w");
    if (!fp) {
        printf("Failed to open timings file.\n");
        return -1;
    }

    float *a, *b, *c, *c_cpu;
    float *a_d, *b_d, *c_d;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // Test different sizes of n
    int test_sizes[] = {1000000, 10000000, 50000000, 100000000, 500000000};
    int num_tests = sizeof(test_sizes) / sizeof(test_sizes[0]);

    for (int test = 0; test < num_tests; test++) {
        int n = test_sizes[test];
        int size = n * sizeof(float);
        
        // Allocate memory on host
        a = (float *)malloc(size);
        b = (float *)malloc(size);
        c = (float *)malloc(size);
        c_cpu = (float *)malloc(size);

        if (!a || !b || !c || !c_cpu) {
            fprintf(stderr, "Host memory allocation failed\n");
            return -1;
        }

        // Allocate memory on device
        cudaMalloc((void **)&a_d, size);
        cudaMalloc((void **)&b_d, size);
        cudaMalloc((void **)&c_d, size);

        // Initialize host data
        for (int i = 0; i < n; i++) {
            a[i] = (float)i;
            b[i] = (float)(n - i);
        }

        // Time CPU execution
        clock_t cpu_start = clock();
        vectoradd_cpu(a, b, c_cpu, n);
        clock_t cpu_end = clock();
        double cpu_time = (double)(cpu_end - cpu_start) / CLOCKS_PER_SEC;

        // Copy data to device
        cudaMemcpy(a_d, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(b_d, b, size, cudaMemcpyHostToDevice);

        // Time GPU execution
        cudaEventRecord(start);
        int threadperblock = 256;
        int blocksize = (n + threadperblock - 1) / threadperblock;
        vectoradd<<<blocksize, threadperblock>>>(a_d, b_d, c_d, n);
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        
        float gpu_time = 0;
        cudaEventElapsedTime(&gpu_time, start, stop);

        // Copy result back to host
        cudaMemcpy(c, c_d, size, cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(a_d);
        cudaFree(b_d);
        cudaFree(c_d);

        // Verify results (optional)
        for (int i = 0; i < 10; i++) {
            if (fabs(c[i] - c_cpu[i]) > 1e-5) {
                printf("Results do not match at index %d: CPU = %f, GPU = %f\n", i, c_cpu[i], c[i]);
            }
        }

        // Save timings to file
        fprintf(fp, "%d %f %f\n", n, cpu_time * 1000, gpu_time);

        // Free host memory
        free(a);
        free(b);
        free(c);
        free(c_cpu);
    }

    fclose(fp);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    return 0;
}

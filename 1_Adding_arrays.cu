    #include <stdio.h>
    #include <cuda.h>
    #include <cuda_runtime.h>
    #include <stdlib.h>
    #include <math.h>

    __global__ void vectoradd(float *a, float *b, float *c, int n)
    {
        int i = threadIdx.x + blockIdx.x * blockDim.x;
        if (i < n)
        {
            c[i] = a[i] + b[i];
        }
    }

    int main()
    {
        float *a, *b, *c;
        float *a_d, *b_d, *c_d;
        int n = 1000000;
        int size = n * sizeof(float);

        // Allocate memory on host
        a = (float *)malloc(size);
        b = (float *)malloc(size);
        c = (float *)malloc(size);

        if (!a || !b || !c) {
            fprintf(stderr, "Host memory allocation failed\n");
            return -1;
        }

        printf("Memory allocation on host successful\n");

        // Allocate memory on device
        cudaError_t err = cudaMalloc((void **)&a_d, size);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Error allocating memory for a_d: %s\n", cudaGetErrorString(err));
            return -1;
        }

        err = cudaMalloc((void **)&b_d, size);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Error allocating memory for b_d: %s\n", cudaGetErrorString(err));
            return -1;
        }

        err = cudaMalloc((void **)&c_d, size);
        if (err != cudaSuccess)
        {
            fprintf(stderr, "Error allocating memory for c_d: %s\n", cudaGetErrorString(err));
            return -1;
        }

        printf("Memory allocation on device successful\n");

        // Initialize host data
        for (int i = 0; i < n; i++)
        {
            a[i] = (float)i;
            b[i] = (float)(n - i);
        }

        // Copy data to device
        cudaMemcpy(a_d, a, size, cudaMemcpyHostToDevice);
        cudaMemcpy(b_d, b, size, cudaMemcpyHostToDevice);

        // Define kernel launch parameters
        int threadperblock = 256;
        int blocksize = (n + threadperblock - 1) / threadperblock;

        // Launch kernel
        vectoradd<<<blocksize, threadperblock>>>(a_d, b_d, c_d, n);

        // Copy result back to host
        cudaMemcpy(c, c_d, size, cudaMemcpyDeviceToHost);

        // Free device memory
        cudaFree(a_d);
        cudaFree(b_d);
        cudaFree(c_d);

        // Print results
        for (int i = 0; i < 10; i++) // Print first 10 values for verification
        {
            printf("c[%d] = %f\n", i, c[i]);
        }

        // Free host memory
        free(a);
        free(b);
        free(c);

        return 0;
    }

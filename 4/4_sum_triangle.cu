#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <ctime>

using namespace std;

// CUDA Kernel function to compute the sum of elements in the upper triangle of a matrix.
__global__ void sumTriangle(float* dm, float* dv, int n) {
    int j = threadIdx.x;  // Index in the current block (corresponds to a column in the matrix)

    // Initialize sum to 0
    float sum = 0;

    // Sum the upper triangle elements of the matrix
    for (int i = 0; i < j; i++) {
        sum += dm[i * n + j];
    }

    // Store the sum in the result vector
    dv[j] = sum;

    // Synchronize threads
    __syncthreads();

    // If it's the last thread, compute the total sum
    if (j == n - 1) {
        float totalSum = 0.0;
        for (int i = 0; i < n; i++) {
            totalSum += dv[i];
        }
        dv[n] = totalSum;  // Store total sum in dv[n]
    }
}

int main() {
    int N = 1024;          // Size of the matrix (N x N)
    int size = N * N;      // Number of elements in the matrix
    int V = N + 1;         // Size of the result vector (N + 1)

    float *hm, *hv;        // Host pointers for matrix and vector
    float *dm, *dv;        // Device pointers for matrix and vector

    // Allocate host memory for matrix and vector
    hm = new float[size];
    hv = new float[V];

    // Initialize matrix with random values
    srand(time(0));
    for (int i = 0; i < size; i++) {
        hm[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // Allocate device memory for matrix and vector
    cudaError_t err;
    err = cudaMalloc((void**)&dm, size * sizeof(float));
    if (err != cudaSuccess) {
        cout << "Error allocating memory for matrix on device" << endl;
        return -1;
    }

    err = cudaMalloc((void**)&dv, V * sizeof(float));
    if (err != cudaSuccess) {
        cout << "Error allocating memory for vector on device" << endl;
        return -1;
    }

    // Copy data from host to device (matrix)
    err = cudaMemcpy(dm, hm, size * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cout << "Error copying matrix to device" << endl;
        return -1;
    }

    // Initialize result vector to 0 on the device
    cudaMemset(dv, 0, V * sizeof(float));

    // Launch the kernel with 1 block and N threads (one per column)
    dim3 grid(1, 1, 1);
    dim3 block(N, 1, 1);
    sumTriangle<<<grid, block>>>(dm, dv, N);

    // Check for errors during kernel launch
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        cout << "Error launching kernel: " << cudaGetErrorString(err) << endl;
        return -1;
    }

    // Synchronize the device with the host
    cudaDeviceSynchronize();

    // Copy the result back from device to host
    err = cudaMemcpy(hv, dv, V * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cout << "Error copying result vector to host" << endl;
        return -1;
    }

    // Output the result vector
    cout << "Column-wise sums:" << endl;
    for (int i = 0; i < N; i++) {
        cout << hv[i] << "\t";
    }
    cout << "\nTotal sum of upper triangle: " << hv[N] << endl;

    // Free device memory
    cudaFree(dm);
    cudaFree(dv);

    // Free host memory
    delete[] hm;
    delete[] hv;

    return 0;
}

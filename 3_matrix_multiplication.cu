#include <iostream>
#include <random>
#include <cuda.h>
#include <cuda_runtime.h>

using namespace std;

// CUDA kernel for matrix multiplication
__global__ void matmul(float *da, float *db, float *dc, int size) {
    int i = blockDim.y * blockIdx.y + threadIdx.y;
    int j = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < size && j < size) {
        float sum = 0;
        for (int k = 0; k < size; k++) {
            sum += da[i * size + k] * db[k * size + j];
        }
        dc[i * size + j] = sum;
    }
}

int main() {
    // Declare matrices
    float *a, *b, *c;
    int size = 16;
    int matrixSize = size * size;

    // Random number generation
    random_device rd;
    mt19937 gen(rd());
    uniform_real_distribution<> float_dist(0.0, 1.0);

    // Host memory allocation
    a = new float[matrixSize];
    b = new float[matrixSize];
    c = new float[matrixSize];

    // Initialize matrices a and b with random values
    for (int i = 0; i < matrixSize; i++) {
        a[i] = float_dist(gen);
        b[i] = float_dist(gen);
    }

    // Device memory allocation
    float *da, *db, *dc;
    cudaError_t err;

    err = cudaMalloc((void **)&da, matrixSize * sizeof(float));
    if (err != cudaSuccess) {
        cerr << "Error while allocating memory for da: " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&db, matrixSize * sizeof(float));
    if (err != cudaSuccess) {
        cerr << "Error while allocating memory for db: " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }

    err = cudaMalloc((void **)&dc, matrixSize * sizeof(float));
    if (err != cudaSuccess) {
        cerr << "Error while allocating memory for dc: " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }

    // Copy data from host to device
    err = cudaMemcpy(da, a, matrixSize * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cerr << "Error while copying memory to da: " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(db, b, matrixSize * sizeof(float), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        cerr << "Error while copying memory to db: " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }

    // Define grid and block sizes
    dim3 block(8, 8, 1);
    dim3 grid((size + block.x - 1) / block.x, (size + block.y - 1) / block.y, 1);

    // Launch matrix multiplication kernel
    matmul<<<grid, block>>>(da, db, dc, size);

    // Copy result back to host
    err = cudaMemcpy(c, dc, matrixSize * sizeof(float), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        cerr << "Error while copying memory to c: " << cudaGetErrorString(err) << endl;
        exit(EXIT_FAILURE);
    }

    // Print the result matrix c
    for (int i = 0; i < matrixSize; i++) {
        if (i % size == 0) cout << endl;
        cout << c[i] << " ";
    }
    cout << endl;

    // Free device and host memory
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
    delete[] a;
    delete[] b;
    delete[] c;

    return 0;
}

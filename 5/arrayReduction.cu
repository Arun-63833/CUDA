#include<iostream>
using namespace std;

#include<cuda.h>
#include<cuda_runtime.h>
#include<cstdlib>
#include<time.h>

__global__ void reduce(float *arr ,  int n)
{
    int j = threadIdx.x;

    for (int s = 1 ; s < n ; s *= 2)
    {
        if ((j % (2* s) == 0) && (j + s) < n)
        {
            arr[j] += arr[j + s];
        }
        __syncthreads();

    }

}

int main()
{
    int N = 1024;

    float *hm, *dm;


    // host memory
    hm = new float[N];
    srand(time(0));
    for (int i = 0; i < N; i++) {
        hm[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    //device memory

    cudaError_t err = cudaSuccess;

    err = cudaMalloc((void ** ) &dm, sizeof(float) * N);

    if (err != CUDA_SUCCESS)
    {
        cout <<"Issue in allocating memory";
    }

    //memory cpy

    err = cudaMemcpy(dm , hm , N * sizeof(float) ,cudaMemcpyHostToDevice );
    if (err != CUDA_SUCCESS)
    {
        cout <<"Issue in copying ";
    }
    
    // initialising kernel

    dim3 grid(1,1,1);
    dim3 block(N,1,1);

    reduce<<<grid,block>>>(dm, N);

    //memory cpy

    err = cudaMemcpy(hm , dm , N * sizeof(float) ,cudaMemcpyDeviceToHost );
    if (err != CUDA_SUCCESS)
    {
        cout <<"Issue in copying ";
    }

    cout << "result"<<hm[0];

    cudaFree(dm);
    delete[] hm;
    return 0;

}
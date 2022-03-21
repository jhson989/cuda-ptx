#include "../include/cuda_c.cuh"

__global__ void matmul_basic(const int* A, const int* B, int* C, const int M, const int N, const int K) {

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y<M && x<N) {
        
        int sum = 0;
        for (int k=0; k<K; k++) {
            sum += A[y*K+k]*B[k*N+x];
        }
        C[y*N+x] = sum;
    }

}

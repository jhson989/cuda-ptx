#include "../include/cuda_c.cuh"

#ifdef SHARED

__global__ void matmul_shared(const int* A, const int* B, int* C, const int M, const int N, const int K) {

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int sy = threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int sx = threadIdx.x;

    extern __shared__ float smem[];
    float *sA = &smem[0];
    float *sB = &smem[blockDim.x*blockDim.x];

    int sum = 0;
    for (int t=0; t<K; t+=blockDim.x) {
        
        if (y<M && sx+t<K) {
            sA[sy*blockDim.x+sx] = A[y*K+(sx+t)];
        } else {
            sA[sy*blockDim.x+sx] = 0;
        }

        if (x<N && sy+t<K) {
            sB[sy*blockDim.x+sx] = B[(sy+t)*N+x];
        } else {
            sB[sy*blockDim.x+sx] = 0;
        }

        __syncthreads();

        for (int k=0; k<blockDim.x; k++) {
            sum += sA[sy*blockDim.x+k]*sB[k*blockDim.x+sx];
        }

        __syncthreads();

    }

    if (y<M && x<N) {
        C[y*N+x] = sum;
    }

}



#else

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

#endif
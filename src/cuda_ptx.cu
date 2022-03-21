#include "../include/cuda_ptx.cuh"



__global__ void matmul_ptx_s32(const int* A, const int* B, int* C, const int M, const int N, const int K) {

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y<M && x<N) {

        // int sum = 0;
        asm(".reg .s32 t1;\n\t"
            "mov.s32 t1, 0;"
           ); 
    
        for (int k=0; k<K; k++) {
            asm("mad.lo.s32 t1, %0, %1, t1;" : :"r"(A[y*K+k]), "r"(B[k*N+x])) ; // sum += A[y*K+k]*B[k*N+x];
        }

        asm("mov.s32 %0, t1;" : "=r"(C[y*N+x])); // C[y*N+x] = sum;
    }

}

#include "../include/cuda_ptx.cuh"



__global__ void matmul_ptx_s32(const int* A, const int* B, int* C, const int M, const int N, const int K) {

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    asm(".reg .b32 t_sum;");
    asm(".reg .b32 t_M;\n\tmov.s32 t_M, %0;" : : "r"(M));
    asm(".reg .b32 t_N;\n\tmov.s32 t_N, %0;" : : "r"(N));
    asm(".reg .b32 t_K;\n\tmov.s32 t_K, %0;" : : "r"(K));
    asm(".reg .b64 t_A;\n\tmov.b64 t_A, %0;" : : "l"(A));
    asm(".reg .b64 t_B;\n\tmov.b64 t_B, %0;" : : "l"(B));
    asm(".reg .b64 t_C;\n\tmov.b64 t_C, %0;" : : "l"(C));

    asm(".reg .b64 t_a_mem;");
    asm(".reg .b32 t_a_mem_temp;");
    asm(".reg .b32 t_a;");

    asm(".reg .b64 t_b_mem;");
    asm(".reg .b32 t_b_mem_temp;");
    asm(".reg .b32 t_b;");


    if (y<M && x<N) {

        // int sum = 0;
        asm("mov.s32 t_sum, 0;"); 
    
        for (int k=0; k<K; k++) {

            // t_a_mem = A[y*K+k]
            asm("mad.lo.s32 t_a_mem_temp, %0, t_K, %1;": : "r"(y), "r"(k));
            asm("mul.wide.s32 t_a_mem, t_a_mem_temp, 4;");
            asm("add.s64 t_a_mem, t_A, t_a_mem;");
            // t_b_mem = B[k*N+x]
            asm("mad.lo.s32 t_b_mem_temp, %0, t_N, %1;": : "r"(k), "r"(x));
            asm("mul.wide.s32 t_b_mem, t_b_mem_temp, 4;");
            asm("add.s64 t_b_mem, t_B, t_b_mem;");

            asm("ld.global.s32 t_a, [t_a_mem];");
            asm("ld.global.s32 t_b, [t_b_mem];");
            asm("mad.lo.s32 t_sum, t_a, t_b, t_sum;") ; // sum += A[y*K+k]*B[k*N+x];
        }

        asm("mov.s32 %0, t_sum;" : "=r"(C[y*N+x])); // C[y*N+x] = sum;
    }

}

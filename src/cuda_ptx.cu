#include "../include/cuda_ptx.cuh"



__global__ void matmul_ptx_s32(const int* A, const int* B, int* C, const int M, const int N, const int K) {

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    asm(".reg .pred %p<5>;");
    asm(".reg .s32 t_y;\n\tmov.s32 t_y, %0;" : : "r"(y));
    asm(".reg .s32 t_x;\n\tmov.s32 t_x, %0;" : : "r"(x));
    asm(".reg .s32 t_M;\n\tmov.s32 t_M, %0;" : : "r"(M));
    asm(".reg .s32 t_N;\n\tmov.s32 t_N, %0;" : : "r"(N));
    asm(".reg .s32 t_K;\n\tmov.s32 t_K, %0;" : : "r"(K));
    asm(".reg .b64 t_A;\n\tmov.b64 t_A, %0;" : : "l"(A));
    asm(".reg .b64 t_B;\n\tmov.b64 t_B, %0;" : : "l"(B));
    asm(".reg .b64 t_C;\n\tmov.b64 t_C, %0;" : : "l"(C));

   
    // if (y >= M || x >= N) return;
    asm("setp.ge.s32 %p1, t_y, t_M;");
    asm("setp.ge.s32 %p1, t_x, t_N;");
    asm("or.pred %p3, %p1, %p2;");
    asm("@%p3 bra RET;");

    asm(".reg .b64 t_a_mem;");
    asm(".reg .b32 t_a_mem_temp;");
    asm(".reg .b32 t_a;");

    asm(".reg .b64 t_b_mem;");
    asm(".reg .b32 t_b_mem_temp;");
    asm(".reg .b32 t_b;");

    asm(".reg .b64 t_c_mem;");
    asm(".reg .b32 t_c_mem_temp;");
    asm(".reg .b32 t_c;");

    // int sum = 0; int k = 0;
    asm(".reg .s32 t_k;\n\tmov.s32 t_k, 0;");
    asm(".reg .s32 t_sum;\n\tmov.s32 t_sum, 0;");

    // Loop start
    asm("Loop_start:");
    asm("setp.ge.s32 %p4, t_k, t_K;");
    asm("@%p4 bra Loop_end;");

    // t_a_mem = A[y*K+k]
    asm("mad.lo.s32 t_a_mem_temp, t_y, t_K, t_k;");
    asm("mul.wide.s32 t_a_mem, t_a_mem_temp, 4;");
    asm("add.u64 t_a_mem, t_A, t_a_mem;");
    asm("ld.global.s32 t_a, [t_a_mem];");

    // t_b_mem = B[k*N+x]
    asm("mad.lo.s32 t_b_mem_temp, t_k, t_N, t_x;");
    asm("mul.wide.s32 t_b_mem, t_b_mem_temp, 4;");
    asm("add.u64 t_b_mem, t_B, t_b_mem;");
    asm("ld.global.s32 t_b, [t_b_mem];");

    // sum += A[y*K+k]*B[k*N+x];
    asm("mad.lo.s32 t_sum, t_a, t_b, t_sum;") ; 


    asm("add.s32 t_k, t_k, 1;");
    asm("bra Loop_start;");
    asm("Loop_end:");
    // Loop end

    asm("mad.lo.s32 t_c_mem_temp, t_y, t_N, t_x;"); // y*N+x
    asm("mul.wide.s32 t_c_mem, t_c_mem_temp, 4;");
    asm("add.u64 t_c_mem, t_C, t_c_mem;"); 
    asm("st.global.s32 [t_c_mem], t_sum;"); // C[y*N+x] = sum;


    asm("RET:");


}

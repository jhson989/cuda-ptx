#include "../include/cuda_ptx.cuh"



__global__ void matmul_ptx_s32(const int* A, const int* B, int* C, const int M, const int N, const int K) {

    // Input registers : M N K A B C
    asm(".reg .pred %p<5>;"
        ".reg .s32 t_M;\n\tmov.s32 t_M, %0;\n\t"
        ".reg .s32 t_N;\n\tmov.s32 t_N, %1;\n\t"
        ".reg .s32 t_K;\n\tmov.s32 t_K, %2;\n\t"
        ".reg .b64 t_A;\n\tmov.b64 t_A, %3;\n\t"
        ".reg .b64 t_B;\n\tmov.b64 t_B, %4;\n\t"
        ".reg .b64 t_C;\n\tmov.b64 t_C, %5;\n\t"
        : : "r"(M), "r"(N), "r"(K), "l"(A), "l"(B), "l"(C));

    // Set 2D thread id (y, x) : y = blockIdx.y * blockDim.y + threadIdx.y;
    asm(".reg .s32 t_y;\n\t"
        ".reg .s32 ntim_y;\n\tmov.s32 ntim_y, %ntid.y;\n\t"
        ".reg .s32 ctaid_y;\n\tmov.s32 ctaid_y, %ctaid.y;\n\t"
        ".reg .s32 tid_y;\n\tmov.s32 tid_y, %tid.y;\n\t"
        "mad.lo.s32 t_y, ntim_y, ctaid_y, tid_y;");

    // Set 2D thread id (y, x) : x = blockIdx.x * blockDim.x + threadIdx.x;
    asm(".reg .s32 t_x;\n\t"
        ".reg .s32 ntim_x;\n\tmov.s32 ntim_x, %ntid.y;\n\t"
        ".reg .s32 ctaid_x;\n\tmov.s32 ctaid_x, %ctaid.y;\n\t"
        ".reg .s32 tid_x;\n\tmov.s32 tid_x, %tid.y;\n\t"
        "mad.lo.s32 t_x, ntim_x, ctaid_x, tid_x;");

    // Only valid thread whose id (y<M, x<N) is executed
    // if (y >= M || x >= N) return;
    asm("setp.ge.s32 %p1, t_y, t_M;\n\t"
        "setp.ge.s32 %p1, t_x, t_N;\n\t"
        "or.pred %p3, %p1, %p2;\n\t"
        "@%p3 bra RET;");

    // Temp registers for calculating global memory address
    asm(".reg .b64 t_a_mem;\n\t"
        ".reg .b32 t_a_mem_temp;\n\t"
        ".reg .b32 t_a;\n\t"
        ".reg .b64 t_b_mem;\n\t"
        ".reg .b32 t_b_mem_temp;\n\t"
        ".reg .b32 t_b;\n\t"
        ".reg .b64 t_c_mem;\n\t"
        ".reg .b32 t_c_mem_temp;\n\t"
        ".reg .b32 t_c;");



    /*******************************************/
    /*** Loop start ***/
    /*******************************************/

    // Initial values : sum, k (int sum = 0; int k = 0;)
    asm(".reg .s32 t_k;\n\tmov.s32 t_k, 0;\n\t"
        ".reg .s32 t_sum;\n\tmov.s32 t_sum, 0;");

    // for (k=0; k<K; k++)
    asm("Loop_start:\n\t"
        "setp.ge.s32 %p4, t_k, t_K;\n\t"
        "@%p4 bra Loop_end;");

    // t_a_mem = address of A[y*K+k], t_a = value of A[y*K+k]
    asm("mad.lo.s32 t_a_mem_temp, t_y, t_K, t_k;\n\t"
        "mul.wide.s32 t_a_mem, t_a_mem_temp, 4;\n\t"
        "add.u64 t_a_mem, t_A, t_a_mem;\n\t"
        "ld.global.s32 t_a, [t_a_mem];");

    // t_b_mem = address of B[k*N+x], t_b = value of B[k*N+x]
    asm("mad.lo.s32 t_b_mem_temp, t_k, t_N, t_x;\n\t"
        "mul.wide.s32 t_b_mem, t_b_mem_temp, 4;\n\t"
        "add.u64 t_b_mem, t_B, t_b_mem;\n\t"
        "ld.global.s32 t_b, [t_b_mem];");

    // sum += A[y*K+k]*B[k*N+x];
    asm("mad.lo.s32 t_sum, t_a, t_b, t_sum;\n\t") ; 

    // k++
    asm("add.s32 t_k, t_k, 1;\n\t"
        "bra Loop_start;\n\t"
        "Loop_end:");

    /*******************************************/
    /*** Loop end ***/
    /*******************************************/



    // t_c_mem = address of C[y*N+x], t_c = value of C[y*N+x]
    asm("mad.lo.s32 t_c_mem_temp, t_y, t_N, t_x;\n\t"
        "mul.wide.s32 t_c_mem, t_c_mem_temp, 4;\n\t"
        "add.u64 t_c_mem, t_C, t_c_mem;"); 

    // Store the result to global memory : C[y*N+x] = sum;
    asm("st.global.s32 [t_c_mem], t_sum;");

    // End of this kernel
    asm("RET:");

}

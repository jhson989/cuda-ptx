
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cstdlib>

#define DTYPE int
#define DEBUG_OFF
#define cudaErrChk(ans) { cudaAssert((ans), __FILE__, __LINE__); }
inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort=true);
void check_result(std::vector<DTYPE>& A, std::vector<DTYPE>& B, std::vector<DTYPE>& C);

int M = 1024*10+0;
int N = 1024*10+0;
int K = 1024*10+0;

/*******************************************************************
  * Kernel code
  ******************************************************************/

__global__ void matmul_basic(const DTYPE* A, const DTYPE* B, DTYPE* C, const int M, const int N, const int K) {

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y<M && x<N) {
        
        DTYPE sum = 0;
        for (int k=0; k<K; k++) {
            sum += A[y*K+k]*B[k*N+x];
        }
        C[y*N+x] = sum;
    }

}


__global__ void matmul_ptx_s32(const DTYPE* A, const DTYPE* B, DTYPE* C, const int M, const int N, const int K) {

    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int x = blockIdx.x * blockDim.x + threadIdx.x;

    if (y<M && x<N) {

        // DTYPE sum = 0;
        asm(".reg .s32 t1;\n\t"
            "mov.s32 t1, 0;"
           ); 
    
        for (int k=0; k<K; k++) {
            asm("mad.lo.s32 t1, %0, %1, t1;" : :"r"(A[y*K+k]), "r"(B[k*N+x])) ; // sum += A[y*K+k]*B[k*N+x];
        }

        asm("mov.s32 %0, t1;" : "=r"(C[y*N+x])); // C[y*N+x] = sum;
    }

}




/*******************************************************************
  * Host code
  ******************************************************************/

void measure_basic(const DTYPE* d_A, DTYPE* d_B, DTYPE* d_C, int loop_exe=10) {

    printf("Basic kernel launched...\n");

    const dim3 dim_threads(16, 16);
    const dim3 dim_blocks((N+dim_threads.x-1)/dim_threads.x, (M+dim_threads.y-1)/dim_threads.y);
    matmul_basic<<<dim_blocks, dim_threads>>>(d_A, d_B, d_C, M, N, K); // for warmning up
    cudaErrChk( cudaDeviceSynchronize() );



    float gops = 1.0*M*K*N*1e-9*loop_exe;
    float msec_total = 0.0f;

    cudaEvent_t start, stop;
    cudaErrChk( cudaEventCreate(&start) );
    cudaErrChk( cudaEventCreate(&stop) );
    cudaErrChk( cudaEventRecord(start, NULL) );
    // Main body
    for (int i=0; i<loop_exe; i++) {
        matmul_basic<<<dim_blocks, dim_threads>>>(d_A, d_B, d_C, M, N, K);
    }
    // End of main body
    cudaErrChk( cudaEventRecord(stop, NULL) );
    cudaErrChk( cudaEventSynchronize(stop) );
    cudaErrChk( cudaEventElapsedTime(&msec_total, start, stop) );


    printf(" -- Total number of multiplications : %.3f Gops\n", gops/loop_exe);
    printf(" -- Avg. elapsed time: %.3f s\n", msec_total/loop_exe*1e-3);
    printf(" -- Avg. GFLOPS : %.3f\n", gops/(msec_total/loop_exe*1e-3));

}

void measure_ptx(const DTYPE* d_A, DTYPE* d_B, DTYPE* d_C, int loop_exe=10) {

    printf("PTX kernel launched...\n");

    const dim3 dim_threads(16, 16);
    const dim3 dim_blocks((N+dim_threads.x-1)/dim_threads.x, (M+dim_threads.y-1)/dim_threads.y);
    matmul_ptx_s32<<<dim_blocks, dim_threads>>>(d_A, d_B, d_C, M, N, K); // for warming up
    cudaErrChk( cudaDeviceSynchronize() );

    float gops = 1.0*M*K*N*1e-9*loop_exe;
    float msec_total = 0.0f;

    cudaEvent_t start, stop;
    cudaErrChk( cudaEventCreate(&start) );
    cudaErrChk( cudaEventCreate(&stop) );
    cudaErrChk( cudaEventRecord(start, NULL) );
    // Main body
    for (int i=0; i<loop_exe; i++) {
        matmul_ptx_s32<<<dim_blocks, dim_threads>>>(d_A, d_B, d_C, M, N, K);
    }
    // End of main body
    cudaErrChk( cudaEventRecord(stop, NULL) );
    cudaErrChk( cudaEventSynchronize(stop) );
    cudaErrChk( cudaEventElapsedTime(&msec_total, start, stop) );

    printf(" -- Total number of multiplications : %.3f Gops\n", gops/loop_exe);
    printf(" -- Avg. elapsed time: %.3f s\n", msec_total/loop_exe*1e-3);
    printf(" -- Avg. GFLOPS : %.3f\n", gops/(msec_total/loop_exe*1e-3));

}



DTYPE init_value() {
    return std::rand()%11-5;
}


int main(void) {

    printf("\n************************************************\n");
    printf("PTX code example - matrix multiplication\n");
    printf(" -- A[%d, %d] * B[%d, %d] = C[%d, %d]\n", M, K, K, N, M, N);
    printf(" -- Total usage of memory : %.3f GB\n", (1.0f*(M*K+K*N+M*N)*sizeof(DTYPE))/(1<<30));
    printf("************************************************\n\n");


    /************************************
      * Data Initialization
      ***********************************/

    // Input matrix A
    std::vector<DTYPE> A(M*K);
    std::generate(A.begin(), A.end(), init_value);
    // Input matrix B
    std::vector<DTYPE> B(K*N);
    std::generate(B.begin(), B.end(), init_value);
    // Input matrix C
    std::vector<DTYPE> C(M*N);

    // Alloc GPU memory
    DTYPE *d_A, *d_B, *d_C;
    cudaErrChk( cudaMalloc((void**)&d_A, sizeof(DTYPE)*M*K) );
    cudaErrChk( cudaMalloc((void**)&d_B, sizeof(DTYPE)*K*N) );
    cudaErrChk( cudaMalloc((void**)&d_C, sizeof(DTYPE)*M*N) );
    
    // Memcpy from host to device
    cudaErrChk( cudaMemcpy(d_A, A.data(), sizeof(DTYPE)*M*K, cudaMemcpyHostToDevice) );
    cudaErrChk( cudaMemcpy(d_B, B.data(), sizeof(DTYPE)*K*N, cudaMemcpyHostToDevice) );
    cudaErrChk( cudaDeviceSynchronize() );
    cudaErrChk( cudaGetLastError() );
    
    /************************************
      * Run kernel
      ***********************************/

    // Basic matrix multiplication
    measure_basic(d_A, d_B, d_C);
    #ifdef DEBUG_ON
    cudaErrChk( cudaMemcpy(C.data(), d_C, sizeof(DTYPE)*M*N, cudaMemcpyDeviceToHost) );
    cudaErrChk( cudaDeviceSynchronize() );
    check_result(A, B, C);
    #endif

    // PTX matrix multiplication
    measure_ptx(d_A, d_B, d_C);
    #ifdef DEBUG_ON
    cudaErrChk( cudaMemcpy(C.data(), d_C, sizeof(DTYPE)*M*N, cudaMemcpyDeviceToHost) );
    cudaErrChk( cudaDeviceSynchronize() );
    check_result(A, B, C);
    #endif


    /*** Finalize ***/
    cudaErrChk( cudaFree(d_A) );
    cudaErrChk( cudaFree(d_B) );
    cudaErrChk( cudaFree(d_C) );

    return 0;
}




/*******************************************************************
  * Debug code
  ******************************************************************/

inline void cudaAssert(cudaError_t code, const char *file, int line, bool abort) {
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"CUDA assert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}


void check_result(std::vector<DTYPE>& A, std::vector<DTYPE>& B, std::vector<DTYPE>& C) {
    
    printf(" -- Checking result ...\n");
    for (int y=0; y<M; y++) {
        for (int x=0; x<N; x++) {
            DTYPE sum = 0;
            for (int k=0; k<K; k++) {
                sum += A[y*K+k]*B[k*N+x];
            }
            if ( C[y*N+x]!= sum) {
                printf(" -- [[ERROR]] Checking result is failed at C[%d, %d](%d) != gt(%d)\n", y, x, C[y*N+x], sum);
                return;
            }
        }
    }
    printf(" -- Chekcing result succeed!!\n");

}




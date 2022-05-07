#pragma once

__global__ void matmul_ptx_s32_basic(const int* A, const int* B, int* C, const int M, const int N, const int K); 
__global__ void matmul_ptx_s32_shared(const int* A, const int* B, int* C, const int M, const int N, const int K); 

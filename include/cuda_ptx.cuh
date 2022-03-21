#pragma once

__global__ void matmul_ptx_s32(const int* A, const int* B, int* C, const int M, const int N, const int K); 

# cuda-ptx: Inline CUDA PTX Assembly Example for Matrix Multiplication

## 1. Introduction

This project shows how to write a CUDA kernel in **inline PTX (Parallel Thread eXecution) assembly**, using integer matrix multiplication (`C = A × B`) as the example.

The same kernel is written twice:

- once in plain **CUDA C**, as a reference
- once almost entirely in **inline PTX** (`asm(...)` statements)

The program runs both kernels on the same input, times them, and checks each result against a CPU computation.

A detailed explanation (in Korean) is available on the blog: https://computing-jhson.tistory.com/15

## 2. Project Structure

```
.
├── main.cu              # Host code: data setup, kernel launch, timing, result check
├── include/
│   ├── cuda_c.cuh       # Declarations of the CUDA C kernels
│   └── cuda_ptx.cuh     # Declarations of the inline-PTX kernels
├── src/
│   ├── cuda_c.cu        # Matrix multiplication in CUDA C
│   └── cuda_ptx.cu      # Matrix multiplication in inline CUDA PTX
└── Makefile
```

## 3. Kernels

| Mode     | CUDA C kernel   | PTX kernel              | Description                                   |
|----------|-----------------|-------------------------|-----------------------------------------------|
| `BASIC`  | `matmul_basic`  | `matmul_ptx_s32_basic`  | Naive kernel: each thread reads global memory directly |
| `SHARED` | `matmul_shared` | `matmul_ptx_s32_shared` | Tiled kernel that uses shared memory (PTX version **in progress**) |

Each thread computes one element `C[y, x]`, using a 2D grid of 16×16 thread blocks.

### What the PTX kernel shows

`src/cuda_ptx.cu` builds the CUDA C kernel's logic out of PTX instructions:

- **Passing kernel arguments into PTX registers**: `M`, `N`, `K` go in with the `"r"` constraint and pointers `A`, `B`, `C` with the `"l"` constraint, then `mov` copies them into named `.reg` variables.
- **Computing the thread index**: `%ntid`, `%ctaid` and `%tid` are combined with `mad.lo.s32` (`y = blockIdx.y * blockDim.y + threadIdx.y`).
- **Bounds checking with predicates**: `setp.ge.s32`, `or.pred`, then a predicated branch `@%p bra RET` replaces the `if` statement.
- **A `for` loop built from labels and branches**: `Loop_start`, `Loop_end`, `bra`.
- **Global memory address arithmetic and access**: `mul.wide.s32`, `add.u64`, `ld.global.s32`, `st.global.s32`.
- **Multiply-accumulate**: `mad.lo.s32 sum, a, b, sum`.

> **Note:** `matmul_ptx_s32_shared` is still a work in progress. For now it has the same logic as the basic PTX kernel and does not use shared memory yet.

## 4. Requirements

- An NVIDIA GPU
- The CUDA Toolkit (`nvcc`)

## 5. How to Build and Run

```bash
# Build (default mode: BASIC)
make

# Run
make run

# Build and run the shared-memory version
make clean
make MODE=SHARED
make run

# Remove the binary
make clean
```

`MODE` is passed to `nvcc` as a preprocessor define (`-DBASIC` or `-DSHARED`), which chooses which pair of kernels gets compiled. Changing `MODE` does not trigger a rebuild by itself, so run `make clean` before switching modes.

## 6. Configuration

You can change these in `main.cu`:

| Item | Default | Description |
|------|---------|-------------|
| `M`, `N`, `K` | `1025`, `1031`, `1035` | Matrix sizes (`A[M, K] × B[K, N] = C[M, N]`). They are deliberately not multiples of the block size, so the bounds checks are exercised. |
| `LOOP` | `2` | Number of times each kernel is launched for timing |
| `DEBUG_ON` | defined | When defined, each GPU result is checked against a CPU reference |

The input matrices are filled with random integers in the range `[-5, 5]`.

## 7. Sample Output

```
************************************************
PTX code example - matrix multiplication
 -- A[1025, 1035] * B[1035, 1031] = C[1025, 1031]
 -- Total usage of memory : 0.012 GB
************************************************

Basic kernel launched...
 -- Total number of multiplications : 1.094 Gops
 -- Avg. elapsed time: ... s
 -- Avg. GILOPS : ...
 -- Checking result ...
 -- Chekcing result succeed!!
PTX kernel launched...
 -- Total number of multiplications : 1.094 Gops
 -- Avg. elapsed time: ... s
 -- Avg. GILOPS : ...
 -- Checking result ...
 -- Chekcing result succeed!!
```

Timing and throughput depend on your GPU.

## 8. References

- [NVIDIA PTX ISA documentation](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html)
- [Inline PTX Assembly in CUDA](https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html)
- Blog post (Korean): https://computing-jhson.tistory.com/15

#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>

// ubsan_shift_oob.cpp
// Compile: clang++ -fsanitize=undefined -g ubsan_shift_oob.cpp
__device__ void hipsmith_kernel_logic(int argc, char **argv)
{
    int x = 1;
    x <<= 32; // BOOM: shift-out-of-bounds (shifting >= bit width)
    int y = x;
    return;
}

__device__ void hipsmith_kernel_logic(int argc, char **argv);
__global__ void hipsmith_kernel(int argc, char **argv) { hipsmith_kernel_logic(argc, argv); }
int main(int argc, char **argv)
{
    hipLaunchKernelGGL(hipsmith_kernel, dim3(1), dim3(1), 0, 0, argc, argv);
    hipDeviceSynchronize();
    return 0;
}

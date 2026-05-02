#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>

// ubsan_div_zero.cpp
// Compile: clang++ -fsanitize=undefined -g ubsan_div_zero.cpp
__device__ void hipsmith_kernel_logic(int argc, char** argv) {
    int zero = argc - 1; // 0 if run with no args
    int x = 10 / zero;  // BOOM: division-by-zero
    return;   
}

__device__ void hipsmith_kernel_logic(int argc, char** argv);
__global__ void hipsmith_kernel(int argc, char** argv) { hipsmith_kernel_logic(argc, argv); }
int main(int argc, char** argv) { hipLaunchKernelGGL(hipsmith_kernel, dim3(1), dim3(1), 0, 0, argc, argv); hipDeviceSynchronize(); return 0; }

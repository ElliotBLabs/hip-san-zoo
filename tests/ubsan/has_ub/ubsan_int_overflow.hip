#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>

// ubsan_int_overflow.cpp
// Compile: clang++ -fsanitize=undefined -g ubsan_int_overflow.cpp
__device__ void hipsmith_kernel_logic(int argc, char** argv) {
    int max = 2147483647; // INT_MAX
    max += 1;             // BOOM: signed-integer-overflow
    int x = max;
    return;
}

__device__ void hipsmith_kernel_logic(int argc, char** argv);
__global__ void hipsmith_kernel(int argc, char** argv) { hipsmith_kernel_logic(argc, argv); }
int main(int argc, char** argv) { hipLaunchKernelGGL(hipsmith_kernel, dim3(1), dim3(1), 0, 0, argc, argv); hipDeviceSynchronize(); return 0; }

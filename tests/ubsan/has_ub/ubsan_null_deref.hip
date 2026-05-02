#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>

// ubsan_null_deref.cpp
// Compile: clang++ -fsanitize=undefined -g ubsan_null_deref.cpp
__device__ void hipsmith_kernel_logic(int argc, char** argv) {
    int *p = nullptr;
    int x = *p;
    return; // BOOM: null-pointer-dereference
}

__device__ void hipsmith_kernel_logic(int argc, char** argv);
__global__ void hipsmith_kernel(int argc, char** argv) { hipsmith_kernel_logic(argc, argv); }
int main(int argc, char** argv) { hipLaunchKernelGGL(hipsmith_kernel, dim3(1), dim3(1), 0, 0, argc, argv); hipDeviceSynchronize(); return 0; }

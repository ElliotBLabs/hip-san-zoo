#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>

__device__ void hipsmith_kernel_logic(int argc, char** argv) {
    int array[100];
    int x = array[100]; // stack-buffer-overflow
    return; 
}

__device__ void hipsmith_kernel_logic(int argc, char** argv);

__global__ void hipsmith_kernel(int argc, char** argv) { 
    hipsmith_kernel_logic(argc, argv); 
}

int main(int argc, char** argv) { 
    hipLaunchKernelGGL(hipsmith_kernel, dim3(1), dim3(1), 0, 0, argc, argv); 
    hipDeviceSynchronize(); 
    return 0; 
}

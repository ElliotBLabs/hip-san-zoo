#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>

__device__ void hipsmith_kernel_logic(int argc, char** argv) {
    int *array = new int[100];
    delete[] array;
    int x = array[5]; // heap-use-after-free
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

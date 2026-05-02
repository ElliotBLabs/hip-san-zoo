#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>

__device__ void hipsmith_kernel_logic(int argc, char** argv) {
    int uninitialized_var;
    if (argc > 5) {
        uninitialized_var = 1;
    }
    // use-of-uninitialized-value (unless you pass > 5 args)
    int4 vec = make_int4(uninitialized_var, 2, 3, 4);
    if (vec == make_int4(0,0,0,0)) { 
        return;
    }
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

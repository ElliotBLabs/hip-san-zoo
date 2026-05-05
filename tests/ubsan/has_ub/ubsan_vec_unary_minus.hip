#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>

__device__ void hipsmith_kernel_logic(int argc, char **argv)
{
    int min = -2147483648; // INT_MIN
    int4 v = make_int4(min, 2, 1, 1);
    int4 v_res = -v;
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

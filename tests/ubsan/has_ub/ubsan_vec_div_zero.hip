#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>

__device__ void hipsmith_kernel_logic(int argc, char **argv)
{
    int zero = argc - 1; // 0 if run with no args

    int4 v_a = make_int4(10, 20, 30, 40);
    int4 v_b = make_int4(zero, 2, 2, 2);
    int4 v_res = v_a / v_b;
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
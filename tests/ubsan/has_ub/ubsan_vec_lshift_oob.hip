#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>

__device__ void hipsmith_kernel_logic(int argc, char** argv) {
    int4 v_base = make_int4(8, 1, 1, 1);
    int4 v_shifts = make_int4(29, 4, 5, 6);
    int4 v_res = v_base << v_shifts;
    return;
}

__device__ void hipsmith_kernel_logic(int argc, char** argv);
__global__ void hipsmith_kernel(int argc, char** argv) { hipsmith_kernel_logic(argc, argv); }
int main(int argc, char** argv) { hipLaunchKernelGGL(hipsmith_kernel, dim3(1), dim3(1), 0, 0, argc, argv); hipDeviceSynchronize(); return 0; }

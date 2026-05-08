#include <hip/hip_runtime.h>

// as these are both globals these must be zero initialsied and so no UB when tryign to read them
__constant__ int32_t uninit_hip_const;
int x;

__device__ void hipsmith_kernel_logic(int argc, char **argv)
{
    if (uninit_hip_const > 5)
    {
        return;
    }
    if (x > 6)
    {
        return;
    }
    return;
}

__device__ void hipsmith_kernel_logic(int argc, char **argv);

__global__ void hipsmith_kernel(int argc, char **argv)
{
    hipsmith_kernel_logic(argc, argv);
}

int main(int argc, char **argv)
{
    hipLaunchKernelGGL(hipsmith_kernel, dim3(1), dim3(1), 0, 0, argc, argv);
    hipDeviceSynchronize();
    return 0;
}
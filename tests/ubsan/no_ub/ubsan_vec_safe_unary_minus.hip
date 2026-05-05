#include <hip/hip_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <limits>
#include <type_traits>

// in the eyes of HIP we build ucharN so typedef supported types
typedef unsigned char uchar;
typedef unsigned short ushort;
typedef unsigned int uint;
typedef unsigned long ulong;

template <typename BaseT, typename ProxyT>
__host__ __device__ constexpr bool check_safe_unary_minus(ProxyT a_in)
{
    // in HIP the internal type of a vector is a proxy but we need properties of the c++ internal type
    BaseT a = static_cast<BaseT>(a_in);

    if constexpr (std::is_signed_v<BaseT>)
    {
        // are we negating TYPE MIN
        if (a == std::numeric_limits<BaseT>::min())
            return false;
        return true;
    }
    else
    {
        // no UB here
        return true;
    }
}

// generate an overloaded safe_unary_minus macro for every vector length and type combination supported
#define HIPSMITH_VEC1_SAFE_UNARY_MINUS(BASE_TYPE, VEC_TYPE)          \
    __host__ __device__ inline VEC_TYPE safe_unary_minus(VEC_TYPE a) \
    {                                                                \
        if (check_safe_unary_minus<BASE_TYPE>(a.x))                  \
            return -a;                                               \
        return a;                                                    \
    }

#define HIPSMITH_VEC2_SAFE_UNARY_MINUS(BASE_TYPE, VEC_TYPE)          \
    __host__ __device__ inline VEC_TYPE safe_unary_minus(VEC_TYPE a) \
    {                                                                \
        if (check_safe_unary_minus<BASE_TYPE>(a.x) &&                \
            check_safe_unary_minus<BASE_TYPE>(a.y))                  \
            return -a;                                               \
        return a;                                                    \
    }

#define HIPSMITH_VEC3_SAFE_UNARY_MINUS(BASE_TYPE, VEC_TYPE)          \
    __host__ __device__ inline VEC_TYPE safe_unary_minus(VEC_TYPE a) \
    {                                                                \
        if (check_safe_unary_minus<BASE_TYPE>(a.x) &&                \
            check_safe_unary_minus<BASE_TYPE>(a.y) &&                \
            check_safe_unary_minus<BASE_TYPE>(a.z))                  \
            return -a;                                               \
        return a;                                                    \
    }

#define HIPSMITH_VEC4_SAFE_UNARY_MINUS(BASE_TYPE, VEC_TYPE)          \
    __host__ __device__ inline VEC_TYPE safe_unary_minus(VEC_TYPE a) \
    {                                                                \
        if (check_safe_unary_minus<BASE_TYPE>(a.x) &&                \
            check_safe_unary_minus<BASE_TYPE>(a.y) &&                \
            check_safe_unary_minus<BASE_TYPE>(a.z) &&                \
            check_safe_unary_minus<BASE_TYPE>(a.w))                  \
            return -a;                                               \
        return a;                                                    \
    }

#define GENERATE_ALL_VEC_SAFE_UNARY_MINUS(BASE_TYPE)              \
    HIPSMITH_VEC1_SAFE_UNARY_MINUS(BASE_TYPE, BASE_TYPE##1)       \
    HIPSMITH_VEC2_SAFE_UNARY_MINUS(BASE_TYPE, BASE_TYPE##2)       \
    HIPSMITH_VEC3_SAFE_UNARY_MINUS(BASE_TYPE, BASE_TYPE##3)       \
    HIPSMITH_VEC4_SAFE_UNARY_MINUS(BASE_TYPE, BASE_TYPE##4)       \
    HIPSMITH_VEC1_SAFE_UNARY_MINUS(u##BASE_TYPE, u##BASE_TYPE##1) \
    HIPSMITH_VEC2_SAFE_UNARY_MINUS(u##BASE_TYPE, u##BASE_TYPE##2) \
    HIPSMITH_VEC3_SAFE_UNARY_MINUS(u##BASE_TYPE, u##BASE_TYPE##3) \
    HIPSMITH_VEC4_SAFE_UNARY_MINUS(u##BASE_TYPE, u##BASE_TYPE##4)

GENERATE_ALL_VEC_SAFE_UNARY_MINUS(char)
GENERATE_ALL_VEC_SAFE_UNARY_MINUS(short)
GENERATE_ALL_VEC_SAFE_UNARY_MINUS(int)
GENERATE_ALL_VEC_SAFE_UNARY_MINUS(long)

// Tests for Unary Minus Overflow (negating MIN)
#define TEST_UNARY_MINUS_UB_VEC1(TYPE)                   \
    {                                                    \
        TYPE min_val = std::numeric_limits<TYPE>::min(); \
        TYPE##1 v_a = make_##TYPE##1(min_val);           \
        TYPE##1 v_res = safe_unary_minus(v_a);           \
    }

#define TEST_UNARY_MINUS_UB_VEC2(TYPE)                   \
    {                                                    \
        TYPE min_val = std::numeric_limits<TYPE>::min(); \
        TYPE##2 v_a = make_##TYPE##2(1, min_val);        \
        TYPE##2 v_res = safe_unary_minus(v_a);           \
    }

#define TEST_UNARY_MINUS_UB_VEC3(TYPE)                     \
    {                                                      \
        TYPE min_val = std::numeric_limits<TYPE>::min();   \
        TYPE##3 v_a = make_##TYPE##3(min_val, 5, min_val); \
        TYPE##3 v_res = safe_unary_minus(v_a);             \
    }

#define TEST_UNARY_MINUS_UB_VEC4(TYPE)                        \
    {                                                         \
        TYPE min_val = std::numeric_limits<TYPE>::min();      \
        TYPE##4 v_a = make_##TYPE##4(min_val, 4, min_val, 2); \
        TYPE##4 v_res = safe_unary_minus(v_a);                \
    }

#define TEST_UNARY_MINUS_UB(TYPE)  \
    TEST_UNARY_MINUS_UB_VEC1(TYPE) \
    TEST_UNARY_MINUS_UB_VEC2(TYPE) \
    TEST_UNARY_MINUS_UB_VEC3(TYPE) \
    TEST_UNARY_MINUS_UB_VEC4(TYPE)

__device__ void hipsmith_kernel_logic(int argc, char **argv){
    TEST_UNARY_MINUS_UB(char)
        TEST_UNARY_MINUS_UB(short)
            TEST_UNARY_MINUS_UB(int)
                TEST_UNARY_MINUS_UB(long)}

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
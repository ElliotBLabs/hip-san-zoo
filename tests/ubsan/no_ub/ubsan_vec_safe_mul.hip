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
__host__ __device__ constexpr bool check_safe_mul(ProxyT a_in, ProxyT b_in)
{
    // in HIP the internal type of a vector is a proxy but we need properties of the c++ internal type
    BaseT a = static_cast<BaseT>(a_in);
    BaseT b = static_cast<BaseT>(b_in);

    if constexpr (std::is_signed_v<BaseT>)
    {
        // Both positive, overflow check
        if (a > 0 && b > 0 && a > std::numeric_limits<BaseT>::max() / b)
            return false;
        // a positive, b, underflow check
        if (a > 0 && b <= 0 && b < std::numeric_limits<BaseT>::min() / a)
            return false;
        // a negative, b positive, underflow check
        if (a <= 0 && b > 0 && a < std::numeric_limits<BaseT>::min() / b)
            return false;
        // Both negative, overflow check
        if (a < 0 && b < 0 && a < std::numeric_limits<BaseT>::max() / b)
            return false;

        return true;
    }
    else
    {
        // no UB to see here
        return true;
    }
}

// generate an overloaded safe_mul macro for every vector length and type combination supported
#define HIPSMITH_VEC1_SAFE_MUL(BASE_TYPE, VEC_TYPE)                      \
    __host__ __device__ inline VEC_TYPE safe_mul(VEC_TYPE a, VEC_TYPE b) \
    {                                                                    \
        if (check_safe_mul<BASE_TYPE>(a.x, b.x))                         \
            return a * b;                                                \
        return a;                                                        \
    }

#define HIPSMITH_VEC2_SAFE_MUL(BASE_TYPE, VEC_TYPE)                      \
    __host__ __device__ inline VEC_TYPE safe_mul(VEC_TYPE a, VEC_TYPE b) \
    {                                                                    \
        if (check_safe_mul<BASE_TYPE>(a.x, b.x) &&                       \
            check_safe_mul<BASE_TYPE>(a.y, b.y))                         \
            return a * b;                                                \
        return a;                                                        \
    }

#define HIPSMITH_VEC3_SAFE_MUL(BASE_TYPE, VEC_TYPE)                      \
    __host__ __device__ inline VEC_TYPE safe_mul(VEC_TYPE a, VEC_TYPE b) \
    {                                                                    \
        if (check_safe_mul<BASE_TYPE>(a.x, b.x) &&                       \
            check_safe_mul<BASE_TYPE>(a.y, b.y) &&                       \
            check_safe_mul<BASE_TYPE>(a.z, b.z))                         \
            return a * b;                                                \
        return a;                                                        \
    }

#define HIPSMITH_VEC4_SAFE_MUL(BASE_TYPE, VEC_TYPE)                      \
    __host__ __device__ inline VEC_TYPE safe_mul(VEC_TYPE a, VEC_TYPE b) \
    {                                                                    \
        if (check_safe_mul<BASE_TYPE>(a.x, b.x) &&                       \
            check_safe_mul<BASE_TYPE>(a.y, b.y) &&                       \
            check_safe_mul<BASE_TYPE>(a.z, b.z) &&                       \
            check_safe_mul<BASE_TYPE>(a.w, b.w))                         \
            return a * b;                                                \
        return a;                                                        \
    }

#define GENERATE_ALL_VEC_SAFE_MUL(BASE_TYPE)              \
    HIPSMITH_VEC1_SAFE_MUL(BASE_TYPE, BASE_TYPE##1)       \
    HIPSMITH_VEC2_SAFE_MUL(BASE_TYPE, BASE_TYPE##2)       \
    HIPSMITH_VEC3_SAFE_MUL(BASE_TYPE, BASE_TYPE##3)       \
    HIPSMITH_VEC4_SAFE_MUL(BASE_TYPE, BASE_TYPE##4)       \
    HIPSMITH_VEC1_SAFE_MUL(u##BASE_TYPE, u##BASE_TYPE##1) \
    HIPSMITH_VEC2_SAFE_MUL(u##BASE_TYPE, u##BASE_TYPE##2) \
    HIPSMITH_VEC3_SAFE_MUL(u##BASE_TYPE, u##BASE_TYPE##3) \
    HIPSMITH_VEC4_SAFE_MUL(u##BASE_TYPE, u##BASE_TYPE##4)

GENERATE_ALL_VEC_SAFE_MUL(char)
GENERATE_ALL_VEC_SAFE_MUL(short)
GENERATE_ALL_VEC_SAFE_MUL(int)
GENERATE_ALL_VEC_SAFE_MUL(long)

// Overflow Tests (Positive * Positive)
#define TEST_MUL_OVERFLOW_POS_VEC1(TYPE)                 \
    {                                                    \
        TYPE max_val = std::numeric_limits<TYPE>::max(); \
        TYPE##1 v_a = make_##TYPE##1(max_val / 2);       \
        TYPE##1 v_b = make_##TYPE##1(offset + 2);        \
        TYPE##1 v_res = safe_mul(v_a, v_b);              \
    }

#define TEST_MUL_OVERFLOW_POS_VEC2(TYPE)                 \
    {                                                    \
        TYPE max_val = std::numeric_limits<TYPE>::max(); \
        TYPE##2 v_a = make_##TYPE##2(max_val / 2, 1);    \
        TYPE##2 v_b = make_##TYPE##2(offset + 2, 4);     \
        TYPE##2 v_res = safe_mul(v_a, v_b);              \
    }

// Overflow Tests (Negative * Negative)
// offset =1 
#define TEST_MUL_OVERFLOW_NEG_VEC3(TYPE)                 \
    {                                                    \
        TYPE min_val = std::numeric_limits<TYPE>::min(); \
        TYPE##3 v_a = make_##TYPE##3(min_val, -5, -5);   \
        TYPE##3 v_b = make_##TYPE##3(-offset, -5, -1);   \
        TYPE##3 v_res = safe_mul(v_a, v_b);              \
    }

// Underflow Tests (Positive * Negative)
#define TEST_MUL_UNDERFLOW_VEC4(TYPE)                           \
    {                                                           \
        TYPE max_val = std::numeric_limits<TYPE>::max();        \
        TYPE##4 v_a = make_##TYPE##4(max_val, 4, 2, 2);         \
        TYPE##4 v_b = make_##TYPE##4(-(offset + 1), 4, 1, 3);   \
        TYPE##4 v_res = safe_mul(v_a, v_b);                     \
    }

#define TEST_MUL_UB(TYPE)             \
    TEST_MUL_OVERFLOW_POS_VEC1(TYPE)  \
    TEST_MUL_OVERFLOW_POS_VEC2(TYPE)  \
    TEST_MUL_OVERFLOW_NEG_VEC3(TYPE)  \
    TEST_MUL_UNDERFLOW_VEC4(TYPE)

__device__ void hipsmith_kernel_logic(int argc, char **argv)
{
    // offset = 1 as no args passed in
    int offset = argc;

    TEST_MUL_UB(char)
    TEST_MUL_UB(short)
    TEST_MUL_UB(int)
    TEST_MUL_UB(long)
}

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
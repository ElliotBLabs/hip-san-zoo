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
__host__ __device__ constexpr bool check_safe_sub(ProxyT a_in, ProxyT b_in)
{
    // in HIP the internal type of a vector is a proxy but we need properties of the c++ internal type
    BaseT a = static_cast<BaseT>(a_in);
    BaseT b = static_cast<BaseT>(b_in);

    // Csmith uses obfuscated bitwise operations (XOR, AND, shifts) to detect overflow we try a more standard approach
    if constexpr (std::is_signed_v<BaseT>)
    {
        // Overflow check (a > 0, b < 0, a - b > MAX)
        if ((b < 0) && (a > std::numeric_limits<BaseT>::max() + b))
            return false;
        // Underflow check (a < 0, b > 0, a - b < MIN)
        if ((b > 0) && (a < std::numeric_limits<BaseT>::min() + b))
            return false;
        return true;
    }
    else
    {
        // no other UB 
        return true;
    }
}

// generate an overloaded safe_sub macro for every vector length and type combination supported
#define HIPSMITH_VEC1_SAFE_SUB(BASE_TYPE, VEC_TYPE)                      \
    __host__ __device__ inline VEC_TYPE safe_sub(VEC_TYPE a, VEC_TYPE b) \
    {                                                                    \
        if (check_safe_sub<BASE_TYPE>(a.x, b.x))                         \
            return a - b;                                                \
        return a;                                                        \
    }

#define HIPSMITH_VEC2_SAFE_SUB(BASE_TYPE, VEC_TYPE)                      \
    __host__ __device__ inline VEC_TYPE safe_sub(VEC_TYPE a, VEC_TYPE b) \
    {                                                                    \
        if (check_safe_sub<BASE_TYPE>(a.x, b.x) &&                       \
            check_safe_sub<BASE_TYPE>(a.y, b.y))                         \
            return a - b;                                                \
        return a;                                                        \
    }

#define HIPSMITH_VEC3_SAFE_SUB(BASE_TYPE, VEC_TYPE)                      \
    __host__ __device__ inline VEC_TYPE safe_sub(VEC_TYPE a, VEC_TYPE b) \
    {                                                                    \
        if (check_safe_sub<BASE_TYPE>(a.x, b.x) &&                       \
            check_safe_sub<BASE_TYPE>(a.y, b.y) &&                       \
            check_safe_sub<BASE_TYPE>(a.z, b.z))                         \
            return a - b;                                                \
        return a;                                                        \
    }

#define HIPSMITH_VEC4_SAFE_SUB(BASE_TYPE, VEC_TYPE)                      \
    __host__ __device__ inline VEC_TYPE safe_sub(VEC_TYPE a, VEC_TYPE b) \
    {                                                                    \
        if (check_safe_sub<BASE_TYPE>(a.x, b.x) &&                       \
            check_safe_sub<BASE_TYPE>(a.y, b.y) &&                       \
            check_safe_sub<BASE_TYPE>(a.z, b.z) &&                       \
            check_safe_sub<BASE_TYPE>(a.w, b.w))                         \
            return a - b;                                                \
        return a;                                                        \
    }

#define GENERATE_ALL_VEC_SAFE_SUB(BASE_TYPE)              \
    HIPSMITH_VEC1_SAFE_SUB(BASE_TYPE, BASE_TYPE##1)       \
    HIPSMITH_VEC2_SAFE_SUB(BASE_TYPE, BASE_TYPE##2)       \
    HIPSMITH_VEC3_SAFE_SUB(BASE_TYPE, BASE_TYPE##3)       \
    HIPSMITH_VEC4_SAFE_SUB(BASE_TYPE, BASE_TYPE##4)       \
    HIPSMITH_VEC1_SAFE_SUB(u##BASE_TYPE, u##BASE_TYPE##1) \
    HIPSMITH_VEC2_SAFE_SUB(u##BASE_TYPE, u##BASE_TYPE##2) \
    HIPSMITH_VEC3_SAFE_SUB(u##BASE_TYPE, u##BASE_TYPE##3) \
    HIPSMITH_VEC4_SAFE_SUB(u##BASE_TYPE, u##BASE_TYPE##4)

GENERATE_ALL_VEC_SAFE_SUB(char)
GENERATE_ALL_VEC_SAFE_SUB(short)
GENERATE_ALL_VEC_SAFE_SUB(int)
GENERATE_ALL_VEC_SAFE_SUB(long)

// Overflow Tests (max - (-offset)) where offset = 1
#define TEST_SUB_OVERFLOW_VEC1(TYPE)                     \
    {                                                    \
        TYPE max_val = std::numeric_limits<TYPE>::max(); \
        TYPE##1 v_a = make_##TYPE##1(max_val);           \
        TYPE##1 v_b = make_##TYPE##1(-offset);           \
        TYPE##1 v_res = safe_sub(v_a, v_b);              \
    }

#define TEST_SUB_OVERFLOW_VEC2(TYPE)                     \
    {                                                    \
        TYPE max_val = std::numeric_limits<TYPE>::max(); \
        TYPE##2 v_a = make_##TYPE##2(max_val, 1);        \
        TYPE##2 v_b = make_##TYPE##2(-offset, 4);        \
        TYPE##2 v_res = safe_sub(v_a, v_b);              \
    }

#define TEST_SUB_OVERFLOW_VEC3(TYPE)                     \
    {                                                    \
        TYPE max_val = std::numeric_limits<TYPE>::max(); \
        TYPE##3 v_a = make_##TYPE##3(max_val, 5, 5);     \
        TYPE##3 v_b = make_##TYPE##3(-offset, 5, 1);     \
        TYPE##3 v_res = safe_sub(v_a, v_b);              \
    }

#define TEST_SUB_OVERFLOW_VEC4(TYPE)                     \
    {                                                    \
        TYPE max_val = std::numeric_limits<TYPE>::max(); \
        TYPE##4 v_a = make_##TYPE##4(max_val, 4, 2, 2);  \
        TYPE##4 v_b = make_##TYPE##4(-offset, 4, 1, 3);  \
        TYPE##4 v_res = safe_sub(v_a, v_b);              \
    }

// Underflow Tests (min - offset) where offset = 1
#define TEST_SUB_UNDERFLOW_VEC1(TYPE)                    \
    {                                                    \
        TYPE min_val = std::numeric_limits<TYPE>::min(); \
        TYPE##1 v_a = make_##TYPE##1(min_val);           \
        TYPE##1 v_b = make_##TYPE##1(offset);            \
        TYPE##1 v_res = safe_sub(v_a, v_b);              \
    }

#define TEST_SUB_UNDERFLOW_VEC2(TYPE)                    \
    {                                                    \
        TYPE min_val = std::numeric_limits<TYPE>::min(); \
        TYPE##2 v_a = make_##TYPE##2(1, min_val);        \
        TYPE##2 v_b = make_##TYPE##2(-4, offset);        \
        TYPE##2 v_res = safe_sub(v_a, v_b);              \
    }

#define TEST_SUB_UNDERFLOW_VEC3(TYPE)                      \
    {                                                      \
        TYPE min_val = std::numeric_limits<TYPE>::min();   \
        TYPE##3 v_a = make_##TYPE##3(min_val, 1, min_val); \
        TYPE##3 v_b = make_##TYPE##3(offset, 1, offset);   \
        TYPE##3 v_res = safe_sub(v_a, v_b);                \
    }

#define TEST_SUB_UNDERFLOW_VEC4(TYPE)                          \
    {                                                          \
        TYPE min_val = std::numeric_limits<TYPE>::min();       \
        TYPE##4 v_a = make_##TYPE##4(1, min_val, -2, min_val); \
        TYPE##4 v_b = make_##TYPE##4(2, offset, 2, offset);    \
        TYPE##4 v_res = safe_sub(v_a, v_b);                    \
    }

#define TEST_SUB_UB(TYPE)         \
    TEST_SUB_OVERFLOW_VEC1(TYPE)  \
    TEST_SUB_OVERFLOW_VEC2(TYPE)  \
    TEST_SUB_OVERFLOW_VEC3(TYPE)  \
    TEST_SUB_OVERFLOW_VEC4(TYPE)  \
    TEST_SUB_UNDERFLOW_VEC1(TYPE) \
    TEST_SUB_UNDERFLOW_VEC2(TYPE) \
    TEST_SUB_UNDERFLOW_VEC3(TYPE) \
    TEST_SUB_UNDERFLOW_VEC4(TYPE)

__device__ void hipsmith_kernel_logic(int argc, char **argv)
{
    // offset = 1 as no args passed in
    int offset = argc;

    TEST_SUB_UB(char)
    TEST_SUB_UB(short)
    TEST_SUB_UB(int)
    TEST_SUB_UB(long)
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
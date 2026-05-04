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
__host__ __device__ constexpr bool check_safe_mod(ProxyT a_in, ProxyT b_in)
{
    // in HIP the internal type of a vector is a proxy but we need properties of the c++ internal type
    BaseT a = static_cast<BaseT>(a_in);
    BaseT b = static_cast<BaseT>(b_in);

    // Modulo by zero is UB
    if (b == 0)
        return false;

    if constexpr (std::is_signed_v<BaseT>)
    {
        // Overflow check, INT_MIN % -1 is UB
        if ((a == std::numeric_limits<BaseT>::min()) && (b == -1))
            return false;
            
        return true;
    }
    else
    {
        // No UB here
        return true;
    }
}

// generate an overloaded safe_mod macro for every vector length and type combination supported
#define HIPSMITH_VEC1_SAFE_MOD(BASE_TYPE, VEC_TYPE)                      \
    __host__ __device__ inline VEC_TYPE safe_mod(VEC_TYPE a, VEC_TYPE b) \
    {                                                                    \
        if (check_safe_mod<BASE_TYPE>(a.x, b.x))                         \
            return a % b;                                                \
        return a;                                                        \
    }

#define HIPSMITH_VEC2_SAFE_MOD(BASE_TYPE, VEC_TYPE)                      \
    __host__ __device__ inline VEC_TYPE safe_mod(VEC_TYPE a, VEC_TYPE b) \
    {                                                                    \
        if (check_safe_mod<BASE_TYPE>(a.x, b.x) &&                       \
            check_safe_mod<BASE_TYPE>(a.y, b.y))                         \
            return a % b;                                                \
        return a;                                                        \
    }

#define HIPSMITH_VEC3_SAFE_MOD(BASE_TYPE, VEC_TYPE)                      \
    __host__ __device__ inline VEC_TYPE safe_mod(VEC_TYPE a, VEC_TYPE b) \
    {                                                                    \
        if (check_safe_mod<BASE_TYPE>(a.x, b.x) &&                       \
            check_safe_mod<BASE_TYPE>(a.y, b.y) &&                       \
            check_safe_mod<BASE_TYPE>(a.z, b.z))                         \
            return a % b;                                                \
        return a;                                                        \
    }

#define HIPSMITH_VEC4_SAFE_MOD(BASE_TYPE, VEC_TYPE)                      \
    __host__ __device__ inline VEC_TYPE safe_mod(VEC_TYPE a, VEC_TYPE b) \
    {                                                                    \
        if (check_safe_mod<BASE_TYPE>(a.x, b.x) &&                       \
            check_safe_mod<BASE_TYPE>(a.y, b.y) &&                       \
            check_safe_mod<BASE_TYPE>(a.z, b.z) &&                       \
            check_safe_mod<BASE_TYPE>(a.w, b.w))                         \
            return a % b;                                                \
        return a;                                                        \
    }

#define GENERATE_ALL_VEC_SAFE_MOD(BASE_TYPE)              \
    HIPSMITH_VEC1_SAFE_MOD(BASE_TYPE, BASE_TYPE##1)       \
    HIPSMITH_VEC2_SAFE_MOD(BASE_TYPE, BASE_TYPE##2)       \
    HIPSMITH_VEC3_SAFE_MOD(BASE_TYPE, BASE_TYPE##3)       \
    HIPSMITH_VEC4_SAFE_MOD(BASE_TYPE, BASE_TYPE##4)       \
    HIPSMITH_VEC1_SAFE_MOD(u##BASE_TYPE, u##BASE_TYPE##1) \
    HIPSMITH_VEC2_SAFE_MOD(u##BASE_TYPE, u##BASE_TYPE##2) \
    HIPSMITH_VEC3_SAFE_MOD(u##BASE_TYPE, u##BASE_TYPE##3) \
    HIPSMITH_VEC4_SAFE_MOD(u##BASE_TYPE, u##BASE_TYPE##4)

GENERATE_ALL_VEC_SAFE_MOD(char)
GENERATE_ALL_VEC_SAFE_MOD(short)
GENERATE_ALL_VEC_SAFE_MOD(int)
GENERATE_ALL_VEC_SAFE_MOD(long)

// Modulo by Zero Tests where offset = 1 (so offset - 1 == 0)
#define TEST_MOD_ZERO_VEC1(TYPE)                         \
    {                                                    \
        TYPE##1 v_a = make_##TYPE##1(10);                \
        TYPE##1 v_b = make_##TYPE##1(offset - 1);        \
        TYPE##1 v_res = safe_mod(v_a, v_b);              \
    }

#define TEST_MOD_ZERO_VEC2(TYPE)                         \
    {                                                    \
        TYPE##2 v_a = make_##TYPE##2(10, 20);            \
        TYPE##2 v_b = make_##TYPE##2(offset - 1, 4);     \
        TYPE##2 v_res = safe_mod(v_a, v_b);              \
    }

#define TEST_MOD_ZERO_VEC3(TYPE)                         \
    {                                                    \
        TYPE##3 v_a = make_##TYPE##3(10, 5, 5);          \
        TYPE##3 v_b = make_##TYPE##3(offset - 1, 5, 1);  \
        TYPE##3 v_res = safe_mod(v_a, v_b);              \
    }

#define TEST_MOD_ZERO_VEC4(TYPE)                         \
    {                                                    \
        TYPE##4 v_a = make_##TYPE##4(10, 4, 2, 2);       \
        TYPE##4 v_b = make_##TYPE##4(offset - 1, 4, 1, 3);\
        TYPE##4 v_res = safe_mod(v_a, v_b);              \
    }

// Overflow Tests (min % -offset) where offset = 1
#define TEST_MOD_OVERFLOW_VEC1(TYPE)                     \
    {                                                    \
        TYPE min_val = std::numeric_limits<TYPE>::min(); \
        TYPE##1 v_a = make_##TYPE##1(min_val);           \
        TYPE##1 v_b = make_##TYPE##1(-offset);           \
        TYPE##1 v_res = safe_mod(v_a, v_b);              \
    }

#define TEST_MOD_OVERFLOW_VEC2(TYPE)                     \
    {                                                    \
        TYPE min_val = std::numeric_limits<TYPE>::min(); \
        TYPE##2 v_a = make_##TYPE##2(1, min_val);        \
        TYPE##2 v_b = make_##TYPE##2(-4, -offset);       \
        TYPE##2 v_res = safe_mod(v_a, v_b);              \
    }

#define TEST_MOD_OVERFLOW_VEC3(TYPE)                       \
    {                                                      \
        TYPE min_val = std::numeric_limits<TYPE>::min();   \
        TYPE##3 v_a = make_##TYPE##3(min_val, 1, min_val); \
        TYPE##3 v_b = make_##TYPE##3(-offset, 1, -offset); \
        TYPE##3 v_res = safe_mod(v_a, v_b);                \
    }

#define TEST_MOD_OVERFLOW_VEC4(TYPE)                           \
    {                                                          \
        TYPE min_val = std::numeric_limits<TYPE>::min();       \
        TYPE##4 v_a = make_##TYPE##4(1, min_val, -2, min_val); \
        TYPE##4 v_b = make_##TYPE##4(2, -offset, 2, -offset);  \
        TYPE##4 v_res = safe_mod(v_a, v_b);                    \
    }

#define TEST_MOD_UB(TYPE)         \
    TEST_MOD_ZERO_VEC1(TYPE)      \
    TEST_MOD_ZERO_VEC2(TYPE)      \
    TEST_MOD_ZERO_VEC3(TYPE)      \
    TEST_MOD_ZERO_VEC4(TYPE)      \
    TEST_MOD_OVERFLOW_VEC1(TYPE)  \
    TEST_MOD_OVERFLOW_VEC2(TYPE)  \
    TEST_MOD_OVERFLOW_VEC3(TYPE)  \
    TEST_MOD_OVERFLOW_VEC4(TYPE)

__device__ void hipsmith_kernel_logic(int argc, char **argv)
{
    // offset = 1 as no args passed in
    int offset = argc;

    TEST_MOD_UB(char)
    TEST_MOD_UB(short)
    TEST_MOD_UB(int)
    TEST_MOD_UB(long)
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
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
__host__ __device__ constexpr bool check_safe_lshift(ProxyT a_in, ProxyT b_in)
{
    // in csmith it was also supported to do mixed int4 << uint4 for example but not in HIP
    // so only make these for matching types exactly
    BaseT a = static_cast<BaseT>(a_in);
    BaseT b = static_cast<BaseT>(b_in);
    constexpr size_t bit_width = sizeof(BaseT) * 8;

    if constexpr (std::is_signed_v<BaseT>)
    {
        // Cannot shift by a negative amount
        if (b < 0 || a < 0)
            return false;
    }

    // cannot shift by bit-width or more
    if (static_cast<size_t>(b) >= bit_width)
        return false;

    // prevent overflow
    if (a > (std::numeric_limits<BaseT>::max() >> b))
        return false;

    return true;
}

// Generate overloaded safe_lshift macros
#define HIPSMITH_VEC1_SAFE_LSHIFT(BASE_TYPE, VEC_TYPE)                      \
    __host__ __device__ inline VEC_TYPE safe_lshift(VEC_TYPE a, VEC_TYPE b) \
    {                                                                       \
        if (check_safe_lshift<BASE_TYPE>(a.x, b.x))                         \
            return a << b;                                                  \
        return a;                                                           \
    }

#define HIPSMITH_VEC2_SAFE_LSHIFT(BASE_TYPE, VEC_TYPE)                      \
    __host__ __device__ inline VEC_TYPE safe_lshift(VEC_TYPE a, VEC_TYPE b) \
    {                                                                       \
        if (check_safe_lshift<BASE_TYPE>(a.x, b.x) &&                       \
            check_safe_lshift<BASE_TYPE>(a.y, b.y))                         \
            return a << b;                                                  \
        return a;                                                           \
    }

#define HIPSMITH_VEC3_SAFE_LSHIFT(BASE_TYPE, VEC_TYPE)                      \
    __host__ __device__ inline VEC_TYPE safe_lshift(VEC_TYPE a, VEC_TYPE b) \
    {                                                                       \
        if (check_safe_lshift<BASE_TYPE>(a.x, b.x) &&                       \
            check_safe_lshift<BASE_TYPE>(a.y, b.y) &&                       \
            check_safe_lshift<BASE_TYPE>(a.z, b.z))                         \
            return a << b;                                                  \
        return a;                                                           \
    }

#define HIPSMITH_VEC4_SAFE_LSHIFT(BASE_TYPE, VEC_TYPE)                      \
    __host__ __device__ inline VEC_TYPE safe_lshift(VEC_TYPE a, VEC_TYPE b) \
    {                                                                       \
        if (check_safe_lshift<BASE_TYPE>(a.x, b.x) &&                       \
            check_safe_lshift<BASE_TYPE>(a.y, b.y) &&                       \
            check_safe_lshift<BASE_TYPE>(a.z, b.z) &&                       \
            check_safe_lshift<BASE_TYPE>(a.w, b.w))                         \
            return a << b;                                                  \
        return a;                                                           \
    }

#define GENERATE_ALL_VEC_SAFE_LSHIFT(BASE_TYPE)                \
    HIPSMITH_VEC1_SAFE_LSHIFT(BASE_TYPE, BASE_TYPE##1)         \
    HIPSMITH_VEC2_SAFE_LSHIFT(BASE_TYPE, BASE_TYPE##2)         \
    HIPSMITH_VEC3_SAFE_LSHIFT(BASE_TYPE, BASE_TYPE##3)         \
    HIPSMITH_VEC4_SAFE_LSHIFT(BASE_TYPE, BASE_TYPE##4)         \
    HIPSMITH_VEC1_SAFE_LSHIFT(u##BASE_TYPE, u##BASE_TYPE##1)   \
    HIPSMITH_VEC2_SAFE_LSHIFT(u##BASE_TYPE, u##BASE_TYPE##2)   \
    HIPSMITH_VEC3_SAFE_LSHIFT(u##BASE_TYPE, u##BASE_TYPE##3)   \
    HIPSMITH_VEC4_SAFE_LSHIFT(u##BASE_TYPE, u##BASE_TYPE##4)

GENERATE_ALL_VEC_SAFE_LSHIFT(char)
GENERATE_ALL_VEC_SAFE_LSHIFT(short)
GENERATE_ALL_VEC_SAFE_LSHIFT(int)
GENERATE_ALL_VEC_SAFE_LSHIFT(long)

// Negative shift amount
#define TEST_LSHIFT_NEG_AMT_VEC1(TYPE)                   \
    {                                                    \
        TYPE##1 v_a = make_##TYPE##1(10);                \
        TYPE##1 v_b = make_##TYPE##1(-offset);           \
        TYPE##1 v_res = safe_lshift(v_a, v_b);           \
    }

// Oversized shift amount ( >= bit width)
#define TEST_LSHIFT_OVERSIZE_VEC2(TYPE)                  \
    {                                                    \
        int bit_width = sizeof(TYPE) * 8;                \
        TYPE##2 v_a = make_##TYPE##2(10, 5);             \
        TYPE##2 v_b = make_##TYPE##2(bit_width, 1);      \
        TYPE##2 v_res = safe_lshift(v_a, v_b);           \
    }

// Left operand is negative
#define TEST_LSHIFT_NEG_VAL_VEC3(TYPE)                   \
    {                                                    \
        TYPE##3 v_a = make_##TYPE##3(-10, 5, 5);         \
        TYPE##3 v_b = make_##TYPE##3(offset, 1, 1);      \
        TYPE##3 v_res = safe_lshift(v_a, v_b);           \
    }

// Left shift overflow (bits fall off)
#define TEST_LSHIFT_OVERFLOW_VEC4(TYPE)                  \
    {                                                    \
        TYPE max_val = std::numeric_limits<TYPE>::max(); \
        TYPE##4 v_a = make_##TYPE##4(max_val, 4, 2, 2);  \
        TYPE##4 v_b = make_##TYPE##4(offset, 4, 1, 3);   \
        TYPE##4 v_res = safe_lshift(v_a, v_b);           \
    }

#define TEST_LSHIFT_UB(TYPE)            \
    TEST_LSHIFT_NEG_AMT_VEC1(TYPE)      \
    TEST_LSHIFT_OVERSIZE_VEC2(TYPE)     \
    TEST_LSHIFT_NEG_VAL_VEC3(TYPE)      \
    TEST_LSHIFT_OVERFLOW_VEC4(TYPE)

__device__ void hipsmith_kernel_logic(int argc, char **argv)
{
    // offset = 1 as no args passed in
    int offset = argc;

    TEST_LSHIFT_UB(char)
    TEST_LSHIFT_UB(short)
    TEST_LSHIFT_UB(int)
    TEST_LSHIFT_UB(long)
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
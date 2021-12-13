//
// Created by lwilkinson on 12/7/21.
//

#ifndef BENCHMARK_CODELET_KERNEL_UTILS_CUH
#define BENCHMARK_CODELET_KERNEL_UTILS_CUH

#include "codelet_kernels.cuh"
#include "common/utils/cuda_utils.h"

__device__ __forceinline__
float4 global_const_vector_load(const float *__restrict__ src) {
  return __ldg(reinterpret_cast<const float4 *>(src));
}

__device__ __forceinline__
int4 global_const_vector_load(const int *__restrict__ src) {
  return __ldg(reinterpret_cast<const int4 *>(src));
}

__device__ __forceinline__
float4 shared_const_vector_load(const float *__restrict__ src) {
  return *reinterpret_cast<const float4 *>(src);
}

__device__ __forceinline__
int4 shared_const_vector_load(const int *__restrict__ src) {
  return *reinterpret_cast<const int4 *>(src);
}

__device__ __forceinline__
void shared_vector_store(float *__restrict__ dst, float4 src) {
  *reinterpret_cast<float4 *>(dst) = src;
}

__device__ __forceinline__
void vec_atomic_add_coeff(Dense& A, int i, int j, float4 val) {
  atomicAdd(&A.values[i * A.cols + j + 1], val.y);
  atomicAdd(&A.values[i * A.cols + j + 0], val.x);
  atomicAdd(&A.values[i * A.cols + j + 2], val.z);
  atomicAdd(&A.values[i * A.cols + j + 3], val.w);
}

#define cache(A) \
    float* A##_values = A.values; \
    int A##_cols = A.cols;        \
    int A##_rows = A.rows;

#define coeff(A, i, j) \
    &A##_values[i * A##_cols + j]

#define coeff_ptr(A, i, j) \
    &A##_values[i * A##_cols + j]

#define FMAA(accumulate, a, b) \
  accumulate.x += (a) * b.x;   \
  accumulate.y += (a) * b.y;   \
  accumulate.z += (a) * b.z;   \
  accumulate.w += (a) * b.w;

#define global_const_vector_load_float4(dst, src) \
  *reinterpret_cast<float4 *>(dst) = __ldg(reinterpret_cast<float4 *>(src));

#define global_const_vector_load_int4(dst, src) \
  *reinterpret_cast<int4 *>(dst) = __ldg(reinterpret_cast<int4 *>(src));


#define _load_b_reg_branchless(base_idx, idx, reg) { \
    int col_idx = base_idx + idx;                                                           \
    int col = block_col_pattern[col_idx];                                                   \
    b##reg = global_const_vector_load(coeff_ptr(B, col, k + thd_x_vec_offset));             \
}

#endif //BENCHMARK_CODELET_KERNEL_UTILS_CUH

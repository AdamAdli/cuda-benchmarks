//
// Created by lwilkinson on 12/7/21.
//

#include "codelet_kernel_utils.cuh"

using namespace CodeletMultiply;
using namespace codelet_4x32x1;

/// This example streams elementsPerThreadBlock worth of data from global memory
/// into a limited sized shared memory (elementsInShared) block to operate on in
/// multiple (two) stages. As stage N is kicked off, we can wait on and operate on stage N-1.
#include <cuda.h>
#include <cuda/barrier>
#include <cooperative_groups.h>
#include <cooperative_groups/memcpy_async.h>

namespace cg = cooperative_groups;

static dim3 codelet_block_dim(BLOCK_X_DIM, BLOCK_Y_DIM);

#define _load_a_reg_branchless(offset, reg) {                                                                   \
    a##reg = global_const_vector_load(&row_segment_values_ptr[threadIdx.y * MAX_ROWS_PER_BLOCK + offset]);      \
}

#define _load_a_reg_branchless(offset, reg) {                                                                   \
    a##reg = global_const_vector_load(&row_segment_values_ptr[threadIdx.y * MAX_ROWS_PER_BLOCK + offset]);      \
}

__global__ static void
_block_multiply_reg_storage_bypass_8x32x1(const Block * blocks, int num_blocks, const CSR<float> A, const Dense B, Dense C) {
  int block_idx = blockIdx.x;
  cg::thread_block tb = cg::this_thread_block();

  const Block *block = &blocks[block_idx];

  //cache(A);
  cache(B);
  cache(C);

  // TODO: make CODELET_MULTIPLY_MAX_COLS_PER_BLOCK dynamic
  //__shared__ __align__(32) float A_s[BLOCK_ROWS * BLOCK_COLS];
  __shared__ __align__(32) int A_rows[MAX_ROWS_PER_BLOCK];
  __shared__ __align__(32) int block_col_pattern[MAX_COLS_PER_BLOCK];
  __shared__ __align__(32) float B_s[MAX_COLS_PER_BLOCK][TILE_K];
  __shared__ __align__(32) float C_s[MAX_ROWS_PER_BLOCK][TILE_K];

  int block_col_pattern_len = block->col_pattern_len;
  int block_num_rows = block->num_rows;
  int* block_col_pattern_ptr = block->col_pattern;
  int * block_rows_ptr = block->rows;
  float* row_segment_values_ptr = block->row_segment_values;

  int non_zeros = block_num_rows * block_col_pattern_len;
  int thd_idx_linear = threadIdx.x + threadIdx.y * blockDim.x;
  int thd_x_vec_offset = threadIdx.x * VECTOR_WIDTH;
  int thd_linear_vec_offset = thd_idx_linear * VECTOR_WIDTH;
  int block_size = blockDim.x * blockDim.y;

  int vector_load_block_width = block_size * VECTOR_WIDTH;
  int a_vector_loads_full_block = non_zeros / vector_load_block_width;
  int a_vector_loads_partial_block_start = a_vector_loads_full_block * vector_load_block_width;
  int a_vector_partial_load = non_zeros - a_vector_loads_partial_block_start;


  int k = 0;

  cg::memcpy_async(tb, block_col_pattern,
                   cuda::aligned_size_t<16>(block_col_pattern_len),
                   block_col_pattern_ptr,
                   cuda::aligned_size_t<16>(block_col_pattern_len));


  cg::memcpy_async(tb, A_rows,
                   cuda::aligned_size_t<16>(block_num_rows),
                   block_rows_ptr,
                   cuda::aligned_size_t<16>(block_num_rows));


  cg::wait_prior<1>(tb); // Wait for block_col_pattern

#pragma unroll
  for (int i = 0; i < MAX_COLS_PER_BLOCK; i++) {
    cg::memcpy_async(tb, B_s[i],
                     cuda::aligned_size_t<16>(TILE_K),
                     coeff_ptr(B, block_col_pattern_ptr[i], k),
                     cuda::aligned_size_t<16>(B_cols - k));
  }

  __syncthreads();

  float4 a0, a1, a2, a3, a4, a5, a6, a7;
  float4 a8, a9, a10, a11, a12, a13, a14, a15;

  _load_a_reg_branchless( 0, 0);
  _load_a_reg_branchless( 4, 1);
  _load_a_reg_branchless( 8, 2);
  _load_a_reg_branchless(12, 3);
  _load_a_reg_branchless(16, 4);
  _load_a_reg_branchless(20, 5);
  _load_a_reg_branchless(24, 6);
  _load_a_reg_branchless(28, 7);

  _load_a_reg_branchless(MAX_ROWS_PER_BLOCK * 4 +  0, 8);
  _load_a_reg_branchless(MAX_ROWS_PER_BLOCK * 4 +  4, 9);
  _load_a_reg_branchless(MAX_ROWS_PER_BLOCK * 4 +  8, 10);
  _load_a_reg_branchless(MAX_ROWS_PER_BLOCK * 4 + 12, 11);
  _load_a_reg_branchless(MAX_ROWS_PER_BLOCK * 4 + 16, 12);
  _load_a_reg_branchless(MAX_ROWS_PER_BLOCK * 4 + 20, 13);
  _load_a_reg_branchless(MAX_ROWS_PER_BLOCK * 4 + 24, 14);
  _load_a_reg_branchless(MAX_ROWS_PER_BLOCK * 4 + 28, 15);

  cg::wait_prior<1>(tb); // Wait for A_rows

  for (int k = 0; k < B_cols; k+= TILE_K) {
#pragma unroll
    for (int i = 0; i < MAX_COLS_PER_BLOCK; i++) {
      cg::memcpy_async(tb, B_s[i], cuda::aligned_size_t<16>(TILE_K),
                       coeff_ptr(B, block_col_pattern_ptr[i], k),
                       cuda::aligned_size_t<16>(B_cols - k));
    }

    cg::wait_prior<MAX_COLS_PER_BLOCK>(tb); //  Wait for previous B_s batch

    int offset = 0;

    float4 c0 = { .x = 0, .y = 0, .z = 0, .w = 0 };
    float4 c1 = { .x = 0, .y = 0, .z = 0, .w = 0 };

    float4 b0 = shared_const_vector_load(&B_s[offset + 0][threadIdx.x * VECTOR_WIDTH]);
    float4 b1 = shared_const_vector_load(&B_s[offset + 1][threadIdx.x * VECTOR_WIDTH]);
    float4 b2 = shared_const_vector_load(&B_s[offset + 2][threadIdx.x * VECTOR_WIDTH]);
    float4 b3 = shared_const_vector_load(&B_s[offset + 3][threadIdx.x * VECTOR_WIDTH]);
    float4 b4 = shared_const_vector_load(&B_s[offset + 4][threadIdx.x * VECTOR_WIDTH]);
    float4 b5 = shared_const_vector_load(&B_s[offset + 5][threadIdx.x * VECTOR_WIDTH]);
    float4 b6 = shared_const_vector_load(&B_s[offset + 6][threadIdx.x * VECTOR_WIDTH]);
    float4 b7 = shared_const_vector_load(&B_s[offset + 7][threadIdx.x * VECTOR_WIDTH]);

    FMAA(c0, a0.x, b0);
    FMAA(c0, a0.y, b1);
    FMAA(c0, a0.z, b2);
    FMAA(c0, a0.w, b3);
    FMAA(c0, a1.x, b4);
    FMAA(c0, a1.y, b5);
    FMAA(c0, a1.z, b6);
    FMAA(c0, a1.w, b7);

    FMAA(c1, a8.x, b0);
    FMAA(c1, a8.y, b1);
    FMAA(c1, a8.z, b2);
    FMAA(c1, a8.w, b3);
    FMAA(c1, a9.x, b4);
    FMAA(c1, a9.y, b5);
    FMAA(c1, a9.z, b6);
    FMAA(c1, a9.w, b7);

    offset = 8;

    b0 = shared_const_vector_load(&B_s[offset + 0][threadIdx.x * VECTOR_WIDTH]);
    b1 = shared_const_vector_load(&B_s[offset + 1][threadIdx.x * VECTOR_WIDTH]);
    b2 = shared_const_vector_load(&B_s[offset + 2][threadIdx.x * VECTOR_WIDTH]);
    b3 = shared_const_vector_load(&B_s[offset + 3][threadIdx.x * VECTOR_WIDTH]);
    b4 = shared_const_vector_load(&B_s[offset + 4][threadIdx.x * VECTOR_WIDTH]);
    b5 = shared_const_vector_load(&B_s[offset + 5][threadIdx.x * VECTOR_WIDTH]);
    b6 = shared_const_vector_load(&B_s[offset + 6][threadIdx.x * VECTOR_WIDTH]);
    b7 = shared_const_vector_load(&B_s[offset + 7][threadIdx.x * VECTOR_WIDTH]);

    FMAA(c0, a2.x, b0);
    FMAA(c0, a2.y, b1);
    FMAA(c0, a2.z, b2);
    FMAA(c0, a2.w, b3);
    FMAA(c0, a3.x, b4);
    FMAA(c0, a3.y, b5);
    FMAA(c0, a3.z, b6);
    FMAA(c0, a3.w, b7);

    FMAA(c1, a10.x, b0);
    FMAA(c1, a10.y, b1);
    FMAA(c1, a10.z, b2);
    FMAA(c1, a10.w, b3);
    FMAA(c1, a11.x, b4);
    FMAA(c1, a11.y, b5);
    FMAA(c1, a11.z, b6);
    FMAA(c1, a11.w, b7);

    offset = 16;

    b0 = shared_const_vector_load(&B_s[offset + 0][threadIdx.x * VECTOR_WIDTH]);
    b1 = shared_const_vector_load(&B_s[offset + 1][threadIdx.x * VECTOR_WIDTH]);
    b2 = shared_const_vector_load(&B_s[offset + 2][threadIdx.x * VECTOR_WIDTH]);
    b3 = shared_const_vector_load(&B_s[offset + 3][threadIdx.x * VECTOR_WIDTH]);
    b4 = shared_const_vector_load(&B_s[offset + 4][threadIdx.x * VECTOR_WIDTH]);
    b5 = shared_const_vector_load(&B_s[offset + 5][threadIdx.x * VECTOR_WIDTH]);
    b6 = shared_const_vector_load(&B_s[offset + 6][threadIdx.x * VECTOR_WIDTH]);
    b7 = shared_const_vector_load(&B_s[offset + 7][threadIdx.x * VECTOR_WIDTH]);

    FMAA(c0, a4.x, b0);
    FMAA(c0, a4.y, b1);
    FMAA(c0, a4.z, b2);
    FMAA(c0, a4.w, b3);
    FMAA(c0, a5.x, b4);
    FMAA(c0, a5.y, b5);
    FMAA(c0, a5.z, b6);
    FMAA(c0, a5.w, b7);

    FMAA(c1, a12.x, b0);
    FMAA(c1, a12.y, b1);
    FMAA(c1, a12.z, b2);
    FMAA(c1, a12.w, b3);
    FMAA(c1, a13.x, b4);
    FMAA(c1, a13.y, b5);
    FMAA(c1, a13.z, b6);
    FMAA(c1, a13.w, b7);

    offset = 24;

    b0 = shared_const_vector_load(&B_s[offset + 0][threadIdx.x * VECTOR_WIDTH]);
    b1 = shared_const_vector_load(&B_s[offset + 1][threadIdx.x * VECTOR_WIDTH]);
    b2 = shared_const_vector_load(&B_s[offset + 2][threadIdx.x * VECTOR_WIDTH]);
    b3 = shared_const_vector_load(&B_s[offset + 3][threadIdx.x * VECTOR_WIDTH]);
    b4 = shared_const_vector_load(&B_s[offset + 4][threadIdx.x * VECTOR_WIDTH]);
    b5 = shared_const_vector_load(&B_s[offset + 5][threadIdx.x * VECTOR_WIDTH]);
    b6 = shared_const_vector_load(&B_s[offset + 6][threadIdx.x * VECTOR_WIDTH]);
    b7 = shared_const_vector_load(&B_s[offset + 7][threadIdx.x * VECTOR_WIDTH]);

    FMAA(c0, a6.x, b0);
    FMAA(c0, a6.y, b1);
    FMAA(c0, a6.z, b2);
    FMAA(c0, a6.w, b3);
    FMAA(c0, a7.x, b4);
    FMAA(c0, a7.y, b5);
    FMAA(c0, a7.z, b6);
    FMAA(c0, a7.w, b7);

    FMAA(c1, a14.x, b0);
    FMAA(c1, a14.y, b1);
    FMAA(c1, a14.z, b2);
    FMAA(c1, a14.w, b3);
    FMAA(c1, a15.x, b4);
    FMAA(c1, a15.y, b5);
    FMAA(c1, a15.z, b6);
    FMAA(c1, a15.w, b7);

    shared_vector_store(&C_s[threadIdx.y][thd_x_vec_offset], c0);
    shared_vector_store(&C_s[4 + threadIdx.y][thd_x_vec_offset], c1);

    for (int row_idx = 0; row_idx < MAX_ROWS_PER_BLOCK; row_idx++) {
      float *C_addr_g = coeff_ptr(C, A_rows[row_idx], k + thd_idx_linear);
      float C_to_add = C_s[row_idx][thd_idx_linear];
      atomicAdd(C_addr_g, C_to_add);
    }
  }
}


#define _load_b_bypass(base_idx, idx)                                                \
    cg::memcpy_async(tb, B_async[idx],                                               \
            cuda::aligned_size_t<16>(TILE_K),                                        \
            coeff_ptr(B, block_col_pattern[col_pattern_idx + base_idx + idx], k),    \
            cuda::aligned_size_t<16>(TILE_K));

#define _load_b_reg_from_bypass(reg, idx)                                           \
    b##reg = shared_const_vector_load(&B_async[idx][thd_x_vec_offset]);             \


__global__ void _block_multiply_reg_storage_orig_4x32x1(const Block * blocks,
                                                        int num_blocks,
                                                        const CSR<float> A,
                                                        const Dense B, Dense C) {
  int block_idx = blockIdx.x;
  cg::thread_block tb = cg::this_thread_block();

  const Block *block = &blocks[block_idx];

  //cache(A);
  cache(B);
  cache(C);

  // TODO: make CODELET_MULTIPLY_MAX_COLS_PER_BLOCK dynamic
  __shared__ __align__(32) float A_s[MAX_ROWS_PER_BLOCK * MAX_COLS_PER_BLOCK];
  __shared__ __align__(32) int A_rows[MAX_ROWS_PER_BLOCK];
  __shared__ __align__(32) int block_col_pattern[MAX_COLS_PER_BLOCK];
  //__shared__ __align__(32) float B_s[BLOCK_COLS][TILE_K];

  __shared__ __align__(32) float B_async[MAX_COLS_PER_BLOCK / 2][TILE_K];
  __shared__ __align__(32) float C_s[MAX_ROWS_PER_BLOCK][TILE_K];

  int block_col_pattern_len = block->col_pattern_len;
  int block_num_rows = block->num_rows;
  int* block_col_pattern_ptr = block->col_pattern;
  int * block_rows_ptr = block->rows;
  float* row_segment_values_ptr = block->row_segment_values;

  int non_zeros = block_num_rows * block_col_pattern_len;
  int thd_idx_linear = threadIdx.x + threadIdx.y * blockDim.x;
  int thd_x_vec_offset = threadIdx.x * VECTOR_WIDTH;
  int thd_linear_vec_offset = thd_idx_linear * VECTOR_WIDTH;
  int block_size = blockDim.x * blockDim.y;

  int vector_load_block_width = block_size * VECTOR_WIDTH;
  int a_vector_loads_full_block = non_zeros / vector_load_block_width;
  int a_vector_loads_partial_block_start = a_vector_loads_full_block * vector_load_block_width;
  int a_vector_partial_load = non_zeros - a_vector_loads_partial_block_start;

  cg::memcpy_async(tb, block_col_pattern,
                   cuda::aligned_size_t<16>(block_col_pattern_len),
                   block_col_pattern_ptr,
                   cuda::aligned_size_t<16>(block_col_pattern_len));

  cg::memcpy_async(tb, A_s,
                   cuda::aligned_size_t<16>(non_zeros),
                   row_segment_values_ptr,
                   cuda::aligned_size_t<16>(non_zeros));

  cg::memcpy_async(tb, A_rows,
                   cuda::aligned_size_t<16>(block_num_rows),
                   block_rows_ptr,
                   cuda::aligned_size_t<16>(block_num_rows));

  cg::wait_prior<1>(tb); // Wait for block_col_pattern

  float4 a0;
  float4 a1;
  float4 a2;
  float4 a3;
  float4 a4;
  float4 a5;
  float4 a6;
  float4 a7;

  int row_idx0 = (0 * blockDim.y + threadIdx.y);
  int row_offset0 = row_idx0 * block_col_pattern_len;

  for (int k = 0; k < B_cols; k+= TILE_K) {

    static_assert(MAX_ROWS_PER_BLOCK / BLOCK_Y_DIM  == 1);
    float4 c0 = { .x = 0, .y = 0, .z = 0, .w = 0 };

    float4 b0, b1, b2, b3, b4, b5, b6, b7;
    float4 b8, b9, b10, b11, b12, b13, b14, b15;

    float4 b16, b17, b18, b19, b20, b21, b22, b23;
    float4 b24, b25, b26, b27, b28, b29, b30, b31;

    _load_b_reg_branchless(0, 0, 0);
    _load_b_reg_branchless(0, 1, 1);
    _load_b_reg_branchless(0, 2, 2);
    _load_b_reg_branchless(0, 3, 3);
    _load_b_reg_branchless(0, 4, 4);
    _load_b_reg_branchless(0, 5, 5);
    _load_b_reg_branchless(0, 6, 6);
    _load_b_reg_branchless(0, 7, 7);

    _load_b_reg_branchless(0, 8, 8);
    _load_b_reg_branchless(0, 9, 9);
    _load_b_reg_branchless(0, 10, 10);
    _load_b_reg_branchless(0, 11, 11);
    _load_b_reg_branchless(0, 12, 12);
    _load_b_reg_branchless(0, 13, 13);
    _load_b_reg_branchless(0, 14, 14);
    _load_b_reg_branchless(0, 15, 15);

    if (k == 0) {
      cg::wait_prior<2>(tb); // Wait for A_s, A_rows

      a0 = shared_const_vector_load(&A_s[row_offset0 + 0]);
      a1 = shared_const_vector_load(&A_s[row_offset0 + 4]);
      a2 = shared_const_vector_load(&A_s[row_offset0 + 8]);
      a3 = shared_const_vector_load(&A_s[row_offset0 + 12]);
      a4 = shared_const_vector_load(&A_s[row_offset0 + 16]);
      a5 = shared_const_vector_load(&A_s[row_offset0 + 20]);
      a6 = shared_const_vector_load(&A_s[row_offset0 + 24]);
      a7 = shared_const_vector_load(&A_s[row_offset0 + 28]);
    }

    _load_b_reg_branchless(0, 16, 16);
    _load_b_reg_branchless(0, 17, 17);
    _load_b_reg_branchless(0, 18, 18);
    _load_b_reg_branchless(0, 19, 19);
    _load_b_reg_branchless(0, 20, 20);
    _load_b_reg_branchless(0, 21, 21);
    _load_b_reg_branchless(0, 22, 22);
    _load_b_reg_branchless(0, 23, 23);

    _load_b_reg_branchless(0, 24, 24);
    _load_b_reg_branchless(0, 25, 25);
    _load_b_reg_branchless(0, 26, 26);
    _load_b_reg_branchless(0, 27, 27);
    _load_b_reg_branchless(0, 28, 28);
    _load_b_reg_branchless(0, 29, 29);
    _load_b_reg_branchless(0, 30, 30);
    _load_b_reg_branchless(0, 31, 31);

    FMAA(c0, a0.x, b0);
    FMAA(c0, a0.y, b1);
    FMAA(c0, a0.z, b2);
    FMAA(c0, a0.w, b3);
    FMAA(c0, a1.x, b4);
    FMAA(c0, a1.y, b5);
    FMAA(c0, a1.z, b6);
    FMAA(c0, a1.w, b7);

    FMAA(c0, a2.x, b8);
    FMAA(c0, a2.y, b9);
    FMAA(c0, a2.z, b10);
    FMAA(c0, a2.w, b11);
    FMAA(c0, a3.x, b12);
    FMAA(c0, a3.y, b13);
    FMAA(c0, a3.z, b14);
    FMAA(c0, a3.w, b15);

    FMAA(c0, a4.x, b16);
    FMAA(c0, a4.y, b17);
    FMAA(c0, a4.z, b18);
    FMAA(c0, a4.w, b19);
    FMAA(c0, a5.x, b20);
    FMAA(c0, a5.y, b21);
    FMAA(c0, a5.z, b22);
    FMAA(c0, a5.w, b23);

    FMAA(c0, a6.x, b24);
    FMAA(c0, a6.y, b25);
    FMAA(c0, a6.z, b26);
    FMAA(c0, a6.w, b27);
    FMAA(c0, a7.x, b28);
    FMAA(c0, a7.y, b29);
    FMAA(c0, a7.z, b30);
    FMAA(c0, a7.w, b31);

    shared_vector_store(&C_s[row_idx0][thd_x_vec_offset], c0);

    __syncthreads();

    for (int row_idx = 0; row_idx < MAX_ROWS_PER_BLOCK; row_idx++) {
      float value = C_s[row_idx][thd_idx_linear];
      float* C_add_addr = coeff_ptr(C, A_rows[row_idx], k + thd_idx_linear);
      atomicAdd(C_add_addr, value);
    }
  }
}

int
codelet_4x32x1::codelet_multiply(cudaStream_t &stream, cudaEvent_t &start, cudaEvent_t &stop, const Block *blocks,
                                size_t num_blocks,
                                const CSR<float> &A_h, const CSR<float> &A, const Dense &B, Dense &C) {
  dim3 grid_dim(num_blocks, 1);

  cudaFuncSetCacheConfig(_block_multiply_reg_storage_orig_4x32x1, cudaFuncCachePreferShared);
  cudaEventRecord(start, stream);
  _block_multiply_reg_storage_orig_4x32x1<<<grid_dim, codelet_block_dim, 0, stream>>>(blocks, num_blocks, A, B, C);
  CHECK_CUDA(cudaGetLastError());
  cudaEventRecord(stop, stream);

  return 0;
}
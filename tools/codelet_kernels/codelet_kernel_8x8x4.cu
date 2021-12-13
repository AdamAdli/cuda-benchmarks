//
// Created by lwilkinson on 12/7/21.
//

#include "codelet_kernel_utils.cuh"

using namespace CodeletMultiply;
using namespace codelet_8x8x1;

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

#define _load_b_bypass(base_idx, idx)                                                \
    cg::memcpy_async(tb, B_async[idx],                                               \
            cuda::aligned_size_t<16>(TILE_K),                                        \
            coeff_ptr(B, block_col_pattern[col_pattern_idx + base_idx + idx], k),    \
            cuda::aligned_size_t<16>(TILE_K));

#define _load_b_reg_from_bypass(reg, idx)                                           \
    b##reg = shared_const_vector_load(&B_async[idx][thd_x_vec_offset]);             \


__global__ void _block_multiply_reg_storage_orig_8x8x4(const Block * blocks,
                                                        int num_blocks,
                                                        const CSR<float> A,
                                                        const Dense B, Dense C) {
  int block_idx = blockIdx.x;
  cg::thread_block tb = cg::this_thread_block();

  const Block *block = &blocks[block_idx];

  //cache(A);
  cache(B);
  cache(C);

  static constexpr int BLOCK_NON_ZEROS = BLOCK_COLS * BLOCK_ROWS;
  static constexpr int BATCH_TOTAL_NON_ZEROS = BLOCK_NON_ZEROS * BATCH_SIZE;
  static constexpr int BATCH_TOTAL_COLS = BLOCK_COLS * BATCH_SIZE;
  static constexpr int BATCH_TOTAL_ROWS = BLOCK_COLS * BATCH_SIZE;

  // TODO: make CODELET_MULTIPLY_MAX_COLS_PER_BLOCK dynamic
  __shared__ __align__(32) float A_s[BATCH_TOTAL_NON_ZEROS];
  __shared__ __align__(32) int A_rows[BATCH_TOTAL_ROWS];
  __shared__ __align__(32) int block_col_pattern[BATCH_TOTAL_COLS];
  __shared__ __align__(32) float C_s[BATCH_TOTAL_ROWS][TILE_K];

  int* block_col_pattern_ptr = block->col_pattern;
  int * block_rows_ptr = block->rows;
  float* row_segment_values_ptr = block->row_segment_values;

  int thd_idx_linear = threadIdx.x + threadIdx.y * blockDim.x;
  int thd_x_vec_offset = threadIdx.x * VECTOR_WIDTH;
  int cuda_block_size = blockDim.x * blockDim.y;

  cg::memcpy_async(tb, block_col_pattern,
                   cuda::aligned_size_t<16>(BATCH_TOTAL_COLS),
                   block_col_pattern_ptr,
                   cuda::aligned_size_t<16>(BATCH_TOTAL_COLS));

  cg::memcpy_async(tb, A_s,
                   cuda::aligned_size_t<16>(BATCH_TOTAL_NON_ZEROS),
                   row_segment_values_ptr,
                   cuda::aligned_size_t<16>(BATCH_TOTAL_NON_ZEROS));

  cg::memcpy_async(tb, A_rows,
                   cuda::aligned_size_t<16>(BATCH_TOTAL_ROWS),
                   block_rows_ptr,
                   cuda::aligned_size_t<16>(BATCH_TOTAL_ROWS));

  cg::wait_prior<1>(tb); // Wait for block_col_pattern

  for (int k = 0; k < B_cols; k+= TILE_K) {

    static_assert(BLOCK_ROWS / BLOCK_Y_DIM == 2);

#define _define_b_regs(_batch) \
    float4 b##_batch##0, b##_batch##1, b##_batch##2, b##_batch##3; \
    float4 b##_batch##4, b##_batch##5, b##_batch##6, b##_batch##7;

#define _define_c_regs(_batch) \
    float4 c##_batch##0 = {.x = 0, .y = 0, .z = 0, .w = 0}; \
    float4 c##_batch##1 = {.x = 0, .y = 0, .z = 0, .w = 0};

#define _load_b_regs_batch(_batch) { \
    int4 cols0 = shared_const_vector_load(&block_col_pattern[(_batch * BLOCK_COLS) + 0]);   \
    int4 cols1 = shared_const_vector_load(&block_col_pattern[(_batch * BLOCK_COLS) + 4]);   \
    b##_batch##0 = global_const_vector_load(coeff_ptr(B, cols0.x, k + thd_x_vec_offset));   \
    b##_batch##1 = global_const_vector_load(coeff_ptr(B, cols0.y, k + thd_x_vec_offset));   \
    b##_batch##2 = global_const_vector_load(coeff_ptr(B, cols0.z, k + thd_x_vec_offset));   \
    b##_batch##3 = global_const_vector_load(coeff_ptr(B, cols0.w, k + thd_x_vec_offset));   \
    b##_batch##4 = global_const_vector_load(coeff_ptr(B, cols1.x, k + thd_x_vec_offset));   \
    b##_batch##5 = global_const_vector_load(coeff_ptr(B, cols1.y, k + thd_x_vec_offset));   \
    b##_batch##6 = global_const_vector_load(coeff_ptr(B, cols1.z, k + thd_x_vec_offset));   \
    b##_batch##7 = global_const_vector_load(coeff_ptr(B, cols1.w, k + thd_x_vec_offset));   \
    }

#define _load_a_regs_batch(_batch) \
    int batch##_batch##_row_offset = _batch * BLOCK_NON_ZEROS; \
    int row_idx##_batch##0 = (0 * blockDim.y + threadIdx.y); \
    int row_idx##_batch##1 = (1 * blockDim.y + threadIdx.y);   \
    int row_offset##_batch##0 = (row_idx##_batch##0 * BLOCK_COLS) + batch##_batch##_row_offset; \
    int row_offset##_batch##1 = (row_idx##_batch##1 * BLOCK_COLS) + batch##_batch##_row_offset; \
    float4 a##_batch##00 = shared_const_vector_load(&A_s[row_offset##_batch##0 + 0]); \
    float4 a##_batch##01 = shared_const_vector_load(&A_s[row_offset##_batch##0 + 4]); \
    float4 a##_batch##10 = shared_const_vector_load(&A_s[row_offset##_batch##0 + 0]); \
    float4 a##_batch##11 = shared_const_vector_load(&A_s[row_offset##_batch##0 + 4]);

#define _batch_fma(_batch) \
    int c_idx##_batch##0 = (0 * blockDim.y + threadIdx.y) + (_batch * BLOCK_ROWS); \
    int c_idx##_batch##1 = (1 * blockDim.y + threadIdx.y) + (_batch * BLOCK_ROWS); \
    FMAA(c##_batch##0, a##_batch##00 .x, b##_batch##0) \
    FMAA(c##_batch##0, a##_batch##00 .y, b##_batch##1) \
    FMAA(c##_batch##0, a##_batch##00 .z, b##_batch##2) \
    FMAA(c##_batch##0, a##_batch##00 .w, b##_batch##3) \
    FMAA(c##_batch##0, a##_batch##01 .x, b##_batch##4) \
    FMAA(c##_batch##0, a##_batch##01 .y, b##_batch##5) \
    FMAA(c##_batch##0, a##_batch##01 .z, b##_batch##6) \
    FMAA(c##_batch##0, a##_batch##01 .w, b##_batch##7) \
    shared_vector_store(&C_s[c_idx##_batch##0][thd_x_vec_offset], c##_batch##0); \
    FMAA(c##_batch##1, a##_batch##10 .x, b##_batch##0) \
    FMAA(c##_batch##1, a##_batch##10 .y, b##_batch##1) \
    FMAA(c##_batch##1, a##_batch##10 .z, b##_batch##2) \
    FMAA(c##_batch##1, a##_batch##10 .w, b##_batch##3) \
    FMAA(c##_batch##1, a##_batch##11 .x, b##_batch##4) \
    FMAA(c##_batch##1, a##_batch##11 .y, b##_batch##5) \
    FMAA(c##_batch##1, a##_batch##11 .z, b##_batch##6) \
    FMAA(c##_batch##1, a##_batch##11 .w, b##_batch##7) \
    shared_vector_store(&C_s[c_idx##_batch##1][thd_x_vec_offset], c##_batch##1);

    _define_b_regs(0);
    _define_b_regs(1);
//    _define_b_regs(4);
//    _define_b_regs(5);
//    _define_b_regs(6);
//    _define_b_regs(7);

    _define_c_regs(0)
    _define_c_regs(1)
    _define_c_regs(2)
    _define_c_regs(3)
//    _define_c_regs(4)
//    _define_c_regs(5)
//    _define_c_regs(6)
//    _define_c_regs(7)

    _define_b_regs(2);
    _define_b_regs(3);

    _load_b_regs_batch(0);
    _load_b_regs_batch(1);
    _load_b_regs_batch(2);
    _load_b_regs_batch(3);

    if (k == 0) {
      cg::wait_prior<2>(tb); // Wait for A_s, A_rows
    }

    _load_a_regs_batch(0);
    _load_a_regs_batch(1);
    _load_a_regs_batch(2);
    _load_a_regs_batch(3);

//    _load_b_regs_batch(4);
//    _load_b_regs_batch(5);
//    _load_b_regs_batch(6);
//    _load_b_regs_batch(7);

//    _load_a_regs_batch(4);
//    _load_a_regs_batch(5);
//    _load_a_regs_batch(6);
//    _load_a_regs_batch(7);

    _batch_fma(0);
    _batch_fma(1);
    _batch_fma(2);
    _batch_fma(3);
//    _batch_fma(4);
//    _batch_fma(5);
//    _batch_fma(6);
//    _batch_fma(7);

    __syncthreads();

    for (int row_idx = 0; row_idx < BATCH_TOTAL_ROWS; row_idx++) {
      float value = C_s[row_idx][thd_idx_linear];
      float* C_add_addr = coeff_ptr(C, A_rows[row_idx], k + thd_idx_linear);
      if (value) atomicAdd(C_add_addr, value);
    }
  }
}

int
codelet_8x8x1::codelet_multiply(cudaStream_t &stream, cudaEvent_t &start, cudaEvent_t &stop, const Block *blocks,
                                size_t num_blocks,
                                const CSR<float> &A_h, const CSR<float> &A, const Dense &B, Dense &C) {
  dim3 grid_dim(num_blocks, 1);

  cudaFuncSetCacheConfig(_block_multiply_reg_storage_orig_8x8x4, cudaFuncCachePreferShared);
  cudaEventRecord(start, stream);
  _block_multiply_reg_storage_orig_8x8x4<<<grid_dim, codelet_block_dim, 0, stream>>>(blocks, num_blocks, A, B, C);
  CHECK_CUDA(cudaGetLastError());
  cudaEventRecord(stop, stream);

  return 0;
}
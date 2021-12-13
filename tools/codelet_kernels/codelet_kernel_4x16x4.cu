//
// Created by lwilkinson on 12/7/21.
//

#include "codelet_kernel_utils.cuh"

using namespace CodeletMultiply;
using namespace codelet_4x16x4;

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

__global__ void _block_multiply_bypass_4x16x4(const Block * blocks,
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
  static constexpr int BATCH_TOTAL_ROWS = BLOCK_ROWS * BATCH_SIZE;

  // TODO: make CODELET_MULTIPLY_MAX_COLS_PER_BLOCK dynamic
  __shared__ __align__(32) float A_s[BATCH_TOTAL_NON_ZEROS];
  __shared__ __align__(32) float B_s[BATCH_TOTAL_COLS][TILE_K];
  __shared__ __align__(32) float C_s[BATCH_TOTAL_ROWS][TILE_K];
  __shared__ __align__(32) int A_rows[BATCH_TOTAL_ROWS];
  __shared__ __align__(32) int block_col_pattern[BATCH_TOTAL_COLS];

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

  cg::memcpy_async(tb, A_rows,
                   cuda::aligned_size_t<16>(BATCH_TOTAL_ROWS),
                   block_rows_ptr,
                   cuda::aligned_size_t<16>(BATCH_TOTAL_ROWS));


#define _define_a_regs(_batch) \
    float4 a##_batch##0,  a##_batch##1,  a##_batch##2,  a##_batch##3;

#define _load_a_regs_batch(_batch) { \
    int batch##_batch##_row_offset = _batch * BLOCK_NON_ZEROS; \
    int row_idx##_batch##0 = (0 * blockDim.y + threadIdx.y);   \
    int row_offset##_batch##0 = (row_idx##_batch##0 * BLOCK_COLS) + batch##_batch##_row_offset;   \
    a##_batch##0 = global_const_vector_load(&row_segment_values_ptr[row_offset##_batch##0 + 0]);  \
    a##_batch##1 = global_const_vector_load(&row_segment_values_ptr[row_offset##_batch##0 + 4]);  \
    a##_batch##2 = global_const_vector_load(&row_segment_values_ptr[row_offset##_batch##0 + 8]);  \
    a##_batch##3 = global_const_vector_load(&row_segment_values_ptr[row_offset##_batch##0 + 12]); \
    }

  _define_a_regs(0)
  _define_a_regs(1)
  _define_a_regs(2)
  _define_a_regs(3)

  _load_a_regs_batch(0)
  _load_a_regs_batch(1)


  cg::wait_prior<1>(tb); // Wait for block_col_pattern

  int k_load = 0;
  int col_idx = 0;

#define _b_async_load_col(_batch, _col) {                                    \
  int col_idx = _batch * BLOCK_COLS + _col;                                  \
  cg::memcpy_async(tb, B_s[col_idx],                                         \
                   cuda::aligned_size_t<16>(TILE_K),                         \
                   &B_values[block_col_pattern[col_idx] * B_cols + k_load],  \
                   cuda::aligned_size_t<16>(B_cols - k_load));               \
  }

#define _b_async_load(_batch) \
  _b_async_load_col(_batch, 0)         \
  _b_async_load_col(_batch, 1)         \
  _b_async_load_col(_batch, 2)         \
  _b_async_load_col(_batch, 3)         \
  _b_async_load_col(_batch, 4)         \
  _b_async_load_col(_batch, 6)         \
  _b_async_load_col(_batch, 7)         \
  _b_async_load_col(_batch, 8)         \
  _b_async_load_col(_batch, 9)         \
  _b_async_load_col(_batch, 10)        \
  _b_async_load_col(_batch, 11)        \
  _b_async_load_col(_batch, 12)        \
  _b_async_load_col(_batch, 13)        \
  _b_async_load_col(_batch, 14)        \
  _b_async_load_col(_batch, 15)

  _b_async_load(0)
  _b_async_load(1)
  _b_async_load(2)
  _b_async_load(3)

  _load_a_regs_batch(2)
  _load_a_regs_batch(3)

  cg::wait_prior<1>(tb); // Wait for A_rows

  for (int k = 0; k < B_cols; k+= TILE_K) {

    static_assert(BLOCK_ROWS / BLOCK_Y_DIM == 1);

#define _define_b_regs(_batch) \
    float4 b##_batch##0,  b##_batch##1,  b##_batch##2,  b##_batch##3;  \
    float4 b##_batch##4,  b##_batch##5,  b##_batch##6,  b##_batch##7;  \
    float4 b##_batch##8,  b##_batch##9,  b##_batch##10, b##_batch##11; \
    float4 b##_batch##12, b##_batch##13, b##_batch##14, b##_batch##15;

#define _define_c_regs(_batch) \
    float4 c##_batch##0 = {.x = 0, .y = 0, .z = 0, .w = 0};

#define _load_b_regs_batch(_batch) { \
    b##_batch##0  = shared_const_vector_load(&B_s[_batch * BLOCK_COLS +  0][thd_x_vec_offset]);  \
    b##_batch##1  = shared_const_vector_load(&B_s[_batch * BLOCK_COLS +  1][thd_x_vec_offset]);  \
    b##_batch##2  = shared_const_vector_load(&B_s[_batch * BLOCK_COLS +  2][thd_x_vec_offset]);  \
    b##_batch##3  = shared_const_vector_load(&B_s[_batch * BLOCK_COLS +  3][thd_x_vec_offset]);  \
    b##_batch##4  = shared_const_vector_load(&B_s[_batch * BLOCK_COLS +  4][thd_x_vec_offset]);  \
    b##_batch##5  = shared_const_vector_load(&B_s[_batch * BLOCK_COLS +  5][thd_x_vec_offset]);  \
    b##_batch##6  = shared_const_vector_load(&B_s[_batch * BLOCK_COLS +  6][thd_x_vec_offset]);  \
    b##_batch##7  = shared_const_vector_load(&B_s[_batch * BLOCK_COLS +  7][thd_x_vec_offset]);  \
    b##_batch##8  = shared_const_vector_load(&B_s[_batch * BLOCK_COLS +  8][thd_x_vec_offset]);  \
    b##_batch##9  = shared_const_vector_load(&B_s[_batch * BLOCK_COLS +  9][thd_x_vec_offset]);  \
    b##_batch##10 = shared_const_vector_load(&B_s[_batch * BLOCK_COLS + 10][thd_x_vec_offset]);  \
    b##_batch##11 = shared_const_vector_load(&B_s[_batch * BLOCK_COLS + 11][thd_x_vec_offset]);  \
    b##_batch##12 = shared_const_vector_load(&B_s[_batch * BLOCK_COLS + 12][thd_x_vec_offset]);  \
    b##_batch##13 = shared_const_vector_load(&B_s[_batch * BLOCK_COLS + 13][thd_x_vec_offset]);  \
    b##_batch##14 = shared_const_vector_load(&B_s[_batch * BLOCK_COLS + 14][thd_x_vec_offset]);  \
    b##_batch##15 = shared_const_vector_load(&B_s[_batch * BLOCK_COLS + 15][thd_x_vec_offset]);  \
    }

#define _batch_fma(_batch) \
    int c_idx##_batch##0 = (0 * blockDim.y + threadIdx.y) + (_batch * BLOCK_ROWS); \
    FMAA(c##_batch##0, a##_batch##0 .x, b##_batch##0)  \
    FMAA(c##_batch##0, a##_batch##0 .y, b##_batch##1)  \
    FMAA(c##_batch##0, a##_batch##0 .z, b##_batch##2)  \
    FMAA(c##_batch##0, a##_batch##0 .w, b##_batch##3)  \
    FMAA(c##_batch##0, a##_batch##1 .x, b##_batch##4)  \
    FMAA(c##_batch##0, a##_batch##1 .y, b##_batch##5)  \
    FMAA(c##_batch##0, a##_batch##1 .z, b##_batch##6)  \
    FMAA(c##_batch##0, a##_batch##1 .w, b##_batch##7)  \
    FMAA(c##_batch##0, a##_batch##2 .x, b##_batch##8)  \
    FMAA(c##_batch##0, a##_batch##2 .y, b##_batch##9)  \
    FMAA(c##_batch##0, a##_batch##2 .z, b##_batch##10) \
    FMAA(c##_batch##0, a##_batch##2 .w, b##_batch##11) \
    FMAA(c##_batch##0, a##_batch##3 .x, b##_batch##12) \
    FMAA(c##_batch##0, a##_batch##3 .y, b##_batch##13) \
    FMAA(c##_batch##0, a##_batch##3 .z, b##_batch##14) \
    FMAA(c##_batch##0, a##_batch##3 .w, b##_batch##15) \
    shared_vector_store(&C_s[c_idx##_batch##0][thd_x_vec_offset], c##_batch##0); \

    cg::wait_prior<16 * BATCH_SIZE>(tb);

    k_load = k + TILE_K;
    _b_async_load(0)
    _b_async_load(1)
    _b_async_load(2)
    _b_async_load(3)

    _define_b_regs(0)
    _define_b_regs(1)
    _define_b_regs(2)
    _define_b_regs(3)

    _define_c_regs(0)
    _define_c_regs(1)
    _define_c_regs(2)
    _define_c_regs(3)

    _load_b_regs_batch(0)
    _load_b_regs_batch(1)
    _load_b_regs_batch(2)
    _load_b_regs_batch(3)

    _batch_fma(0)
    _batch_fma(1)
    _batch_fma(2)
    _batch_fma(3)

    __syncthreads();

    for (int row_idx = 0; row_idx < BATCH_TOTAL_ROWS; row_idx++) {
      float value = C_s[row_idx][thd_idx_linear];
      float* C_add_addr = coeff_ptr(C, A_rows[row_idx], k + thd_idx_linear);
      if (value) atomicAdd(C_add_addr, value);
    }
  }
}


__global__ void _block_multiply_reg_storage_orig_4x16x4(const Block * blocks,
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
  static constexpr int BATCH_TOTAL_ROWS = BLOCK_ROWS * BATCH_SIZE;

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

    static_assert(BLOCK_ROWS / BLOCK_Y_DIM == 1);

#define _define_b_regs(_batch) \
    float4 b##_batch##0,  b##_batch##1,  b##_batch##2,  b##_batch##3;  \
    float4 b##_batch##4,  b##_batch##5,  b##_batch##6,  b##_batch##7;  \
    float4 b##_batch##8,  b##_batch##9,  b##_batch##10, b##_batch##11; \
    float4 b##_batch##12, b##_batch##13, b##_batch##14, b##_batch##15;

#define _define_c_regs(_batch) \
    float4 c##_batch##0 = {.x = 0, .y = 0, .z = 0, .w = 0};

#define _load_b_regs_batch(_batch) { \
    int4 cols0 = shared_const_vector_load(&block_col_pattern[(_batch * BLOCK_COLS) + 0]);    \
    int4 cols1 = shared_const_vector_load(&block_col_pattern[(_batch * BLOCK_COLS) + 4]);    \
    int4 cols2 = shared_const_vector_load(&block_col_pattern[(_batch * BLOCK_COLS) + 8]);    \
    int4 cols3 = shared_const_vector_load(&block_col_pattern[(_batch * BLOCK_COLS) + 12]);   \
    b##_batch##0  = global_const_vector_load(coeff_ptr(B, cols0.x, k + thd_x_vec_offset));   \
    b##_batch##1  = global_const_vector_load(coeff_ptr(B, cols0.y, k + thd_x_vec_offset));   \
    b##_batch##2  = global_const_vector_load(coeff_ptr(B, cols0.z, k + thd_x_vec_offset));   \
    b##_batch##3  = global_const_vector_load(coeff_ptr(B, cols0.w, k + thd_x_vec_offset));   \
    b##_batch##4  = global_const_vector_load(coeff_ptr(B, cols1.x, k + thd_x_vec_offset));   \
    b##_batch##5  = global_const_vector_load(coeff_ptr(B, cols1.y, k + thd_x_vec_offset));   \
    b##_batch##6  = global_const_vector_load(coeff_ptr(B, cols1.z, k + thd_x_vec_offset));   \
    b##_batch##7  = global_const_vector_load(coeff_ptr(B, cols1.w, k + thd_x_vec_offset));   \
    b##_batch##8  = global_const_vector_load(coeff_ptr(B, cols2.x, k + thd_x_vec_offset));   \
    b##_batch##9  = global_const_vector_load(coeff_ptr(B, cols2.y, k + thd_x_vec_offset));   \
    b##_batch##10 = global_const_vector_load(coeff_ptr(B, cols2.z, k + thd_x_vec_offset));   \
    b##_batch##11 = global_const_vector_load(coeff_ptr(B, cols2.w, k + thd_x_vec_offset));   \
    b##_batch##12 = global_const_vector_load(coeff_ptr(B, cols3.x, k + thd_x_vec_offset));   \
    b##_batch##13 = global_const_vector_load(coeff_ptr(B, cols3.y, k + thd_x_vec_offset));   \
    b##_batch##14 = global_const_vector_load(coeff_ptr(B, cols3.z, k + thd_x_vec_offset));   \
    b##_batch##15 = global_const_vector_load(coeff_ptr(B, cols3.w, k + thd_x_vec_offset));   \
    }

#define _load_a_regs_batch(_batch) \
    int batch##_batch##_row_offset = _batch * BLOCK_NON_ZEROS; \
    int row_idx##_batch##0 = (0 * blockDim.y + threadIdx.y);   \
    int row_offset##_batch##0 = (row_idx##_batch##0 * BLOCK_COLS) + batch##_batch##_row_offset; \
    float4 a##_batch##00 = shared_const_vector_load(&A_s[row_offset##_batch##0 + 0]);  \
    float4 a##_batch##01 = shared_const_vector_load(&A_s[row_offset##_batch##0 + 4]);  \
    float4 a##_batch##02 = shared_const_vector_load(&A_s[row_offset##_batch##0 + 8]);  \
    float4 a##_batch##03 = shared_const_vector_load(&A_s[row_offset##_batch##0 + 12]);

#define _batch_fma(_batch) \
    int c_idx##_batch##0 = (0 * blockDim.y + threadIdx.y) + (_batch * BLOCK_ROWS); \
    FMAA(c##_batch##0, a##_batch##00 .x, b##_batch##0)  \
    FMAA(c##_batch##0, a##_batch##00 .y, b##_batch##1)  \
    FMAA(c##_batch##0, a##_batch##00 .z, b##_batch##2)  \
    FMAA(c##_batch##0, a##_batch##00 .w, b##_batch##3)  \
    FMAA(c##_batch##0, a##_batch##01 .x, b##_batch##4)  \
    FMAA(c##_batch##0, a##_batch##01 .y, b##_batch##5)  \
    FMAA(c##_batch##0, a##_batch##01 .z, b##_batch##6)  \
    FMAA(c##_batch##0, a##_batch##01 .w, b##_batch##7)  \
    FMAA(c##_batch##0, a##_batch##02 .x, b##_batch##8)  \
    FMAA(c##_batch##0, a##_batch##02 .y, b##_batch##9)  \
    FMAA(c##_batch##0, a##_batch##02 .z, b##_batch##10) \
    FMAA(c##_batch##0, a##_batch##02 .w, b##_batch##11) \
    FMAA(c##_batch##0, a##_batch##03 .x, b##_batch##12) \
    FMAA(c##_batch##0, a##_batch##03 .y, b##_batch##13) \
    FMAA(c##_batch##0, a##_batch##03 .z, b##_batch##14) \
    FMAA(c##_batch##0, a##_batch##03 .w, b##_batch##15) \
    shared_vector_store(&C_s[c_idx##_batch##0][thd_x_vec_offset], c##_batch##0); \

    _define_b_regs(0)
    _define_b_regs(1)
    _define_b_regs(2)
    _define_b_regs(3)

    _define_c_regs(0)
    _define_c_regs(1)
    _define_c_regs(2)
    _define_c_regs(3)


    _load_b_regs_batch(0)
    _load_b_regs_batch(1)

    if (k == 0) {
      cg::wait_prior<2>(tb); // Wait for A_s, A_rows
    }

    _load_a_regs_batch(0)
    _load_a_regs_batch(1)

    _load_b_regs_batch(2)
    _load_b_regs_batch(3)

    _batch_fma(0)
    _batch_fma(1)

    _load_a_regs_batch(2)
    _load_a_regs_batch(3)

    _batch_fma(2)
    _batch_fma(3)

    __syncthreads();

    for (int row_idx = 0; row_idx < BATCH_TOTAL_ROWS; row_idx++) {
      float value = C_s[row_idx][thd_idx_linear];
      float* C_add_addr = coeff_ptr(C, A_rows[row_idx], k + thd_idx_linear);
      if (value) atomicAdd(C_add_addr, value);
    }
  }
}

int
codelet_4x16x4::codelet_multiply(cudaStream_t &stream, cudaEvent_t &start, cudaEvent_t &stop, const Block *blocks,
                                 size_t num_blocks,
                                 const CSR<float> &A_h, const CSR<float> &A, const Dense &B, Dense &C) {
  dim3 grid_dim(num_blocks, 1);

  cudaFuncSetCacheConfig(_block_multiply_bypass_4x16x4, cudaFuncCachePreferShared);
  cudaEventRecord(start, stream);
  _block_multiply_bypass_4x16x4<<<grid_dim, codelet_block_dim, 0, stream>>>(blocks, num_blocks, A, B, C);
  CHECK_CUDA(cudaGetLastError());
  cudaEventRecord(stop, stream);

  return 0;
}
//
// Created by lwilkinson on 12/7/21.
//

#include "codelet_kernel_utils.cuh"

using namespace CodeletMultiply;
using namespace codelet_4x8x8;

static dim3 codelet_block_dim(BLOCK_X_DIM, BLOCK_Y_DIM);

static constexpr int BLOCK_SIZE = MAX_ROWS_PER_BLOCK * MAX_COLS_PER_BLOCK;
static constexpr int BATCH_COL_TOTAL = MAX_COLS_PER_BLOCK * BATCH_SIZE;
static constexpr int BATCH_ROW_TOTAL = MAX_ROWS_PER_BLOCK * BATCH_SIZE;
static constexpr int BATCH_VALUE_TOTAL = BLOCK_SIZE * BATCH_SIZE;

__global__ static void
_block_multiply_reg_storage_4x8x8(const Block *blocks, int num_blocks, const CSR<float> A, const Dense B, Dense C) {
  static_assert(MAX_COLS_PER_BLOCK % VECTOR_WIDTH == 0);

  int block_idx = blockIdx.x;

  const Block *block = &blocks[block_idx];
  cache(A);
  cache(B);
  cache(C);

  // TODO: make CODELET_MULTIPLY_MAX_COLS_PER_BLOCK dynamic
  __shared__ __align__(32) float A_s[BATCH_VALUE_TOTAL];
  __shared__ __align__(32) int block_rows[BATCH_ROW_TOTAL];
  __shared__ __align__(32) int block_col_pattern[BATCH_COL_TOTAL];
  //__shared__ __align__(32) float B_s[BLOCK_COLS][TILE_K];
  __shared__ __align__(32) float C_s[MAX_ROWS_PER_BLOCK * 4][32];

  int *batch_col_pattern_ptr = block->col_pattern;
  int *batch_rows_ptr = block->rows;
  float *batch_values_ptr = block->row_segment_values;

  int thd_idx_linear = threadIdx.x + threadIdx.y * blockDim.x;
  int thd_x_vec_offset = threadIdx.x * VECTOR_WIDTH;
  int thd_linear_vec_offset = thd_idx_linear * VECTOR_WIDTH;

  static constexpr int cuda_block_size = BLOCK_X_DIM * BLOCK_Y_DIM;
  static constexpr int cuda_vector_block_width = cuda_block_size * VECTOR_WIDTH;

  static_assert(BATCH_COL_TOTAL <= 128);
  if (thd_linear_vec_offset < BATCH_COL_TOTAL) {
    global_const_vector_load_int4(&block_col_pattern[thd_linear_vec_offset],
                                  &batch_col_pattern_ptr[thd_linear_vec_offset]);
  }


  //static_assert(BATCH_ROW_TOTAL == 128);
  if (thd_linear_vec_offset < BATCH_ROW_TOTAL) {
    global_const_vector_load_int4(&block_rows[thd_linear_vec_offset], &batch_rows_ptr[thd_linear_vec_offset]);
  }

  __syncthreads();

  //printf("%d %d %d %d ... %d %d\n", block_col_pattern[0], block_col_pattern[1], block_col_pattern[2], block_col_pattern[3], block_col_pattern[30], block_col_pattern[31]);

  int idx;
  static_assert(BATCH_VALUE_TOTAL == 256);
  static_assert(cuda_vector_block_width == 128);
  idx = 0 * cuda_vector_block_width + thd_linear_vec_offset;
  global_const_vector_load_float4(&A_s[idx], &batch_values_ptr[idx]);
  idx = 1 * cuda_vector_block_width + thd_linear_vec_offset;
  global_const_vector_load_float4(&A_s[idx], &batch_values_ptr[idx]);
//  idx = 2 * cuda_vector_block_width + thd_linear_vec_offset;
//  global_const_vector_load_float4(&A_s[idx], &batch_values_ptr[idx]);
//  idx = 3 * cuda_vector_block_width + thd_linear_vec_offset;
//  global_const_vector_load_float4(&A_s[idx], &batch_values_ptr[idx]);

  __syncthreads();

  for (int k = 0; k < B_cols; k += TILE_K) {

#pragma unroll
    for (int block_base = 0; block_base < BATCH_SIZE; block_base += 4) {
      float4 c0 = {.x = 0, .y = 0, .z = 0, .w = 0}; // Block 0
      float4 c1 = {.x = 0, .y = 0, .z = 0, .w = 0}; // Block 1
      float4 c2 = {.x = 0, .y = 0, .z = 0, .w = 0}; // Block 2
      float4 c3 = {.x = 0, .y = 0, .z = 0, .w = 0}; // Block 3

      float4 b0, b1, b2, b3, b4, b5, b6, b7;
      float4 b8, b9, b10, b11, b12, b13, b14, b15;
      float4 b16, b17, b18, b19, b20, b21, b22, b23;
      float4 b24, b25, b26, b27, b28, b29, b30, b31;

      __syncthreads();

      // Block 0
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (0 + block_base), 0, 0);
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (0 + block_base), 1, 1);
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (0 + block_base), 2, 2);
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (0 + block_base), 3, 3);
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (0 + block_base), 4, 4);
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (0 + block_base), 5, 5);
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (0 + block_base), 6, 6);
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (0 + block_base), 7, 7);

      // Block 1
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (1 + block_base), 0, 8);
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (1 + block_base), 1, 9);
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (1 + block_base), 2, 10);
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (1 + block_base), 3, 11);
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (1 + block_base), 4, 12);
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (1 + block_base), 5, 13);
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (1 + block_base), 6, 14);
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (1 + block_base), 7, 15);

      // Block 2
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (2 + block_base), 0, 16);
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (2 + block_base), 1, 17);
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (2 + block_base), 2, 18);
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (2 + block_base), 3, 19);
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (2 + block_base), 4, 20);
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (2 + block_base), 5, 21);
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (2 + block_base), 6, 22);
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (2 + block_base), 7, 23);

      // Block 3
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (3 + block_base), 0, 24);
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (3 + block_base), 1, 25);
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (3 + block_base), 2, 26);
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (3 + block_base), 3, 27);
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (3 + block_base), 4, 28);
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (3 + block_base), 5, 29);
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (3 + block_base), 6, 30);
      _load_b_reg_branchless(MAX_COLS_PER_BLOCK * (3 + block_base), 7, 31);

      float4 a00, a01;
      float4 a10, a11;
      float4 a20, a21;
      float4 a30, a31;

      static_assert(BLOCK_Y_DIM == 4);
      static_assert(MAX_COLS_PER_BLOCK == 8);

      int row_idx = (0 * blockDim.y + threadIdx.y);
      int row_offset = row_idx * MAX_COLS_PER_BLOCK;

      { // Block 0
        int block_offset = (0 + block_base) * BLOCK_SIZE;
        a00 = shared_const_vector_load(&A_s[block_offset + row_offset + 0]);
        a01 = shared_const_vector_load(&A_s[block_offset + row_offset + 4]);
      }

      { // Block 1
        int block_offset = (1 + block_base) * BLOCK_SIZE;
        a10 = shared_const_vector_load(&A_s[block_offset + row_offset + 0]);
        a11 = shared_const_vector_load(&A_s[block_offset + row_offset + 4]);
      }

      { // Block 2
        int block_offset = (2 + block_base) * BLOCK_SIZE;
        a20 = shared_const_vector_load(&A_s[block_offset + row_offset + 0]);
        a21 = shared_const_vector_load(&A_s[block_offset + row_offset + 4]);
      }

      { // Block 3
        int block_offset = (3 + block_base) * BLOCK_SIZE;
        a30 = shared_const_vector_load(&A_s[block_offset + row_offset + 0]);
        a31 = shared_const_vector_load(&A_s[block_offset + row_offset + 4]);
      }

      // Block 0
      FMAA(c0, a00.x, b0);
      FMAA(c0, a00.y, b1);
      FMAA(c0, a00.z, b2);
      FMAA(c0, a00.w, b3);
      FMAA(c0, a01.x, b4);
      FMAA(c0, a01.y, b5);
      FMAA(c0, a01.z, b6);
      FMAA(c0, a01.w, b7);

      {
        int block_offset = (0) * MAX_ROWS_PER_BLOCK;
        shared_vector_store(&C_s[block_offset + row_idx][thd_x_vec_offset], c0);
      }

      // Block 1
      FMAA(c1, a10.x, b8);
      FMAA(c1, a10.y, b9);
      FMAA(c1, a10.z, b10);
      FMAA(c1, a10.w, b11);
      FMAA(c1, a11.x, b12);
      FMAA(c1, a11.y, b13);
      FMAA(c1, a11.z, b14);
      FMAA(c1, a11.w, b15);

      {
        int block_offset = (1) * MAX_ROWS_PER_BLOCK;
        shared_vector_store(&C_s[block_offset + row_idx][thd_x_vec_offset], c1);
      }

      // Block 2
      FMAA(c2, a20.x, b16);
      FMAA(c2, a20.y, b17);
      FMAA(c2, a20.z, b18);
      FMAA(c2, a20.w, b19);
      FMAA(c2, a21.x, b20);
      FMAA(c2, a21.y, b21);
      FMAA(c2, a21.z, b22);
      FMAA(c2, a21.w, b23);

      {
        int block_offset = (2) * MAX_ROWS_PER_BLOCK;
        shared_vector_store(&C_s[block_offset + row_idx][thd_x_vec_offset], c2);
      }

      // Block 3
      FMAA(c3, a30.x, b24);
      FMAA(c3, a30.y, b25);
      FMAA(c3, a30.z, b26);
      FMAA(c3, a30.w, b27);
      FMAA(c3, a31.x, b28);
      FMAA(c3, a31.y, b29);
      FMAA(c3, a31.z, b30);
      FMAA(c3, a31.w, b31);

      {
        int block_offset = (3) * MAX_ROWS_PER_BLOCK;
        shared_vector_store(&C_s[block_offset + row_idx][thd_x_vec_offset], c3);
      }

      __syncthreads();

      static_assert(MAX_ROWS_PER_BLOCK == 4);
      {
        int block_offset = (0 + block_base) * MAX_ROWS_PER_BLOCK;
        int block_offset_c = (0) * MAX_ROWS_PER_BLOCK;
        //printf("k: %d, row %d\n", k, block_rows[block_offset + 0]);
        atomicAdd(coeff_ptr(C, block_rows[block_offset + 0], k + thd_idx_linear),
                  C_s[block_offset_c + 0][thd_idx_linear]);
        atomicAdd(coeff_ptr(C, block_rows[block_offset + 1], k + thd_idx_linear),
                  C_s[block_offset_c + 1][thd_idx_linear]);
        atomicAdd(coeff_ptr(C, block_rows[block_offset + 2], k + thd_idx_linear),
                  C_s[block_offset_c + 2][thd_idx_linear]);
        atomicAdd(coeff_ptr(C, block_rows[block_offset + 3], k + thd_idx_linear),
                  C_s[block_offset_c + 3][thd_idx_linear]);
      }

      {
        int block_offset = (1 + block_base) * MAX_ROWS_PER_BLOCK;
        int block_offset_c = (0) * MAX_ROWS_PER_BLOCK;
        atomicAdd(coeff_ptr(C, block_rows[block_offset + 0], k + thd_idx_linear),
                  C_s[block_offset_c + 0][thd_idx_linear]);
        atomicAdd(coeff_ptr(C, block_rows[block_offset + 1], k + thd_idx_linear),
                  C_s[block_offset_c + 1][thd_idx_linear]);
        atomicAdd(coeff_ptr(C, block_rows[block_offset + 2], k + thd_idx_linear),
                  C_s[block_offset_c + 2][thd_idx_linear]);
        atomicAdd(coeff_ptr(C, block_rows[block_offset + 3], k + thd_idx_linear),
                  C_s[block_offset_c + 3][thd_idx_linear]);
      }
      {
        int block_offset = (2 + block_base) * MAX_ROWS_PER_BLOCK;
        int block_offset_c = (2) * MAX_ROWS_PER_BLOCK;
        atomicAdd(coeff_ptr(C, block_rows[block_offset + 0], k + thd_idx_linear),
                  C_s[block_offset_c + 0][thd_idx_linear]);
        atomicAdd(coeff_ptr(C, block_rows[block_offset + 1], k + thd_idx_linear),
                  C_s[block_offset_c + 1][thd_idx_linear]);
        atomicAdd(coeff_ptr(C, block_rows[block_offset + 2], k + thd_idx_linear),
                  C_s[block_offset_c + 2][thd_idx_linear]);
        atomicAdd(coeff_ptr(C, block_rows[block_offset + 3], k + thd_idx_linear),
                  C_s[block_offset_c + 3][thd_idx_linear]);
      }

      {
        int block_offset = (3 + block_base) * MAX_ROWS_PER_BLOCK;
        int block_offset_c = (3) * MAX_ROWS_PER_BLOCK;
        atomicAdd(coeff_ptr(C, block_rows[block_offset + 0], k + thd_idx_linear),
                  C_s[block_offset_c + 0][thd_idx_linear]);
        atomicAdd(coeff_ptr(C, block_rows[block_offset + 1], k + thd_idx_linear),
                  C_s[block_offset_c + 1][thd_idx_linear]);
        atomicAdd(coeff_ptr(C, block_rows[block_offset + 2], k + thd_idx_linear),
                  C_s[block_offset_c + 2][thd_idx_linear]);
        atomicAdd(coeff_ptr(C, block_rows[block_offset + 3], k + thd_idx_linear),
                  C_s[block_offset_c + 3][thd_idx_linear]);
      }
    }
  }
}

int
codelet_4x8x8::codelet_multiply(cudaStream_t &stream, cudaEvent_t &start, cudaEvent_t &stop, const Block *blocks,
                 size_t num_blocks,
                 const CSR<float> &A_h, const CSR<float> &A, const Dense &B, Dense &C) {
  dim3 grid_dim(num_blocks, 1);

  cudaEventRecord(start, stream);
  _block_multiply_reg_storage_4x8x8<<<grid_dim, codelet_block_dim, 0, stream>>>(blocks, num_blocks, A, B, C);
  CHECK_CUDA(cudaGetLastError());
  cudaEventRecord(stop, stream);

  return 0;
}
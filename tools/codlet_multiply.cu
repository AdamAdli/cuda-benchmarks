//
// Created by lwilkinson on 11/25/21.
//

#include "common/utils/cuda_utils.h"
#include "codlet_multiply.cuh"

using namespace CodeletMultiply;

#define cache(A) \
    float* A##_values = A.values; \
    int A##_cols = A.cols;

#define coeff(A, i, j) \
    &A##_values[i * A##_cols + j]

#define coeff_ptr(A, i, j) \
    &A##_values[i * A##_cols + j]

__device__ __forceinline__
void vec_atomic_add_coeff(Dense& A, int i, int j, float4 val) {
  atomicAdd(&A.values[i * A.cols + j + 1], val.y);
  atomicAdd(&A.values[i * A.cols + j + 0], val.x);
  atomicAdd(&A.values[i * A.cols + j + 2], val.z);
  atomicAdd(&A.values[i * A.cols + j + 3], val.w);

//    *reinterpret_cast<float4*>(&A.values[i * A.cols + j + 0]) = val;

//    A.values[i * A.cols + j + 0] = val.x;
//    A.values[i * A.cols + j + 1] = val.y;
//    A.values[i * A.cols + j + 2] = val.z;
//    A.values[i * A.cols + j + 3] = val.w;
}

//__device__ __forceinline__
//float coeff(const Block& A, int row, int col_pattern_idx) {
//  return A.row_segment_values[row * A.col_pattern_len + col_pattern_idx];
//}

#define FMAA(accumulate, a, b) \
  accumulate.x += (a) * b.x;   \
  accumulate.y += (a) * b.y;   \
  accumulate.z += (a) * b.z;   \
  accumulate.w += (a) * b.w;

#define global_const_vector_load_float4(dst, src) \
  *reinterpret_cast<float4 *>(dst) = __ldg(reinterpret_cast<float4 *>(src));

#define global_const_vector_load_int4(dst, src) \
  *reinterpret_cast<int4 *>(dst) = __ldg(reinterpret_cast<int4 *>(src));


#define _load_b_reg_branchless(base_idx, idx) {                                             \
    int col_idx = base_idx + idx;                                                           \
    int col = block_col_pattern[col_idx];                                                   \
    b##idx = global_const_vector_load(coeff_ptr(B, col, k + thd_linear_vec_offset));        \
}

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
void shared_vector_store(float *__restrict__ dst, float4 src) {
  *reinterpret_cast<float4 *>(dst) = src;
}


dim3 codelet_block_dim(BLOCK_X_DIM, BLOCK_Y_DIM);

__global__ void _block_multiply_reg_storage(const Block * blocks, int num_blocks, const CSR<float> A, const Dense B, Dense C) {
  int block_idx = blockIdx.x;

  const Block *block = &blocks[block_idx];

  cache(A);
  cache(B);
  cache(C);

  // TODO: make CODELET_MULTIPLY_MAX_COLS_PER_BLOCK dynamic
  __shared__ __align__(32) float A_s[MAX_ROWS_PER_BLOCK * MAX_COLS_PER_BLOCK];
  __shared__ __align__(32) int A_rows[MAX_ROWS_PER_BLOCK];
  __shared__ __align__(32) int block_col_pattern[MAX_COLS_PER_BLOCK];
  //__shared__ __align__(32) float B_s[MAX_COLS_PER_BLOCK][TILE_K];
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


  if (thd_linear_vec_offset < block_col_pattern_len) {
    global_const_vector_load_int4(&block_col_pattern[thd_linear_vec_offset], &block_col_pattern_ptr[thd_linear_vec_offset]);
  }


  if (thd_linear_vec_offset < block_num_rows) {
    global_const_vector_load_int4(&A_rows[thd_linear_vec_offset], &block_rows_ptr[thd_linear_vec_offset]);
  }
  __syncthreads();

  //printf("%d %d %d %d ... %d %d\n", block_col_pattern[0], block_col_pattern[1], block_col_pattern[2], block_col_pattern[3], block_col_pattern[30], block_col_pattern[31]);

  for (int i = 0; i < a_vector_loads_full_block; i ++) {

    int dst_idx = (i * block_size * VECTOR_WIDTH) + (thd_linear_vec_offset);
    int src_idx = (i * block_size * VECTOR_WIDTH) + (thd_linear_vec_offset);

    global_const_vector_load_float4(&A_s[dst_idx], &row_segment_values_ptr[src_idx]);
  }

  if (thd_idx_linear < a_vector_partial_load) {

    int dst_idx = (a_vector_loads_partial_block_start) + (thd_linear_vec_offset);
    int src_idx = (a_vector_loads_partial_block_start) + (thd_linear_vec_offset);

    global_const_vector_load_float4(&A_s[dst_idx], &row_segment_values_ptr[src_idx]);
  }
  __syncthreads();

  for (int k = 0; k < B_cols; k+= TILE_K) {

    static_assert(MAX_ROWS_PER_BLOCK / BLOCK_Y_DIM  == 2);
    float4 c0 = { .x = 0, .y = 0, .z = 0, .w = 0 };
    float4 c1 = { .x = 0, .y = 0, .z = 0, .w = 0 };

    for (int col_pattern_idx = 0; col_pattern_idx < block_col_pattern_len; col_pattern_idx += 16) {
      float4 b0, b1, b2, b3, b4, b5, b6, b7;
      float4 b8, b9, b10, b11, b12, b13, b14, b15;

      _load_b_reg_branchless(col_pattern_idx, 0);
      _load_b_reg_branchless(col_pattern_idx, 1);
      _load_b_reg_branchless(col_pattern_idx, 2);
      _load_b_reg_branchless(col_pattern_idx, 3);
      _load_b_reg_branchless(col_pattern_idx, 4);
      _load_b_reg_branchless(col_pattern_idx, 5);
      _load_b_reg_branchless(col_pattern_idx, 6);
      _load_b_reg_branchless(col_pattern_idx, 7);

      _load_b_reg_branchless(col_pattern_idx, 8);
      _load_b_reg_branchless(col_pattern_idx, 9);
      _load_b_reg_branchless(col_pattern_idx, 10);
      _load_b_reg_branchless(col_pattern_idx, 11);
      _load_b_reg_branchless(col_pattern_idx, 12);
      _load_b_reg_branchless(col_pattern_idx, 13);
      _load_b_reg_branchless(col_pattern_idx, 14);
      _load_b_reg_branchless(col_pattern_idx, 15);

      int row_idx0 = (0 * blockDim.y + threadIdx.y);
      int row_offset = row_idx0 * block_col_pattern_len;
      float4 a0 = shared_const_vector_load(&A_s[row_offset + col_pattern_idx + 0]);
      float4 a1 = shared_const_vector_load(&A_s[row_offset + col_pattern_idx + 4]);
      float4 a2 = shared_const_vector_load(&A_s[row_offset + col_pattern_idx + 8]);
      float4 a3 = shared_const_vector_load(&A_s[row_offset + col_pattern_idx + 12]);

      int row_idx1 = (1 * blockDim.y + threadIdx.y);
      row_offset = row_idx1 * block_col_pattern_len;
      float4 a4 = shared_const_vector_load(&A_s[row_offset + col_pattern_idx + 0]);
      float4 a5 = shared_const_vector_load(&A_s[row_offset + col_pattern_idx + 4]);
      float4 a6 = shared_const_vector_load(&A_s[row_offset + col_pattern_idx + 8]);
      float4 a7 = shared_const_vector_load(&A_s[row_offset + col_pattern_idx + 12]);

      //if (row_idx < block->num_rows) {
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
      shared_vector_store(&C_s[row_idx0][thd_x_vec_offset], c0);
      //}

      //if (row_idx < block->num_rows) {
      FMAA(c1, a4.x, b0);
      FMAA(c1, a4.y, b1);
      FMAA(c1, a4.z, b2);
      FMAA(c1, a4.w, b3);
      FMAA(c1, a5.x, b4);
      FMAA(c1, a5.y, b5);
      FMAA(c1, a5.z, b6);
      FMAA(c1, a5.w, b7);

      FMAA(c1, a6.x, b8);
      FMAA(c1, a6.y, b9);
      FMAA(c1, a6.z, b10);
      FMAA(c1, a6.w, b11);
      FMAA(c1, a7.x, b12);
      FMAA(c1, a7.y, b13);
      FMAA(c1, a7.z, b14);
      FMAA(c1, a7.w, b15);
      //}
      shared_vector_store(&C_s[row_idx1][thd_x_vec_offset], c1);

      __syncthreads();
      //}
    }

//    for (int col_pattern_idx = col_pattern_idx_end_of_aligned;
//         col_pattern_idx < block->col_pattern_len; col_pattern_idx++) {
//
//      int row_idx = (0 * blockDim.y + threadIdx.y);
//      int row_offset = row_idx * block->col_pattern_len;
//      //if (row_idx < block->num_rows) {
//      float4 b = global_const_vector_load(&B_s[col_pattern_idx + 0][thd_x_vec_offset]);
//      FMAA(c0, A_s[row_offset + col_pattern_idx + 0], b);
//      //}
//
//      row_idx = (1 * blockDim.y + threadIdx.y);
//      row_offset = row_idx * block->col_pattern_len;
//      //if (row_idx < block->num_rows) {
//      b = global_const_vector_load(&B_s[col_pattern_idx + 0][thd_x_vec_offset]);
//      FMAA(c1, A_s[row_offset + col_pattern_idx + 0], b);
//      //}
//
//      __syncthreads();
//    }


    for (int row_idx = 0; row_idx < block_num_rows; row_idx++) {
      atomicAdd(coeff_ptr(C, A_rows[row_idx], k + thd_idx_linear), C_s[row_idx][thd_idx_linear]);
    }

//    int row_idx = (0 * blockDim.y + threadIdx.y);
//    if (row_idx < block->num_rows) {
//      vec_atomic_add_coeff(C, block->rows[row_idx], k + thd_x_vec_offset, c0);
//    }
//
//    row_idx = (1 * blockDim.y + threadIdx.y);
//    if (row_idx < block->num_rows) {
//      vec_atomic_add_coeff(C, block->rows[row_idx], k + thd_x_vec_offset, c1);
//    }
  }
}


__global__ void _block_multiply(const Block * blocks, int num_blocks, const CSR<float> A, const Dense B, Dense C) {
  int block_idx = blockIdx.x;

  cache(A);
  cache(B);
  cache(C);

  const Block *block = &blocks[block_idx];

  // TODO: make CODELET_MULTIPLY_MAX_COLS_PER_BLOCK dynamic
  __shared__ __align__(32) float A_s[MAX_ROWS_PER_BLOCK * MAX_COLS_PER_BLOCK];
  __shared__ __align__(32) float B_s[MAX_COLS_PER_BLOCK][TILE_K];

  int non_zeros = block->num_rows * block->col_pattern_len;
  int thd_idx_linear = threadIdx.x + threadIdx.y * blockDim.x;
  int thd_x_vec_offset = threadIdx.x * VECTOR_WIDTH;
  int thd_linear_vec_offset = thd_idx_linear * VECTOR_WIDTH;
  int block_size = blockDim.x * blockDim.y;


  int vector_load_block_width = block_size * VECTOR_WIDTH;
  int a_vector_loads_full_block = non_zeros / vector_load_block_width;
  int a_vector_loads_partial_block_start = a_vector_loads_full_block * vector_load_block_width;


  int a_vector_partial_load = non_zeros - a_vector_loads_partial_block_start;

  for (int i = 0; i < a_vector_loads_full_block; i ++) {

    int dst_idx = (i * block_size * VECTOR_WIDTH) + (thd_linear_vec_offset);
    int src_idx = (i * block_size * VECTOR_WIDTH) + (thd_linear_vec_offset);

    global_const_vector_load_float4(&A_s[dst_idx], &block->row_segment_values[src_idx]);
  }

  if (thd_idx_linear < a_vector_partial_load) {

    int dst_idx = (a_vector_loads_partial_block_start) + (thd_linear_vec_offset);
    int src_idx = (a_vector_loads_partial_block_start) + (thd_linear_vec_offset);

    global_const_vector_load_float4(&A_s[dst_idx], &block->row_segment_values[src_idx]);
  }

  __syncthreads();

  for (int k = 0; k < B.cols; k+= TILE_K) {

    for (int col_pattern_idx = threadIdx.y; col_pattern_idx < block->col_pattern_len; col_pattern_idx += blockDim.y) {
      int col = block->col_pattern[col_pattern_idx];

      global_const_vector_load_float4(&B_s[col_pattern_idx][thd_x_vec_offset], coeff_ptr(B, col, k + thd_x_vec_offset));
    }

    static_assert(MAX_ROWS_PER_BLOCK / BLOCK_Y_DIM  == 2);
    float4 c0 = { .x = 0, .y = 0, .z = 0, .w = 0 };
    float4 c1 = { .x = 0, .y = 0, .z = 0, .w = 0 };

    int col_pattern_idx_end_of_aligned = (block->col_pattern_len / 4) * 4;
    for (int col_pattern_idx = 0; col_pattern_idx < col_pattern_idx_end_of_aligned; col_pattern_idx += 4) {

      __syncthreads();

      float4 b0 = shared_const_vector_load(&B_s[col_pattern_idx + 0][thd_x_vec_offset]);
      float4 b1 = shared_const_vector_load(&B_s[col_pattern_idx + 1][thd_x_vec_offset]);
      float4 b2 = shared_const_vector_load(&B_s[col_pattern_idx + 2][thd_x_vec_offset]);
      float4 b3 = shared_const_vector_load(&B_s[col_pattern_idx + 3][thd_x_vec_offset]);

      int row_idx = (0 * blockDim.y + threadIdx.y);
      int row_offset = row_idx * block->col_pattern_len;
      float4 a = shared_const_vector_load(&A_s[row_offset + col_pattern_idx + 0]);

      //if (row_idx < block->num_rows) {
        FMAA(c0, a.x, b0);
        FMAA(c0, a.y, b1);
        FMAA(c0, a.z, b2);
        FMAA(c0, a.w, b3);
      //}

      row_idx = (1 * blockDim.y + threadIdx.y);
      row_offset = row_idx * block->col_pattern_len;
      a = shared_const_vector_load(&A_s[row_offset + col_pattern_idx + 0]);

      //if (row_idx < block->num_rows) {
        FMAA(c1, a.x, b0);
        FMAA(c1, a.y, b1);
        FMAA(c1, a.z, b2);
        FMAA(c1, a.w, b3);
      //}
    }

    for (int col_pattern_idx = col_pattern_idx_end_of_aligned;
         col_pattern_idx < block->col_pattern_len; col_pattern_idx++) {

      int row_idx = (0 * blockDim.y + threadIdx.y);
      int row_offset = row_idx * block->col_pattern_len;
      //if (row_idx < block->num_rows) {
        float4 b = shared_const_vector_load(&B_s[col_pattern_idx + 0][thd_x_vec_offset]);
        FMAA(c0, A_s[row_offset + col_pattern_idx + 0], b);
      //}

      row_idx = (1 * blockDim.y + threadIdx.y);
      row_offset = row_idx * block->col_pattern_len;
      //if (row_idx < block->num_rows) {
        b = shared_const_vector_load(&B_s[col_pattern_idx + 0][thd_x_vec_offset]);
        FMAA(c1, A_s[row_offset + col_pattern_idx + 0], b);
      //}

      __syncthreads();
    }

    int row_idx = (0 * blockDim.y + threadIdx.y);
    if (row_idx < block->num_rows) {
      vec_atomic_add_coeff(C, block->rows[row_idx], k + thd_x_vec_offset, c0);
    }

    row_idx = (1 * blockDim.y + threadIdx.y);
    if (row_idx < block->num_rows) {
      vec_atomic_add_coeff(C, block->rows[row_idx], k + thd_x_vec_offset, c1);
    }
  }
}

int CodeletMultiply::codelet_multiply(cudaStream_t& stream, cudaEvent_t& start, cudaEvent_t& stop, const Block * blocks, size_t num_blocks,
                     const CSR<float>& A_h, const CSR<float>& A, const Dense& B, Dense& C) {
  dim3 grid_dim(num_blocks, 1);

  cudaEventRecord(start, stream);
  _block_multiply_reg_storage<<<grid_dim, codelet_block_dim, 0, stream>>>(blocks, num_blocks, A, B, C);
  CHECK_CUDA(cudaGetLastError());
  cudaEventRecord(stop, stream);

  return 0;
}


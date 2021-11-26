//
// Created by lwilkinson on 11/25/21.
//

#include "common/utils/cuda_utils.h"
#include "codlet_multiply.cuh"

using namespace CodeletMultiply;

__device__ __forceinline__
float coeff(const Dense& A, int i, int j) {
  return A.values[i * A.cols + j];
}

__device__ __forceinline__
float* coeff_ptr(const Dense& A, int i, int j) {
  return &A.values[i * A.cols + j];
}

__device__ __forceinline__
float atomic_add_coeff(Dense& A, int i, int j, float val) {
  atomicAdd(&A.values[i * A.cols + j], val);
}

__device__ __forceinline__
void vec_atomic_add_coeff(Dense& A, int i, int j, float4 val) {
  atomicAdd(&A.values[i * A.cols + j + 0], val.x);
  atomicAdd(&A.values[i * A.cols + j + 1], val.y);
  atomicAdd(&A.values[i * A.cols + j + 2], val.z);
  atomicAdd(&A.values[i * A.cols + j + 3], val.w);

//    *reinterpret_cast<float4*>(&A.values[i * A.cols + j + 0]) = val;

//    A.values[i * A.cols + j + 0] = val.x;
//    A.values[i * A.cols + j + 1] = val.y;
//    A.values[i * A.cols + j + 2] = val.z;
//    A.values[i * A.cols + j + 3] = val.w;
}

__device__ __forceinline__
float coeff(const Block& A, int row, int col_pattern_idx) {
  return A.row_segment_values[row * A.col_pattern_len + col_pattern_idx];
}

__device__ __forceinline__
void FMAA(float4 &accumulate, float a, float *__restrict__ b){
  auto b4 = *reinterpret_cast<float4 *>(b);

  accumulate.x += (a) * b4.x;
  accumulate.y += (a) * b4.y;
  accumulate.z += (a) * b4.z;
  accumulate.w += (a) * b4.w;
}

__device__ __forceinline__
void FMAA(float4 &accumulate, float a, float4 b){
  accumulate.x += (a) * b.x;
  accumulate.y += (a) * b.y;
  accumulate.z += (a) * b.z;
  accumulate.w += (a) * b.w;
}

template<int vector_size>
__device__ __forceinline__
void vector_load(float *__restrict__ dst, float *__restrict__ src);

template<>
__device__ __forceinline__
void vector_load<4>(float *__restrict__ dst, float *__restrict__ src) {
  *reinterpret_cast<float4 *>(dst) = *reinterpret_cast<float4 *>(src);
}

__device__ __forceinline__
float4  vector_load(float *__restrict__ src) {
  return *reinterpret_cast<float4 *>(src);
}

dim3 codelet_block_dim(BLOCK_X_DIM, BLOCK_Y_DIM);

// Taken from: https://stackoverflow.com/a/18856054
__global__ void _block_multiply(const Block * blocks, int num_blocks, const CSR<float> A, const Dense B, Dense C) {
  int block_idx = blockIdx.x;

  const Block *block = &blocks[block_idx];

  // TODO: make CODELET_MULTIPLY_MAX_COLS_PER_BLOCK dynamic
  __shared__ __align__(32) float A_s[MAX_ROWS_PER_BLOCK * MAX_COLS_PER_BLOCK];
  __shared__ __align__(32) float B_s[MAX_COLS_PER_BLOCK][TILE_K];

  int non_zeros = block->num_rows * block->col_pattern_len;
  int thread_idx_linear = threadIdx.x + threadIdx.y * blockDim.x;
  int thread_vec_offset = threadIdx.x * VECTOR_WIDTH;
  int thread_vec_offset_linear = thread_idx_linear * VECTOR_WIDTH;
  int block_size = blockDim.x * blockDim.y;


  int vector_load_block_width = block_size * VECTOR_WIDTH;
  int a_vector_loads_full_block = non_zeros / vector_load_block_width;
  int a_vector_loads_partial_block_start = a_vector_loads_full_block * vector_load_block_width;


  int a_vector_partial_load = non_zeros - a_vector_loads_partial_block_start;

  for (int i = 0; i < a_vector_loads_full_block; i ++) {

    int dst_idx = (i * block_size * VECTOR_WIDTH) + (thread_vec_offset_linear);
    int src_idx = (i * block_size * VECTOR_WIDTH) + (thread_vec_offset_linear);

    vector_load<4>(&A_s[src_idx], &block->row_segment_values[dst_idx]);
  }

  if (thread_idx_linear < a_vector_partial_load) {

    int dst_idx = (a_vector_loads_partial_block_start) + (thread_vec_offset_linear);
    int src_idx = (a_vector_loads_partial_block_start) + (thread_vec_offset_linear);

    vector_load<4>(&A_s[src_idx], &block->row_segment_values[dst_idx]);
  }

  __syncthreads();

  for (int k = 0; k < B.cols; k+= TILE_K) {

    for (int col_pattern_idx = threadIdx.y; col_pattern_idx < block->col_pattern_len; col_pattern_idx += blockDim.y) {
      int col = block->col_pattern[col_pattern_idx];

      vector_load<4>(&B_s[col_pattern_idx][thread_vec_offset], coeff_ptr(B, col, k + thread_vec_offset));
    }

    __syncthreads();

    float4 c[MAX_ROWS_PER_BLOCK / BLOCK_Y_DIM] = {0};
#pragma unroll
    for (int row_iter = 0; row_iter < MAX_ROWS_PER_BLOCK / BLOCK_Y_DIM; row_iter++) {
      c[row_iter] = { .x = 0, .y = 0, .z = 0, .w = 0 };
    }

    int col_pattern_idx_end_of_aligned = (block->col_pattern_len / 4) * 4;
    for (int col_pattern_idx = 0; col_pattern_idx < col_pattern_idx_end_of_aligned; col_pattern_idx += 4) {

      float4 b0 = vector_load(&B_s[col_pattern_idx + 0][thread_vec_offset]);
      float4 b1 = vector_load(&B_s[col_pattern_idx + 1][thread_vec_offset]);
      float4 b2 = vector_load(&B_s[col_pattern_idx + 2][thread_vec_offset]);
      float4 b3 = vector_load(&B_s[col_pattern_idx + 3][thread_vec_offset]);

#pragma unroll
      for (int row_iter = 0; row_iter < MAX_ROWS_PER_BLOCK / BLOCK_Y_DIM; row_iter++) {
        int row_idx = (row_iter * blockDim.y + threadIdx.y);
        int row_offset = row_idx * block->col_pattern_len;
        float4 a = *reinterpret_cast<float4 *>(&A_s[row_offset + col_pattern_idx + 0]);

        if (row_idx < block->num_rows) {
          FMAA(c[row_iter], a.x, b0);
          FMAA(c[row_iter], a.y, b1);
          FMAA(c[row_iter], a.z, b2);
          FMAA(c[row_iter], a.w, b3);
        }
      }

      __syncthreads();
    }

    for (int col_pattern_idx = col_pattern_idx_end_of_aligned;
         col_pattern_idx < block->col_pattern_len; col_pattern_idx++) {

#pragma unroll
      for (int row_iter = 0; row_iter < MAX_ROWS_PER_BLOCK / BLOCK_Y_DIM; row_iter++) {
        int row_idx = (row_iter * blockDim.y + threadIdx.y);
        int row_offset = row_idx * block->col_pattern_len;

        if (row_idx < block->num_rows) {
          FMAA(c[row_iter], A_s[row_offset + col_pattern_idx + 0], &B_s[col_pattern_idx + 0][thread_vec_offset]);
        }
      }

      __syncthreads();
    }

#pragma unroll
    for (int row_iter = 0;
         row_iter < MAX_ROWS_PER_BLOCK / BLOCK_Y_DIM; row_iter++) {
      int row_idx = (row_iter * threadIdx.y);
      int row_offset = row_idx * block->col_pattern_len;

      if (row_idx < block->num_rows) {
        vec_atomic_add_coeff(C, block->rows[(row_idx * threadIdx.y)], k + thread_vec_offset, c[row_iter]);
      }
    }
  }

//      __shared__ __align__(32) float C_s[4][CODELET_MULTIPLY_TILE_K];
//      *reinterpret_cast<float4 *>(&C_s[threadIdx.y][thread_vec_offset + 0]) = c;
//      if (coalesced_atomic_add_offset + 4 < block->num_rows) {
//
//        atomic_add_coeff(C, block->rows[coalesced_atomic_add_offset + 0], k + thread_idx_linear, C_s[0][thread_idx_linear]);
//        atomic_add_coeff(C, block->rows[coalesced_atomic_add_offset + 1], k + thread_idx_linear, C_s[1][thread_idx_linear]);
//        atomic_add_coeff(C, block->rows[coalesced_atomic_add_offset + 2], k + thread_idx_linear, C_s[2][thread_idx_linear]);
//        atomic_add_coeff(C, block->rows[coalesced_atomic_add_offset + 3], k + thread_idx_linear, C_s[3][thread_idx_linear]);
//
//        coalesced_atomic_add_offset += 4;
//      } else {
//        for (int i = 0; coalesced_atomic_add_offset < block->num_rows; coalesced_atomic_add_offset++, i++) {
//          atomic_add_coeff(C, block->rows[coalesced_atomic_add_offset], k + thread_idx_linear, C_s[i][thread_idx_linear]);
//        }
//      };
}

int CodeletMultiply::codelet_multiply(cudaStream_t& stream, cudaEvent_t& start, cudaEvent_t& stop, const Block * blocks, size_t num_blocks,
                     const CSR<float>& A_h, const CSR<float>& A, const Dense& B, Dense& C) {
  dim3 grid_dim(num_blocks, 1);

  cudaEventRecord(start, stream);
  _block_multiply<<<grid_dim, codelet_block_dim, 0, stream>>>(blocks, num_blocks, A, B, C);
  CHECK_CUDA(cudaGetLastError());
  cudaEventRecord(stop, stream);

  return 0;
}


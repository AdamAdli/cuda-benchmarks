//
// Created by lwilkinson on 11/4/21.
//

#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "cublas_v2.h"
#include "cupti_profiler.h"
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <array>
#include <numeric>
#include "common/utils/matrix_utils.h"
#include "common/utils/cuda_utils.h"
#include "sputnik/sputnik.h"

#include "synthetic_codlets.cuh"
#include "codlet_multiply.cuh"

using namespace std;

#define CHECK_CUDA(func)                                                        \
    {                                                                           \
        cudaError_t status = (func);                                            \
        if (status != cudaSuccess)                                              \
        {                                                                       \
            printf("CUDA API failed at %s:%d with error: %s (%d)\n",            \
                   __FILE__, __LINE__, cudaGetErrorString(status), status);     \
            return EXIT_FAILURE;                                                \
        }                                                                       \
    }

struct Chunks {
  int * col_pattern;
  int * row_pattern;
  int * row_offsets;
};


// Taken from: https://stackoverflow.com/a/18856054
//__global__ void _codelet_multiply(const std::vector<> Dense A, Dense B, Dense C)
//{
//
//}

static const int TILE_DIM = 32;
static const int ITERATIONS = 1;

int cublas_multiply(cudaStream_t& stream, cudaEvent_t& start, cudaEvent_t& stop, const Dense& A, const Dense& B, Dense& C)
{
    float alpha = 1.0, beta = 1.0;
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate_v2(&handle));
    cudaEventRecord(start, stream);

    // https://stackoverflow.com/a/56064726
    CHECK_CUBLAS(cublasSgemm(handle,
      CUBLAS_OP_N, CUBLAS_OP_N,
      C.cols, C.rows, A.cols, /* m, n, k */
      &alpha,
      B.values, B.cols, /* *A, lda */
      A.values, A.cols, /* *B, lda */
      &beta,
      C.values, C.cols))
    cudaEventRecord(stop, stream);
    return 0;
}

// Taken from: https://stackoverflow.com/a/18856054
__global__ void _dense_multiply(const Dense& A, const Dense& B, const Dense& C)
{
  float CValue = 0;

  int Row = blockIdx.y*TILE_DIM + threadIdx.y;
  int Col = blockIdx.x*TILE_DIM + threadIdx.x;

  __shared__ float As[TILE_DIM][TILE_DIM];
  __shared__ float Bs[TILE_DIM][TILE_DIM];

  for (int k = 0; k < (A.cols + TILE_DIM - 1)/TILE_DIM; k++) {

    if (k*TILE_DIM + threadIdx.x < A.cols && Row < A.rows)
      As[threadIdx.y][threadIdx.x] = A.values[Row * A.cols + k*TILE_DIM + threadIdx.x];
    else
      As[threadIdx.y][threadIdx.x] = 0.0;

    if (k*TILE_DIM + threadIdx.y < B.rows && Col < B.cols)
      Bs[threadIdx.y][threadIdx.x] = B.values[(k*TILE_DIM + threadIdx.y) * B.cols + Col];
    else
      Bs[threadIdx.y][threadIdx.x] = 0.0;

    __syncthreads();

    for (int n = 0; n < TILE_DIM; ++n)
      CValue += As[threadIdx.y][n] * Bs[n][threadIdx.x];

    __syncthreads();
  }


  if (Row < C.rows && Col < C.cols) {
    C.values[(Row * C.cols) + Col] = CValue;
  }
}

int dense_multiply(cudaStream_t& stream, cudaEvent_t& start, cudaEvent_t& stop, const Dense& A, const Dense& B, const Dense& C) {
  dim3 block_dim(TILE_DIM, TILE_DIM);
  dim3 grid_dim((C.cols + (TILE_DIM - 1)) / TILE_DIM, (C.rows + (TILE_DIM - 1)) / TILE_DIM);

  cudaEventRecord(start, stream);
  _dense_multiply<<<grid_dim, block_dim, 0, stream>>>(A, B, C);
  CHECK_CUDA(cudaGetLastError());
  cudaEventRecord(stop, stream);

  return 0;
}

int sgk_multiply(cudaStream_t& stream, cudaEvent_t& start, cudaEvent_t& stop, const CSR<float>& A_h, const CSR<float>& A, const Dense& B, Dense& C) {
  float* bias = nullptr;

  // Sort rows - Copied from sputnik
  // Create our unsorted row indices.
  std::vector<int> swizzle_staging(A_h.rows);
  std::iota(swizzle_staging.begin(), swizzle_staging.end(), 0);

  // Argsort the row indices based on their length.
  std::sort(swizzle_staging.begin(), swizzle_staging.end(),
            [&A_h](int idx_a, int idx_b) {
              int length_a = A_h.row_offsets[idx_a + 1] - A_h.row_offsets[idx_a];
              int length_b = A_h.row_offsets[idx_b + 1] - A_h.row_offsets[idx_b];
              return length_a > length_b;
            });

  int *swizzle_d;
  CHECK_CUDA(cudaMalloc((void **)&swizzle_d, sizeof(int) * A_h.rows));
  CHECK_CUDA(cudaMemcpy(swizzle_d, swizzle_staging.data(), sizeof(int) * A_h.rows, cudaMemcpyHostToDevice));

  cudaDeviceSynchronize();

  cudaEventRecord(start, stream);
  CHECK_CUDA(sputnik::CudaSpmmBiasRelu(A.rows, A.cols, B.cols, A.nnz, swizzle_d,
                                       A.values, A.row_offsets, A.col_indices,
                                       B.values,
                                       bias,
                                       C.values,
                                       stream))
  cudaEventRecord(stop, stream);
  CHECK_CUDA(cudaFree(swizzle_d))
  return 0;
}

int run_kernel(const Dense& A, const Dense& B, const Dense& C, const std::string& name,
                int(*kernel)(cudaStream_t& stream, cudaEvent_t& start, cudaEvent_t& stop, const Dense& A, const Dense& B, Dense& C)) {
  float *A_values_d, *B_values_d, *C_values_d;

  CHECK_CUDA(cudaMalloc(&A_values_d, A.rows * A.cols * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&B_values_d, B.rows * B.cols * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&C_values_d, C.rows * C.cols * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(A_values_d, A.values, A.rows * A.cols * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(B_values_d, B.values, B.rows * B.cols * sizeof(float), cudaMemcpyHostToDevice));

  Dense A_d = A;
  Dense B_d = B;
  Dense C_d = C;

  A_d.values = A_values_d;
  B_d.values = B_values_d;
  C_d.values = C_values_d;

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaStream_t stream = NULL;
  CHECK_CUDA(cudaStreamCreate(&stream))

  cudaDeviceSynchronize();

  float total_time = 0;
  for (int iter = 0; iter < ITERATIONS + 1; iter++) {
    CHECK_CUDA(cudaMemset(C_values_d, 0, C.rows * C.cols * sizeof(float)));
    cudaDeviceSynchronize();

    kernel(stream, start, stop, A_d, B_d, C_d);
    cudaEventSynchronize(stop);
    CHECK_CUDA(cudaGetLastError());

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    if (iter >= 1) total_time += milliseconds; // Skip a warm up
  }

  std::cout << name << " took " << total_time / ITERATIONS << "ms (avg)" << std::endl;

  CHECK_CUDA(cudaMemcpy(C.values, C_values_d, C.rows * C.cols * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(A_values_d))
  CHECK_CUDA(cudaFree(B_values_d))
  CHECK_CUDA(cudaFree(C_values_d))

  return 0;
}

int run_kernel(const CSR<float>& A, const Dense& B, Dense& C, const std::string& name,
               int(*kernel)(cudaStream_t& stream, cudaEvent_t& start, cudaEvent_t& stop, const CSR<float>& A_h, const CSR<float>& A, const Dense& B, Dense& C)) {
  float *A_values_d, *B_values_d, *C_values_d;
  int *A_row_offsets_d, *A_col_indices_d;

  CHECK_CUDA(cudaMalloc(&A_values_d, A.nnz * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&A_row_offsets_d, (A.rows + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&A_col_indices_d, A.nnz * sizeof(int)));

  CHECK_CUDA(cudaMalloc(&B_values_d, B.rows * B.cols * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&C_values_d, C.rows * C.cols * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(A_values_d, A.values, A.nnz * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(A_row_offsets_d, A.row_offsets, (A.rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(A_col_indices_d, A.col_indices, A.nnz * sizeof(int), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpy(B_values_d, B.values, B.rows * B.cols * sizeof(float), cudaMemcpyHostToDevice));

  CSR<float> A_d = A;
  Dense B_d = B;
  Dense C_d = C;

  A_d.values = A_values_d;
  A_d.row_offsets = A_row_offsets_d;
  A_d.col_indices = A_col_indices_d;
  B_d.values = B_values_d;
  C_d.values = C_values_d;

  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaStream_t stream = NULL;
  CHECK_CUDA(cudaStreamCreate(&stream))

  cudaDeviceSynchronize();

  float total_time = 0;
  for (int iter = 0; iter < ITERATIONS + 1; iter++) {
    CHECK_CUDA(cudaMemset(C_values_d, 0, C.rows * C.cols * sizeof(float)));
    cudaDeviceSynchronize();

    kernel(stream, start, stop, A, A_d, B_d, C_d);
    cudaEventSynchronize(stop);
    CHECK_CUDA(cudaGetLastError());

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    if (iter >= 1) total_time += milliseconds; // Skip a warm up
  }

  std::cout << name << " took " << total_time / ITERATIONS << "ms (avg)" << std::endl;

  CHECK_CUDA(cudaMemcpy(C.values, C_values_d, C.rows * C.cols * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(A_values_d));
  CHECK_CUDA(cudaFree(A_row_offsets_d));
  CHECK_CUDA(cudaFree(A_col_indices_d));

  CHECK_CUDA(cudaFree(B_values_d))
  CHECK_CUDA(cudaFree(C_values_d))

  return 0;
}

int run_kernel(const std::vector<CodeletMultiply::Block> &blocks, const CSR<float>& A, const Dense& B, Dense& C, const std::string& name,
               int(*kernel)(cudaStream_t& stream, cudaEvent_t& start, cudaEvent_t& stop, const CodeletMultiply::Block * blocks, size_t num_blocks, const CSR<float>& A_h, const CSR<float>& A, const Dense& B, Dense& C)) {
  float *A_values_d, *B_values_d, *C_values_d;
  int *A_row_offsets_d, *A_col_indices_d;
  std::vector<CodeletMultiply::Block> blocks_d_temp(blocks);
  CodeletMultiply::Block* blocks_d;

  CHECK_CUDA(cudaMalloc(&A_values_d, A.nnz * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&A_row_offsets_d, (A.rows + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&A_col_indices_d, A.nnz * sizeof(int)));

  for (int i = 0; i < blocks.size(); i++) {
    CHECK_CUDA(cudaMalloc(&blocks_d_temp[i].rows, blocks_d_temp[i].num_rows * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(blocks_d_temp[i].rows, blocks[i].rows, blocks_d_temp[i].num_rows * sizeof(int), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMalloc(&blocks_d_temp[i].col_pattern, blocks_d_temp[i].col_pattern_len * sizeof(int)));
    CHECK_CUDA(cudaMemcpy(blocks_d_temp[i].col_pattern, blocks[i].col_pattern, blocks_d_temp[i].col_pattern_len * sizeof(int), cudaMemcpyHostToDevice));

    CHECK_CUDA(cudaMalloc(&blocks_d_temp[i].row_segment_values, blocks_d_temp[i].num_rows * blocks_d_temp[i].col_pattern_len * sizeof(float)));
    CHECK_CUDA(cudaMemcpy(blocks_d_temp[i].row_segment_values, blocks[i].row_segment_values, blocks_d_temp[i].num_rows * blocks_d_temp[i].col_pattern_len * sizeof(float), cudaMemcpyHostToDevice));
  }

  CHECK_CUDA(cudaMalloc(&blocks_d, blocks_d_temp.size() * sizeof(CodeletMultiply::Block)));
  CHECK_CUDA(cudaMemcpy(blocks_d, blocks_d_temp.data(), blocks_d_temp.size() * sizeof(CodeletMultiply::Block), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&B_values_d, B.rows * B.cols * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&C_values_d, C.rows * C.cols * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(A_values_d, A.values, A.nnz * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(A_row_offsets_d, A.row_offsets, (A.rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(A_col_indices_d, A.col_indices, A.nnz * sizeof(int), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpy(B_values_d, B.values, B.rows * B.cols * sizeof(float), cudaMemcpyHostToDevice));

  CSR<float> A_d = A;
  Dense B_d = B;
  Dense C_d = C;

  A_d.values = A_values_d;
  A_d.row_offsets = A_row_offsets_d;
  A_d.col_indices = A_col_indices_d;
  B_d.values = B_values_d;
  C_d.values = C_values_d;

  cudaDeviceSynchronize();

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaStream_t stream = NULL;
  CHECK_CUDA(cudaStreamCreate(&stream))

  cudaDeviceSynchronize();

  float total_time = 0;
  for (int iter = 0; iter < ITERATIONS + 1; iter++) {
    CHECK_CUDA(cudaMemset(C_values_d, 0, C.rows * C.cols * sizeof(float)));
    cudaDeviceSynchronize();

    kernel(stream, start, stop, blocks_d, blocks.size(), A, A_d, B_d, C_d);
    cudaEventSynchronize(stop);
    CHECK_CUDA(cudaGetLastError());

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    if (iter >= 1) total_time += milliseconds; // Skip a warm up
  }

  std::cout << name << " took " << total_time / ITERATIONS << "ms (avg)" << std::endl;

  CHECK_CUDA(cudaMemcpy(C.values, C_values_d, C.rows * C.cols * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(A_values_d));
  CHECK_CUDA(cudaFree(A_row_offsets_d));
  CHECK_CUDA(cudaFree(A_col_indices_d));

  for (int i = 0; i < blocks.size(); i++) {
    CHECK_CUDA(cudaFree(blocks_d_temp[i].rows));
    CHECK_CUDA(cudaFree(blocks_d_temp[i].col_pattern));
    CHECK_CUDA(cudaFree(blocks_d_temp[i].row_segment_values));
  }

  CHECK_CUDA(cudaFree(B_values_d));
  CHECK_CUDA(cudaFree(C_values_d));

  return 0;
}


typedef struct codelet {
  std::vector<int> row_offsets;
  std::vector<int> col_offsets;
} codelet_t;

static inline int partition_evenly(int size, int target) {
  int blocks = (size + target - 1) / target;
  return (size + blocks - 1) / blocks;
}

const std::vector<CodeletMultiply::Block> gen_blocks(const std::vector<codelet_t>& codelets) {
  std::vector<CodeletMultiply::Block> blocks;

  for (auto& codelet : codelets) {
    int cols_per_block = partition_evenly(codelet.col_offsets.size(), CodeletMultiply::MAX_COLS_PER_BLOCK);
    int rows_per_block = partition_evenly(codelet.row_offsets.size(), CodeletMultiply::MAX_ROWS_PER_BLOCK);

    for (int i = 0; i < codelet.row_offsets.size(); i+= rows_per_block) {
      for (int j = 0; j < codelet.col_offsets.size(); j+= cols_per_block) {
        int rows_in_block = std::min(codelet.row_offsets.size() - i, (size_t) rows_per_block);
        int cols_in_block = std::min(codelet.col_offsets.size() - i, (size_t) cols_per_block);

        CodeletMultiply::Block blk;
        blk.num_rows = rows_in_block;
        blk.col_pattern_len = cols_in_block;
        blk.rows = new int[rows_in_block];
        blk.col_pattern = new int[cols_in_block];
        blk.row_segment_values = new float[rows_in_block * cols_in_block];

        for (int ii = 0; ii < rows_in_block; ii++) { blk.rows[ii] = codelet.row_offsets[ii + i]; }
        for (int jj = 0; jj < cols_in_block; jj++) { blk.col_pattern[jj] = codelet.col_offsets[jj + j]; }

        for (int i = 0; i < rows_in_block * cols_in_block; i++) {
          blk.row_segment_values[i] = 1;
        }

        blocks.push_back(blk);
      }
    }
  }

  return blocks;
}

CSR<float> gen_csr(size_t m, size_t n, const std::vector<codelet_t>& codelets) {
  std::vector<std::vector<int>> rows(m);

  for (auto& codelet : codelets) {
    for (auto& row_offset : codelet.row_offsets) {
      auto& row = rows[row_offset];
      if (codelet.col_offsets.size() > 0) {
        auto end_of_existing = row.size();

        row.insert(row.end(), codelet.col_offsets.begin(), codelet.col_offsets.end());
        std::inplace_merge(row.begin(), row.begin() + end_of_existing, row.end());
      }
    }
  }

  for (auto& row : rows) {
    auto last = std::unique(row.begin(), row.end());
    row.erase(last, row.end());
  }

  int nnz = 0; for (auto& row : rows) nnz += row.size();

  CSR<float> csr(m, n, nnz);
  int curr_offset = 0;
  csr.row_offsets[0] = 0;
  int i = 0;
  for (auto& row : rows) {
    std::copy( row.begin(), row.end(), &csr.col_indices[curr_offset]);
    curr_offset += row.size();
    csr.row_offsets[++i] = curr_offset;
  }

  for (int i = 0; i < nnz; i++) { csr.values[i] = 1; }

  return csr;
}

Dense csr_to_dense(const CSR<float> &csr) {
  float * out = new float[csr.rows * csr.cols];

  for (int i = 0; i < csr.rows; i ++) {
    for (int p = csr.row_offsets[i]; p < csr.row_offsets[i+1]; p++) {
      out[i * csr.cols + csr.col_indices[p]] = csr.values[p];
    }
  }

  return Dense(csr.rows, csr.cols, out);
}

void print_csr(const CSR<float> &csr) {
  for (int i = 0; i < csr.rows; i ++) {
    std::cout << i << ": ";
    for (int p = csr.row_offsets[i]; p < csr.row_offsets[i+1]; p++) {
      std::cout << csr.col_indices[p] << " ";
    }
    std::cout << std::endl;
  }
}

void print_dense(const Dense &dense) {
  for (int i = 0; i < dense.rows; i++) {
    for (int j = 0; j < dense.cols; j++) {
      std::cout << " " << dense.coeff(i, j);
    }
    std::cout << std::endl;
  }
}

void compare_dense(const Dense &A, const Dense &B) {
  assert(A.rows == A.rows && B.cols == B.cols);
  int total_errors = 0;
  for (int i = 0; i < A.rows; i++) {
    for (int j = 0; j < A.cols; j++) {
      if (A.coeff(i, j) != B.coeff(i, j)) {
        total_errors++;
        if (total_errors < 10) {
          printf("[ERROR] Mismatch at (%d, %d) %f != %f\n", i, j, A.coeff(i, j), B.coeff(i, j));
        }
      }
    }
  }
  printf("[ERROR] Total Mismatch %d\n", total_errors);
}

std::vector<codelet_t> gen_checkerboard(int m, int n, int stride) {

  std::vector<codelet_t> checkerboard_codelets(2);

  checkerboard_codelets[0].col_offsets.reserve((n + 2) / stride);
  checkerboard_codelets[0].row_offsets.reserve((m + 2) / stride);
  checkerboard_codelets[1].col_offsets.reserve((n + 2) / stride);
  checkerboard_codelets[1].row_offsets.reserve((m + 2) / stride);

  for (int i = 0; i < n; i += stride) { checkerboard_codelets[0].col_offsets.push_back(i); }
  for (int i = 0; i < m; i += stride) { checkerboard_codelets[0].row_offsets.push_back(i); }

  for (int i = 1; i < n; i += stride) { checkerboard_codelets[1].col_offsets.push_back(i); }
  for (int i = 1; i < m; i += stride) { checkerboard_codelets[1].row_offsets.push_back(i); }

  return std::move(checkerboard_codelets);
}

int main() {
  const int batch_size = 256;
  const int A_rows = 1024;
  const int A_cols = 2048;

  std::cout << "Generating pattern ..." << std::endl;
  auto codelets = gen_checkerboard(A_rows, A_cols, 4);
  std::cout << "Constructing CSR matrix ..." << std::endl;
  CSR<float> csr = gen_csr(A_rows, A_cols, codelets);
  std::cout << "Sparsity " << (1.f - csr.nnz / float(csr.rows * csr.cols)) * 100 << "%" << std::endl;
  std::cout << "Generating dense version ..." << std::endl;
  auto A = csr_to_dense(csr);

  auto blocks = gen_blocks(codelets);

  Dense B(csr.cols, batch_size, 5.f);
  Dense C(csr.rows, batch_size, 0.f);
  Dense C_golden(csr.rows, batch_size, 0.f);

  std::cout << "Running kernels ..." << std::endl;


  //
  //  Run Kernels
  //


  run_kernel(A, B, C_golden, "cublas", cublas_multiply);

  run_kernel(csr, B, C, "sgk", sgk_multiply);
  compare_dense(C_golden, C);
  C.fill(0);
  run_kernel(blocks, csr, B, C, "codelets", CodeletMultiply::codelet_multiply);
  compare_dense(C_golden, C);

  delete A.values;
}


//  for (int i = 0; i < csr.rows; i++) {
//    for (int p = csr.row_offsets[i]; p < csr.row_offsets[i+1]; p++) {
//      for (int j = 0; j < B.cols; j++) {
//        C_golden.values[i * C_golden.cols + j] += csr.values[p] * B.coeff(csr.col_indices[p], j);
//      }
//    }
//  }

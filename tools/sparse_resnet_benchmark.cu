//
// Created by lwilkinson on 12/13/21.
//

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
#include "common/utils/csv_log_io.h"
#include "sputnik/sputnik.h"

#include "dense_matrix.cuh"
#include "codlet_multiply.cuh"
#include "run_kernel.cuh"

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

  CHECK_CUBLAS(cublasDestroy_v2(handle));
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

typedef struct codelet {
    std::vector<int> row_offsets;
    std::vector<int> col_offsets;
} codelet_t;

static inline int partition_evenly(int size, int target) {
  int blocks = (size + target - 1) / target;
  return (size + blocks - 1) / blocks;
}

const std::vector<CodeletMultiply::Block> gen_blocks(const std::vector<codelet_t>& codelets,
                                                     size_t rows,
                                                     size_t cols,
                                                     size_t batch) {
  std::vector<CodeletMultiply::Block> blocks;

  int curr_num_blks_in_batch = 0;
  CodeletMultiply::Block curr_batch;
  for (auto& codelet : codelets) {
    int cols_per_block = partition_evenly(codelet.col_offsets.size(), cols);
    int rows_per_block = partition_evenly(codelet.row_offsets.size(), rows);

    if (rows_per_block != rows) exit(116);
    if (cols_per_block != cols) exit(116);

    for (int i = 0; i < codelet.row_offsets.size(); i+= rows_per_block) {
      for (int j = 0; j < codelet.col_offsets.size(); j+= cols_per_block) {
        int rows_in_block = std::min(codelet.row_offsets.size() - i, (size_t) rows_per_block);
        int cols_in_block = std::min(codelet.col_offsets.size() - j, (size_t) cols_per_block);

        if (rows_in_block != rows)
          exit(117);
        if (cols_in_block != cols)
          exit(117);

        if (curr_num_blks_in_batch == 0) {
          curr_batch.rows = new int[rows_in_block * batch];
          curr_batch.col_pattern = new int[cols_in_block * batch];
          curr_batch.row_segment_values = new float[rows_in_block * cols_in_block * batch];

          // Cheat: pre-init to 1
          for (int i = 0; i < rows_in_block * cols_in_block * batch; i++) {
            curr_batch.row_segment_values[i] = 1;
          }

          // Cheat init once, not used
          curr_batch.batch_size = batch;
          curr_batch.num_rows = rows_in_block;
          curr_batch.col_pattern_len = cols_in_block;
        }

        int row_offset = curr_num_blks_in_batch * rows_in_block;
        int col_offset = curr_num_blks_in_batch * cols_in_block;
        for (int ii = 0; ii < rows_in_block; ii++) { curr_batch.rows[row_offset + ii] = codelet.row_offsets[i + ii]; }
        for (int jj = 0; jj < cols_in_block; jj++) { curr_batch.col_pattern[col_offset + jj] = codelet.col_offsets[j + jj]; }

        curr_num_blks_in_batch++;
        if (curr_num_blks_in_batch == batch) {
          curr_num_blks_in_batch = 0;
          blocks.push_back(curr_batch);
        }
      }
    }
  }


  std::cout << "num blocks = " << blocks.size() << std::endl;

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

  checkerboard_codelets[0].col_offsets.reserve((n + stride - 1) / stride);
  checkerboard_codelets[0].row_offsets.reserve((m + stride - 1) / stride);
  checkerboard_codelets[1].col_offsets.reserve((n + stride - 1) / stride);
  checkerboard_codelets[1].row_offsets.reserve((m + stride - 1) / stride);

  for (int i = 0; i < n; i += stride) { checkerboard_codelets[0].col_offsets.push_back(i); }
  for (int i = 0; i < m; i += stride) { checkerboard_codelets[0].row_offsets.push_back(i); }

  for (int i = 1; i < n; i += stride) { checkerboard_codelets[1].col_offsets.push_back(i); }
  for (int i = 1; i < m; i += stride) { checkerboard_codelets[1].row_offsets.push_back(i); }

  return std::move(checkerboard_codelets);
}


CSR<float> read_smtx(const std::string &path) {
  std::ifstream file;
  file.open(path, std::ios_base::in);
  if (!file.is_open()) {
    std::cout << "File could not be found..." << std::endl;
    exit(1);
  }

  std::string line;
  int i;

  std::getline(file, line);
  std::replace(line.begin(), line.end(), ',', ' ');
  std::istringstream first_line(line);

  int rows, cols, nnz;
  first_line >> rows;
  first_line >> cols;
  first_line >> nnz;

  CSR<float> csr(rows, cols, nnz);

  for (int i = 0; i < csr.rows + 1; i++) { file >> csr.row_offsets[i]; }
  // Go to next line
  char next;
  while (file.get(next)) { if (next == '\n') break; }

  // Read in col_indices
  for (int i = 0; i < csr.nnz; i++) { file >> csr.col_indices[i]; }
  for (i = 0; i < csr.nnz; i++) { csr.values[i] = 1.0f; }

  return std::move(csr);
}

CSR<float> read_coo_blocked(const std::string &path, std::vector<CodeletMultiply::Block> &blocks, int& nnz_in_blocks, int& total_nnz) {
  std::ifstream file;
  file.open(path, std::ios_base::in);
  if (!file.is_open()) { std::cout << "File could not be found..." << std::endl; exit(1); }

  std::string line;
  int i;

  std::getline(file, line);
  std::istringstream first_line(line);

  int rows, cols;
  std::string token;
  first_line >> token;
  first_line >> rows;
  first_line >> cols;
  first_line >> nnz_in_blocks;
  first_line >> total_nnz;

  std::cout << token << std::endl;
  if (token != "%csr") { std::cout << "csr token mismatch" << std::endl; exit(1); }
  CSR<float> csr(rows, cols, nnz_in_blocks);
  for (int i = 0; i < csr.rows + 1; i++) { file >> csr.row_offsets[i]; }

  // Go to next line
  char next; while (file.get(next)) { if (next == '\n') break; }

  // Read in col_indices
  for (int i = 0; i < csr.nnz; i++) { file >> csr.col_indices[i]; }
  for (i = 0; i < csr.nnz; i++) { csr.values[i] = 1.0f; }

  while (file.get(next)) { if (next == '\n') break; } // Go to next line

  int count = 0;
  blocks.reserve(4096);
  while(!std::getline(file, line).eof()) {
    std::istringstream first_block_line(line);

    count++;

    first_block_line >> token;
    first_block_line >> rows;
    first_block_line >> cols;

    if (token != "%block") {
      std::cout << "block token mismatch " << token << " " << rows << " " << cols << std::endl;
      exit(1);
    }

    CodeletMultiply::Block block;
    block.num_rows = rows;
    block.col_pattern_len = cols;
    block.batch_size = 1;

    block.rows = new int[rows];
    block.col_pattern = new int[cols];
    block.row_segment_values = new float[rows * cols];

    for (int i = 0; i < rows; i++) { file >> block.rows[i]; }

    while (file.get(next)) { if (next == '\n') break; } // Go to next line
    for (int j = 0; j < cols; j++) { file >> block.col_pattern[j]; }

    for (int i = 0; i < rows; i++) {
      while (file.get(next)) { if (next == '\n') break; } // Go to next line
      for (int j = 0; j < cols; j++) {
        file >> block.row_segment_values[i * cols + j];
      }
    }

    while (file.get(next)) { if (next == '\n') break; } // Go to next line
    blocks.push_back(block);
  }

  return std::move(csr);
}

int main() {
  const std::vector<int> batch_sizes = { 64, 128, 256, 512, 768, 1024 };
  const int A_rows = 512;
  const int A_cols = 512;
  const int B_cols = 128;

  struct file_to_test {
      string file;
      string name;
      string sparsity;
  };

  int nnz_in_blocks, total_nnz;

  std::vector<struct file_to_test> files_to_test = {
    {"../matrices/rn50/magnitude_pruning/0.7/bottleneck_1_block_group3_2_1", "bn_1_blk_3_2_1", "0.7"},
    {"../matrices/rn50/magnitude_pruning/0.8/bottleneck_1_block_group3_2_1", "bn_1_blk_3_2_1", "0.8"},
    {"../matrices/rn50/magnitude_pruning/0.7/bottleneck_1_block_group_projection_block_group4", "bn_1_blk_proj_4", "0.7"},
    {"../matrices/rn50/magnitude_pruning/0.8/bottleneck_1_block_group_projection_block_group4", "bn_1_blk_proj_4", "0.8"},
    {"../matrices/rn50/magnitude_pruning/0.7/final_dense", "final_dense", "0.7" },
    {"../matrices/rn50/magnitude_pruning/0.8/final_dense", "final_dense", "0.8" },
    {"../matrices/rn50/magnitude_pruning/0.7/initial_conv", "initial_conv", "0.7" },
    {"../matrices/rn50/magnitude_pruning/0.8/initial_conv", "initial_conv", "0.8" },
    {"../matrices/rn50/magnitude_pruning/0.7/bottleneck_3_block_group2_1_1", "bottleneck_3_block_group2_1_1", "0.7" },
    {"../matrices/rn50/magnitude_pruning/0.8/bottleneck_3_block_group2_1_1", "bottleneck_3_block_group2_1_1", "0.8" }
  };

  for (auto& file : files_to_test) {
    std::cout << file.name << " " << file.sparsity << std::endl;

    std::vector<CodeletMultiply::Block> blocks;
    auto csr = read_coo_blocked(file.file + ".coo_blocked.txt", blocks, nnz_in_blocks, total_nnz);
    auto csr_full = read_smtx(file.file + ".smtx");

    test_harness::csv_row_t csv_row;
    test_harness::csv_row_insert(csv_row, "file", file.file);
    test_harness::csv_row_insert(csv_row, "name", file.name);
    test_harness::csv_row_insert(csv_row, "sparsity", file.sparsity);
    test_harness::csv_row_insert(csv_row, "nnz_in_blocks", nnz_in_blocks);
    test_harness::csv_row_insert(csv_row, "total_nnz", total_nnz);
    test_harness::csv_row_insert(csv_row, "num_blocks", blocks.size());

    Dense B(csr.cols, B_cols, 1.f);
    Dense C(csr.rows, B_cols, 0.f);

    C.fill(0);
    run_kernel(blocks, csr, B, C, "codelets_8x32x1", csv_row,
               CodeletMultiply::codelet_8x32x1::codelet_multiply);

    C.fill(0);
    run_kernel(csr, B, C, "sgk_part", csv_row, sgk_multiply);


    C.fill(0);
    run_kernel(csr_full, B, C, "sgk_full", csv_row, sgk_multiply);

    cudaDeviceReset();
    cudaDeviceSynchronize();

    test_harness::write_csv_row("../output/benchmark_codelets.csv", csv_row);
  }

  return 0;
}
//
// Created by lwilkinson on 11/4/21.
//

#include <math.h>
#include <iostream>
#include "cuda_runtime.h"
#include "cupti_profiler.h"
#include <stdlib.h>
#include <iostream>
#include <vector>
#include <algorithm>
#include <array>
#include "common/utils/matrix_utils.h"
#include "sputnik/sputnik.h"


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

#define CHECK_CUSPARSE(func)                                                    \
    {                                                                           \
        cusparseStatus_t status = (func);                                       \
        if (status != CUSPARSE_STATUS_SUCCESS)                                  \
        {                                                                       \
            printf("CUSPARSE API failed at %s:%d with error: %s (%d)\n",        \
                   __FILE__, __LINE__, cusparseGetErrorString(status), status); \
            return EXIT_FAILURE;                                                \
        }                                                                       \
    }

using namespace std;

typedef struct Dense {
    int rows;
    int cols;
    float * values;

    Dense(int m, int n, float fill): rows(m), cols(n) {
      values = new float[rows * cols];
      for (int i = 0; i < rows * cols; i++) values[i] = fill;
    }

    Dense(int m, int n, float * values): rows(m), cols(n), values(values) {}

    float coeff(int i, int j) const {
      return values[i * cols + j];
    }

} Dense;

static const int TILE_DIM = 32;

// Taken from: https://stackoverflow.com/a/18856054
__global__ void _dense_multiply(Dense A, Dense B, Dense C)
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

void dense_multiply(const Dense& A, const Dense& B, const Dense& C) {
  dim3 block_dim(TILE_DIM,TILE_DIM);
  dim3 grid_dim((C.cols + (TILE_DIM-1))/TILE_DIM,(C.rows + (TILE_DIM-1))/TILE_DIM);
  _dense_multiply<<<grid_dim, block_dim>>>(A, B, C);
}

typedef struct codelet {
    std::vector<int> row_offsets;
    std::vector<int> col_offsets;
} codelet_t;

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
      out[i * csr.rows + csr.col_indices[p]] = csr.values[p];
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

static float _test_sgk_spmm(
        int m, int k, int n, int nonzeros,
        float *h_values, int *h_row_indices, int *h_row_offsets, int *h_col_indices,
        float *h_dense_matrix
) {
  typedef std::chrono::steady_clock Clock;
  typedef std::chrono::nanoseconds nanoseconds;

  int *row_indices, *row_offsets, *column_indices;
  float *values, *dense_matrix, *output_matrix;

  //allocate A
  CHECK_CUDA(cudaMalloc((void **)&values, sizeof(float) * nonzeros))
  CHECK_CUDA(cudaMalloc((void **)&row_indices, sizeof(int) * m))
  CHECK_CUDA(cudaMalloc((void **)&row_offsets, sizeof(int) * (m + 1)))
  CHECK_CUDA(cudaMalloc((void **)&column_indices, sizeof(int) * nonzeros))
  // allocate B
  CHECK_CUDA(cudaMalloc((void **)&dense_matrix, sizeof(float) * n * k))
  // allocate C
  CHECK_CUDA(cudaMalloc((void **)&output_matrix, sizeof(float) * n * m))

  //to device mtx A
  CHECK_CUDA(cudaMemcpy(row_indices, h_row_indices, sizeof(int) * m, cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(row_offsets, h_row_offsets, sizeof(int) * (m+1), cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(column_indices, h_col_indices, sizeof(int) * nonzeros, cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(values, h_values, sizeof(float) * nonzeros, cudaMemcpyHostToDevice))
  //to device mtx B
  CHECK_CUDA(cudaMemcpy(dense_matrix, h_dense_matrix, sizeof(float) * n * k, cudaMemcpyHostToDevice))

  cudaStream_t handle = NULL;
  CHECK_CUDA(cudaStreamCreate(&handle))

  float* bias = nullptr;

  cudaDeviceSynchronize();
  Clock::time_point start = Clock::now();
  CHECK_CUDA(sputnik::CudaSpmmBiasRelu(m, k, n, nonzeros, row_indices, values,
                                       row_offsets, column_indices, dense_matrix,
                                       bias, output_matrix, handle))
  cudaDeviceSynchronize();
  Clock::time_point end = Clock::now();

  CHECK_CUDA(cudaFree(row_indices))
  CHECK_CUDA(cudaFree(row_offsets))
  CHECK_CUDA(cudaFree(column_indices))
  CHECK_CUDA(cudaFree(values))
  CHECK_CUDA(cudaFree(dense_matrix))
  CHECK_CUDA(cudaFree(output_matrix))

  CHECK_CUDA(cudaStreamDestroy(handle))
  nanoseconds ms = std::chrono::duration_cast<nanoseconds>(end - start);
  return ms.count();
}


template<typename AMatrix>
int run_kernel(const AMatrix& A, const Dense& B, const Dense& C, const std::string& name,
                void(*kernel)(const AMatrix& A, const Dense& B, const Dense& C)) {}

template<>
int run_kernel(const Dense& A, const Dense& B, const Dense& C, const std::string& name,
                void(*kernel)(const Dense& A, const Dense& B, const Dense& C)) {
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

  cudaEventRecord(start);
  kernel(A_d, B_d, C_d);
  CHECK_CUDA(cudaGetLastError());
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  std::cout << name << " took " << milliseconds << "ms" << std::endl;

  CHECK_CUDA(cudaMemcpy(C.values, C_values_d, C.rows * C.cols * sizeof(float), cudaMemcpyDeviceToHost));

  return 0;
}


std::vector<codelet_t> gen_checkerboard(int m, int n) {

  std::vector<codelet_t> checkerboard_codelets(2);

  checkerboard_codelets[0].col_offsets.reserve((n + 2) / 3);
  checkerboard_codelets[0].row_offsets.reserve((m + 2) / 3);
  checkerboard_codelets[1].col_offsets.reserve((n + 2) / 3);
  checkerboard_codelets[1].row_offsets.reserve((m + 2) / 3);

  for (int i = 0; i < n; i += 3) { checkerboard_codelets[0].col_offsets.push_back(i); }
  for (int i = 0; i < m; i += 3) { checkerboard_codelets[0].row_offsets.push_back(i); }

  for (int i = 1; i < n; i += 3) { checkerboard_codelets[1].col_offsets.push_back(i); }
  for (int i = 1; i < m; i += 3) { checkerboard_codelets[1].row_offsets.push_back(i); }

  return std::move(checkerboard_codelets);
}



int main() {
  const int batch_size = 1024;

  auto codelets = gen_checkerboard(256, 256);
  CSR<float> csr = gen_csr(256, 256, codelets);

  auto A = csr_to_dense(csr);

  Dense B(csr.cols, batch_size, 5.f);
  Dense C(csr.rows, batch_size, 0.f);

  run_kernel(A, B, C, "tiled_dense", dense_multiply);

  delete A.values;

  Dense C_golden(csr.rows, batch_size, 0.f);
  for (int i = 0; i < csr.rows; i++) {
    for (int p = csr.row_offsets[i]; p < csr.row_offsets[i+1]; p++) {
      for (int j = 0; j < B.cols; j++) {
        C_golden.values[i * C_golden.cols + j] += csr.values[p] * B.coeff(csr.col_indices[p], j);
      }
    }
  }

  for (int i = 0; i < C.rows; i++) {
    for (int j = 0; j < C.cols; j++) {
      if (C_golden.coeff(i, j) != C.coeff(i, j)) {
        printf("[ERROR] Mismatch at (%d, %d)\n", i, j);
      }
    }
  }



//  auto A = new float[N][N];
//  auto B = new float[N][N];
//  auto C = new float[N][N];
//
//  for (int i = 0; i < N; i++) {
//    for (int j = i % 2; j < N; j++) {
//      A[i][j] = 1.0;
//    }
//  }
//
//  for (int i = 0; i < N; i++) {
//    for (int j = 0; j < N; j++) {
//      B[i][j] = 3.0;
//    }
//  }
//
//  matrixMultiplication((float *) A, (float *) B, (float *) C, N);
//
//  std::cout << "Printing" << std::endl;
//  for (int i = 0; i < N; i++) {
//    for (int j = 0; j < N; j++) {
//      std::cout << " " << C[i][j];
//    }
//    std::cout << std::endl;
//  }
}
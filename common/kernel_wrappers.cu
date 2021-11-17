//
// Created by lwilkinson on 10/26/21.
//

#include "kernel_wrappers.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <chrono>
#include <numeric>
#include <vector>

// Kernels
#include <cublas_v2.h>
#include <cusparse.h>
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

// #define DEBUG

/**
 * @brief This function time the cuSparse sparse-dense matrix multiplication: A @ B where A is of
 * shape m * k and B is of shape k * n
 * @return float : the time in millisecond
 */

float test_cusparse_gemm(int m, int k, int n, int A_nnz,
                         float *hA_csr_values, int *hA_csr_offsets, int *hA_csr_columns, float *hB_values) {
  typedef std::chrono::steady_clock Clock;
  typedef std::chrono::nanoseconds nanoseconds;

  float alpha = 1.0f;
  float beta = 0.0f;
  int ldb = k;
  int ldc = m;

  //device memory
  int *dA_csr_offsets, *dA_csr_columns;
  float *dA_csr_values, *dB_values, *dC_values;
  int A_num_rows = m;
  int A_num_cols = k;
  int B_num_rows = k;
  int B_num_cols = n;

  //allocate A
  CHECK_CUDA(cudaMalloc((void **)&dA_csr_offsets, (A_num_rows + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&dA_csr_columns, A_nnz * sizeof(int)));
  CHECK_CUDA(cudaMalloc((void **)&dA_csr_values, A_nnz * sizeof(float)));

  //allocate B
  CHECK_CUDA(cudaMalloc((void **)&dB_values, sizeof(float) * B_num_rows * B_num_cols));
  //allocate C
  CHECK_CUDA(cudaMalloc((void **)&dC_values, sizeof(float) * A_num_rows * B_num_cols));

  //to device mtx A
  CHECK_CUDA(cudaMemcpy(dA_csr_offsets, hA_csr_offsets, (A_num_rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dA_csr_columns, hA_csr_columns, A_nnz * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(dA_csr_values, hA_csr_values, A_nnz * sizeof(float), cudaMemcpyHostToDevice));
  //to device mtx B
  CHECK_CUDA(cudaMemcpy(dB_values, hB_values, (B_num_rows * B_num_cols) * sizeof(float), cudaMemcpyHostToDevice));
  //create the matrices
  cusparseHandle_t handle = NULL;
  cusparseSpMatDescr_t matA;
  cusparseDnMatDescr_t matB, matC;
  size_t bufferSize = 0;
  CHECK_CUSPARSE(cusparseCreateCsr(&matA, A_num_rows, A_num_cols, A_nnz,
                                   dA_csr_offsets, dA_csr_columns, dA_csr_values, CUSPARSE_INDEX_32I,
                                   CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F));

  CHECK_CUSPARSE(cusparseCreateDnMat(&matB, B_num_rows, B_num_cols, ldb, dB_values,
                                     CUDA_R_32F, CUSPARSE_ORDER_COL))
  CHECK_CUSPARSE(cusparseCreateDnMat(&matC, A_num_rows, B_num_cols, ldc, dC_values,
                                     CUDA_R_32F, CUSPARSE_ORDER_COL))
  CHECK_CUSPARSE(cusparseCreate(&handle))

  //SpGEMM
  cudaDeviceSynchronize();
  Clock::time_point start = Clock::now();
  CHECK_CUSPARSE(cusparseSpMM_bufferSize(
          handle,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          CUSPARSE_OPERATION_NON_TRANSPOSE,
          &alpha, matA, matB, &beta, matC, CUDA_R_32F,
          CUSPARSE_SPMM_ALG_DEFAULT, &bufferSize))

  void *dBuffer = NULL;
  CHECK_CUDA(cudaMalloc(&dBuffer, bufferSize));
  CHECK_CUSPARSE(cusparseSpMM(handle,
                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                              CUSPARSE_OPERATION_NON_TRANSPOSE,
                              &alpha, matA, matB, &beta, matC, CUDA_R_32F,
                              CUSPARSE_SPMM_ALG_DEFAULT, dBuffer))
  cudaDeviceSynchronize();
  Clock::time_point end = Clock::now();
  // destroy matrix/vector descriptors
  CHECK_CUSPARSE(cusparseDestroySpMat(matA))
  CHECK_CUSPARSE(cusparseDestroyDnMat(matB))
  CHECK_CUSPARSE(cusparseDestroyDnMat(matC))
  CHECK_CUSPARSE(cusparseDestroy(handle))

#ifdef DEBUG
  float *hC_values;
    hC_values = (float *)malloc(A_num_rows * B_num_cols * sizeof(float));
    //copy the result to the host
    CHECK_CUDA(cudaMemcpy(hC_values, dC_values, A_num_rows * B_num_cols * sizeof(float), cudaMemcpyDeviceToHost))
    //print out the result
    fprintf(stderr, "printing the multiplication result \n");
    for (int i = 0; i < A_num_rows * B_num_cols; i++)
    {
        fprintf(stderr, "%f\n", hC_values[i]);
    }
#endif
  //device memory free
  CHECK_CUDA(cudaFree(dBuffer))
  CHECK_CUDA(cudaFree(dA_csr_offsets))
  CHECK_CUDA(cudaFree(dA_csr_columns))
  CHECK_CUDA(cudaFree(dA_csr_values))
  CHECK_CUDA(cudaFree(dB_values))
  CHECK_CUDA(cudaFree(dC_values))
  //get the time
  nanoseconds ms = std::chrono::duration_cast<nanoseconds>(end - start);
  return ms.count();
}

float test_cublas_sgemm(int m, int k, int n, float *A, float *B)
{
  // float test_cublas_sgemm(int m, int n, int k, float * arr_A, float * arr_B) {
  //remember the mtx is col based!!!
  //init the variables
  typedef std::chrono::steady_clock Clock;
  typedef std::chrono::nanoseconds nanoseconds;
  float *d_A, *d_B, *d_C;
#ifdef DEBUG
  //define the output variable C
    float *C;
    C = (float *)malloc(sizeof(float) * m * n);
#endif

  //cuda code
  cublasHandle_t handle;
  if (cublasCreate(&handle) != CUBLAS_STATUS_SUCCESS)
  {
    fprintf(stderr, "fail handle");
  }

  CHECK_CUDA(cudaMalloc((void **)&d_A, sizeof(float) * m * k))
  CHECK_CUDA(cudaMalloc((void **)&d_B, sizeof(float) * n * k))
  CHECK_CUDA(cudaMalloc((void **)&d_C, sizeof(float) * m * n))

  CHECK_CUDA(cudaMemcpy(d_A, A, sizeof(float) * m * k, cudaMemcpyHostToDevice))
  CHECK_CUDA(cudaMemcpy(d_B, B, sizeof(float) * n * k, cudaMemcpyHostToDevice))

  const float a = 1.0, b = 0.0;
  cudaDeviceSynchronize();
  Clock::time_point start = Clock::now();

  cublasStatus_t ret = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &a, d_A, m, d_B, k, &b, d_C, m);
  cudaDeviceSynchronize();
  Clock::time_point end = Clock::now();

#ifdef DEBUG
  //copy the result back to host memory ofr latter printing
    CHECK_CUDA(cudaMemcpy(C, d_C, sizeof(float) * m * n, cudaMemcpyDeviceToHost))
#endif

  CHECK_CUDA(cudaFree(d_A))
  CHECK_CUDA(cudaFree(d_B))
  CHECK_CUDA(cudaFree(d_C))

  cublasDestroy(handle);

  nanoseconds ms = std::chrono::duration_cast<nanoseconds>(end - start);

#ifdef DEBUG
  fprintf(stderr, "printing the multiplication result col by col, matrix is %d X %d\n\n", m, n);
    for (int i = 0; i < m * n; i++)
    {
        fprintf(stderr, "%f \n", C[i]);
    }
#endif
  return ms.count();
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


float test_sgk_spmm(int m, int k, int n, int nonzeros,
                    float *A_csr_values, int *A_csr_offsets, int *A_csr_columns,
                    float *B_values
) {
  // Sort rows - Copied from sputnik
  // Create our unsorted row indices.
  std::vector<int> swizzle_staging(m);
  std::iota(swizzle_staging.begin(), swizzle_staging.end(), 0);

  // Argsort the row indices based on their length.
  std::sort(swizzle_staging.begin(), swizzle_staging.end(),
            [&A_csr_offsets](int idx_a, int idx_b) {
                int length_a = A_csr_offsets[idx_a + 1] - A_csr_offsets[idx_a];
                int length_b = A_csr_offsets[idx_b + 1] - A_csr_offsets[idx_b];
                return length_a > length_b;
            });

  return _test_sgk_spmm(
          m, k, n, nonzeros,
          A_csr_values, swizzle_staging.data(), A_csr_offsets, A_csr_columns,
          B_values
  );
}

float test_sgk_spmm_custom_row_order(
        int m, int k, int n, int nonzeros,
        float *A_csr_values, int *A_csr_offsets, int *A_csr_columns, int *A_row_ordering,
        float *B_values
) {
  return _test_sgk_spmm(
          m, k, n, nonzeros,
          A_csr_values, A_row_ordering, A_csr_offsets, A_csr_columns,
          B_values
  );
}
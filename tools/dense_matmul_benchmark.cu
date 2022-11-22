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

// Taken from: https://stackoverflow.com/a/18856054
static const int TILE_DIM = 32;

int
cublas_multiply(cudaStream_t &stream, cudaEvent_t &start, cudaEvent_t &stop, const Dense &A, const Dense &B, Dense &C) {
    float alpha = 1.0, beta = 1.0;
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate_v2(&handle));

    // Set the math mode to disable cuBLAS to use Tensor Cores:
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));

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

int
cublas_multiply_ex(cudaStream_t &stream, cudaEvent_t &start, cudaEvent_t &stop, const Dense &A, const Dense &B, Dense &C) {
    float alpha = 1.0, beta = 1.0;
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate_v2(&handle));

    // Set the math mode to disable cuBLAS to use Tensor Cores:
    CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION));

    cudaEventRecord(start, stream);

    CHECK_CUBLAS(cublasGemmEx(handle,
                              CUBLAS_OP_N, CUBLAS_OP_N,
                              C.cols, C.rows, A.cols, /* m, n, k */
                              &alpha,
                              A.values, CUDA_R_32F, A.cols, /* *A, lda */
                              B.values, CUDA_R_32F, B.cols, /* *B, lda */
                              &beta,
                              C.values, CUDA_R_32F, C.cols,
                              CUBLAS_COMPUTE_32F,
                              CUBLAS_GEMM_DEFAULT))

    cudaEventRecord(stop, stream);

    CHECK_CUBLAS(cublasDestroy_v2(handle));
    return 0;
}

int
cublas_tensorcore_multiply(cudaStream_t &stream, cudaEvent_t &start, cudaEvent_t &stop, const Dense &A, const Dense &B, Dense &C) {
    float alpha = 1.0, beta = 1.0;
    cublasHandle_t handle;
    CHECK_CUBLAS(cublasCreate_v2(&handle));

    // Set the math mode to allow cuBLAS to use Tensor Cores:
    // CHECK_CUBLAS(cublasSetMathMode(handle, CUBLAS_TF32_TENSOR_OP_MATH));
    // Actually this is not needed - it is enabled by default on Ampere!

    cudaEventRecord(start, stream);

    CHECK_CUBLAS(cublasGemmEx(handle,
                             CUBLAS_OP_N, CUBLAS_OP_N,
                             C.cols, C.rows, A.cols, /* m, n, k */
                             &alpha,
                             A.values, CUDA_R_32F, A.cols, /* *A, lda */
                             B.values, CUDA_R_32F, B.cols, /* *B, lda */
                             &beta,
                             C.values, CUDA_R_32F, C.cols,
                              CUBLAS_COMPUTE_32F_FAST_16F,
                             //CUBLAS_COMPUTE_32F_FAST_TF32,
                             CUBLAS_GEMM_DEFAULT))

    cudaEventRecord(stop, stream);

    CHECK_CUBLAS(cublasDestroy_v2(handle));
    return 0;
}

// Taken from: https://stackoverflow.com/a/18856054
__global__ void _dense_multiply(const Dense &A, const Dense &B, Dense &C) {
    float CValue = 0;

    int Row = blockIdx.y * TILE_DIM + threadIdx.y;
    int Col = blockIdx.x * TILE_DIM + threadIdx.x;

    __shared__ float As[TILE_DIM][TILE_DIM];
    __shared__ float Bs[TILE_DIM][TILE_DIM];

    for (int k = 0; k < (A.cols + TILE_DIM - 1) / TILE_DIM; k++) {

        if (k * TILE_DIM + threadIdx.x < A.cols && Row < A.rows)
            As[threadIdx.y][threadIdx.x] = A.values[Row * A.cols + k * TILE_DIM + threadIdx.x];
        else
            As[threadIdx.y][threadIdx.x] = 0.0;

        if (k * TILE_DIM + threadIdx.y < B.rows && Col < B.cols)
            Bs[threadIdx.y][threadIdx.x] = B.values[(k * TILE_DIM + threadIdx.y) * B.cols + Col];
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

int dense_multiply(cudaStream_t &stream, cudaEvent_t &start, cudaEvent_t &stop, const Dense &A, const Dense &B, Dense &C) {
    dim3 block_dim(TILE_DIM, TILE_DIM);
    dim3 grid_dim((C.cols + (TILE_DIM - 1)) / TILE_DIM, (C.rows + (TILE_DIM - 1)) / TILE_DIM);

    cudaEventRecord(start, stream);
    _dense_multiply<<<grid_dim, block_dim, 0, stream>>>(A, B, C);
    CHECK_CUDA(cudaGetLastError());
    cudaEventRecord(stop, stream);

    return 0;
}


int run_dense_kernel(const Dense& A, const Dense& B, Dense C,
                     const std::string& name, test_harness::csv_row_t &csv_row,
                     int(*kernel)(cudaStream_t& stream, cudaEvent_t& start, cudaEvent_t& stop, const Dense& A, const Dense& B, Dense& C)
) {
    float *A_values_d, *B_values_d, *C_values_d;

    CHECK_CUDA(cudaMalloc(&A_values_d, A.rows * A.cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&B_values_d, B.rows * B.cols * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&C_values_d, C.rows * C.cols * sizeof(float)));

    CHECK_CUDA(cudaMemcpy(A_values_d, A.values, A.rows * A.cols * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(B_values_d, B.values, B.rows * B.cols * sizeof(float), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(C_values_d, C.values, C.rows * C.cols * sizeof(float), cudaMemcpyHostToDevice));

    Dense A_d = A;
    Dense B_d = B;
    Dense C_d = C;

    A_d.values = A_values_d;
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

        kernel(stream, start, stop, A_d, B_d, C_d);
        cudaEventSynchronize(stop);
        CHECK_CUDA(cudaGetLastError());

        float milliseconds = 0;
        cudaEventElapsedTime(&milliseconds, start, stop);

        if (iter >= 1) total_time += milliseconds; // Skip a warm up
    }

    std::cout << name << " took " << total_time / ITERATIONS << "ms (avg)" << std::endl;
    test_harness::csv_row_insert(csv_row, name, total_time / ITERATIONS);

    CHECK_CUDA(cudaMemcpy(C.values, C_values_d, C.rows * C.cols * sizeof(float), cudaMemcpyDeviceToHost));

    cudaDeviceSynchronize();

    CHECK_CUDA(cudaFree(A_values_d));
    CHECK_CUDA(cudaFree(B_values_d))
    CHECK_CUDA(cudaFree(C_values_d))

    cudaDeviceSynchronize();

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);

    cudaDeviceSynchronize();

    return 0;
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
    if (total_errors > 0) {
        printf("[ERROR] Total Mismatch %d\n", total_errors);
    }
}


int main() {
    const int SIZE = 8192;
    const int A_h = SIZE, A_w = SIZE;
    const int B_h = A_w, B_w = SIZE;
    const int C_h = A_h, C_w = B_w;

    Dense A(A_h, A_w, 0.0f);
    Dense B(B_h, B_w, 0.0f);
    Dense C(C_h, C_w, 0.0f);

    for (int i = 0; i < A_h * A_w; ++i) A.values[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    for (int i = 0; i < B_h * B_w; ++i) B.values[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    // for (int i = 0; i < C_h * C_w; ++i) C.values[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);

    test_harness::csv_row_t csv_row;

    C.fill(0.0f);
    run_dense_kernel(A, B, C, "cublas_multiply", csv_row, cublas_multiply);

    // Make a copy of C from cublas as our ground-truh to check for correctness!
    Dense C_gt(C.rows, C.cols, C.values);

    cudaDeviceReset();
    cudaDeviceSynchronize();

    C.fill(0.0f);
    run_dense_kernel(A, B, C, "cublas_multiply_ex", csv_row, cublas_multiply_ex);
    compare_dense(C, C_gt);
    cudaDeviceReset();
    cudaDeviceSynchronize();

    C.fill(0.0f);
    run_dense_kernel(A, B, C, "cublas_tensorcore_multiply", csv_row, cublas_tensorcore_multiply);
    compare_dense(C, C_gt);
    cudaDeviceReset();
    cudaDeviceSynchronize();

    C.fill(0.0f);
    run_dense_kernel(A, B, C, "cublas_tensorcore_multiply", csv_row, cublas_tensorcore_multiply);
    compare_dense(C, C_gt);
    cudaDeviceReset();
    cudaDeviceSynchronize();

    return 0;
}
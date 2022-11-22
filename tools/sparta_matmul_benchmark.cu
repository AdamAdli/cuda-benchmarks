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


int main() {
    const int SIZE = 8192;
    const int A_h = SIZE, A_w = SIZE;
    const int B_h = A_w, B_w = SIZE;
    const int C_h = A_h, C_w = B_w;

    /*
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
    */

    // TODO.
    return 0;
}
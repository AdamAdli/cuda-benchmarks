//
// Created by lwilkinson on 10/26/21.
//

#ifndef BENCHMARK_KERNEL_WRAPPERS_H
#define BENCHMARK_KERNEL_WRAPPERS_H


float test_cublas_sgemm(int m, int k, int n, float *A, float *B);
float test_cusparse_gemm(int m, int k, int n, int A_nnz,
                         float *A_csr_values, int *A_csr_offsets, int *A_csr_columns,
                         float *B_values);

float test_sgk_spmm(int m, int k, int n, int nonzeros,
                    float *A_csr_values, int *A_csr_offsets, int *A_csr_columns,
                    float *B_values);

float test_sgk_spmm_custom_row_order(
        int m, int k, int n, int nonzeros,
        float *A_csr_values, int *A_csr_offsets, int *A_csr_columns, int *A_row_ordering,
        float *B_values);

#endif //BENCHMARK_KERNEL_WRAPPERS_H

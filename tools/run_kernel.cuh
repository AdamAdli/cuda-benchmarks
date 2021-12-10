//
// Created by lwilkinson on 12/7/21.
//

#ifndef BENCHMARK_RUN_KERNEL_CUH
#define BENCHMARK_RUN_KERNEL_CUH

#include <vector>

#include "common/utils/cuda_utils.h"
#include "common/utils/csv_log_io.h"
#include "common/utils/matrix_utils.h"

#include "synthetic_codlets.cuh"
#include "codlet_multiply.cuh"

constexpr int ITERATIONS = 1;

int run_kernel(const Dense& A, const Dense& B, const Dense& C, const std::string& name, test_harness::csv_row_t &csv_row,
               int(*kernel)(cudaStream_t& stream, cudaEvent_t& start, cudaEvent_t& stop,
                       const Dense& A, const Dense& B, Dense& C));

int run_kernel(const CSR<float>& A, const Dense& B, Dense& C, const std::string& name, test_harness::csv_row_t &csv_row,
               int(*kernel)(cudaStream_t& stream, cudaEvent_t& start, cudaEvent_t& stop,
                       const CSR<float>& A_h, const CSR<float>& A, const Dense& B, Dense& C));

int run_kernel(const std::vector<CodeletMultiply::Block> &blocks, const CSR<float>& A, const Dense& B, Dense& C,
               const std::string& name, test_harness::csv_row_t &csv_row,
               int(*kernel)(cudaStream_t& stream, cudaEvent_t& start, cudaEvent_t& stop,
                       const CodeletMultiply::Block * blocks, size_t num_blocks,
                       const CSR<float>& A_h, const CSR<float>& A, const Dense& B, Dense& C));


#endif //BENCHMARK_RUN_KERNEL_CUH

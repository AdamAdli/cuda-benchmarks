//
// Created by lwilkinson on 12/7/21.
//

#include "run_kernel.cuh"


int run_kernel(const Dense& A, const Dense& B, const Dense& C, const std::string& name, test_harness::csv_row_t &csv_row,
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
  test_harness::csv_row_insert(csv_row, name, total_time / ITERATIONS);

  cudaDeviceSynchronize();

  CHECK_CUDA(cudaMemcpy(C.values, C_values_d, C.rows * C.cols * sizeof(float), cudaMemcpyDeviceToHost));

  CHECK_CUDA(cudaFree(A_values_d))
  CHECK_CUDA(cudaFree(B_values_d))
  CHECK_CUDA(cudaFree(C_values_d))

  cudaDeviceSynchronize();

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);

  cudaDeviceSynchronize();

  return 0;
}

int run_kernel(const CSR<float>& A, const Dense& B, Dense& C, const std::string& name, test_harness::csv_row_t &csv_row,
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
  test_harness::csv_row_insert(csv_row, name, total_time / ITERATIONS);

  CHECK_CUDA(cudaMemcpy(C.values, C_values_d, C.rows * C.cols * sizeof(float), cudaMemcpyDeviceToHost));

  cudaDeviceSynchronize();

  CHECK_CUDA(cudaFree(A_values_d));
  CHECK_CUDA(cudaFree(A_row_offsets_d));
  CHECK_CUDA(cudaFree(A_col_indices_d));

  CHECK_CUDA(cudaFree(B_values_d))
  CHECK_CUDA(cudaFree(C_values_d))

  cudaDeviceSynchronize();

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);

  cudaDeviceSynchronize();

  return 0;
}

int run_kernel(const std::vector<CodeletMultiply::Block> &blocks, const CSR<float>& A, const Dense& B, Dense& C, const std::string& name, test_harness::csv_row_t &csv_row,
               int(*kernel)(cudaStream_t& stream, cudaEvent_t& start, cudaEvent_t& stop, const CodeletMultiply::Block * blocks, size_t num_blocks, const CSR<float>& A_h, const CSR<float>& A, const Dense& B, Dense& C)) {
  float *A_values_d, *B_values_d, *C_values_d;
  int *A_row_offsets_d, *A_col_indices_d;
  std::vector<CodeletMultiply::Block> blocks_d_temp(blocks);
  CodeletMultiply::Block* blocks_d;

  CHECK_CUDA(cudaMalloc(&A_values_d, A.nnz * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&A_row_offsets_d, (A.rows + 1) * sizeof(int)));
  CHECK_CUDA(cudaMalloc(&A_col_indices_d, A.nnz * sizeof(int)));

  for (int i = 0; i < blocks.size(); i++) {
    size_t num_rows = blocks_d_temp[i].num_rows * sizeof(int) * blocks[0].batch_size;
    CHECK_CUDA(cudaMalloc(&blocks_d_temp[i].rows, num_rows));
    CHECK_CUDA(cudaMemcpy(blocks_d_temp[i].rows, blocks[i].rows, num_rows, cudaMemcpyHostToDevice));

    size_t col_pattern_len = blocks_d_temp[i].col_pattern_len * sizeof(int) * blocks[0].batch_size;
    CHECK_CUDA(cudaMalloc(&blocks_d_temp[i].col_pattern, col_pattern_len));
    CHECK_CUDA(cudaMemcpy(blocks_d_temp[i].col_pattern, blocks[i].col_pattern, col_pattern_len, cudaMemcpyHostToDevice));

    size_t values_len = blocks_d_temp[i].num_rows * blocks_d_temp[i].col_pattern_len * sizeof(float) * blocks[0].batch_size;
    CHECK_CUDA(cudaMalloc(&blocks_d_temp[i].row_segment_values, values_len));
    CHECK_CUDA(cudaMemcpy(blocks_d_temp[i].row_segment_values, blocks[i].row_segment_values, values_len, cudaMemcpyHostToDevice));
  }

  CHECK_CUDA(cudaMalloc(&blocks_d, blocks_d_temp.size() * sizeof(CodeletMultiply::Block)));
  CHECK_CUDA(cudaMemcpy(blocks_d, blocks_d_temp.data(), blocks_d_temp.size() * sizeof(CodeletMultiply::Block), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMalloc(&B_values_d, B.rows * B.cols * sizeof(float)));
  CHECK_CUDA(cudaMalloc(&C_values_d, C.rows * C.cols * sizeof(float)));

  CHECK_CUDA(cudaMemcpy(A_values_d, A.values, A.nnz * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(A_row_offsets_d, A.row_offsets, (A.rows + 1) * sizeof(int), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemcpy(A_col_indices_d, A.col_indices, A.nnz * sizeof(int), cudaMemcpyHostToDevice));

  CHECK_CUDA(cudaMemcpy(B_values_d, B.values, B.rows * B.cols * sizeof(float), cudaMemcpyHostToDevice));
  CHECK_CUDA(cudaMemset(C_values_d, 0, C.rows * C.cols * sizeof(float)));

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
    CHECK_CUDA(cudaGetLastError());
    cudaDeviceSynchronize();
    CHECK_CUDA(cudaGetLastError());
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
  CHECK_CUDA(cudaFree(A_row_offsets_d));
  CHECK_CUDA(cudaFree(A_col_indices_d));

  for (int i = 0; i < blocks.size(); i++) {
    CHECK_CUDA(cudaFree(blocks_d_temp[i].rows));
    CHECK_CUDA(cudaFree(blocks_d_temp[i].col_pattern));
    CHECK_CUDA(cudaFree(blocks_d_temp[i].row_segment_values));
  }

  CHECK_CUDA(cudaFree(B_values_d));
  CHECK_CUDA(cudaFree(C_values_d));

  cudaDeviceSynchronize();

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  cudaStreamDestroy(stream);

  cudaDeviceSynchronize();

  return 0;
}

//
// Created by lwilkinson on 10/25/21.
//

#include "common/utils/smtx_io.h"
#include "common/kernel_wrappers.h"

#include <stdio.h>
#include <iostream>

int main(int argc, char* argv[]) {
  printf("argc %d %s\n", argc, argv[0]);

  if (argc != 4) {
    printf("Usage: kernel_driver <kernel> <path to smtx file> <batch size>");
    return -1;
  }

  std::string kernel(argv[1]);
  std::string file_path(argv[2]);
  int batch_size = std::atoi(argv[3]);

  if (batch_size <= 0) {
    printf("Invalid batch size %d\n", batch_size);
    return -1;
  }

  std::cout << kernel << file_path << batch_size << std::endl;

  auto A = load_smtx(file_path);
  DenseMtx<float> B(A->cols, batch_size);

  printf("%d %d\n", A->rows, A->cols);
  if (kernel == "sgk") {
    auto time =  test_sgk_spmm(A->rows, A->cols, batch_size, A->nnz,
                        A->values, A->row_offsets, A->col_indices,
                        B.values);
    printf("sgk time %f\n", time);
  }

  return 0;
}
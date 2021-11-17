//
// Created by lwilkinson on 10/27/21.
//

#ifndef BENCHMARK_MATRIX_UTILS_H
#define BENCHMARK_MATRIX_UTILS_H

#include <stdlib.h>
#include <iostream>

template<typename ValueType>
class CSR {
public:
    int rows, cols, nnz;
    ValueType *values;
    int *row_offsets;
    int *col_indices;

    CSR(int rows, int cols, int nnz);
    ~CSR();

    // Shallow copy
    CSR(const CSR &t): rows(t.rows), cols(t.cols), nnz(t.nnz),
      values(t.values), row_offsets(t.row_offsets), col_indices(t.col_indices)
    { }

    CSR& operator = (const CSR &t) { return CSR<ValueType>(*this); }
};

template<typename ValueType>
class DenseMtx {
public:
    int rows, cols;
    ValueType *values;

    DenseMtx(int rows, int cols);
    ~DenseMtx();
};

//
//  Template implementations
//

template<typename ValueType>
CSR<ValueType>::CSR(int rows, int cols, int nnz): rows(rows), cols(cols), nnz(nnz) {
  values = new ValueType[nnz];
  row_offsets = new int[rows + 1];
  col_indices = new int[nnz];
}

template<typename ValueType>
CSR<ValueType>::~CSR() {
//  delete[] values;
//  delete[] row_offsets;
//  delete[] col_indices;
}

template<typename ValueType>
DenseMtx<ValueType>::DenseMtx(int rows, int cols): rows(rows), cols(cols) {
  values = new ValueType[rows * cols];
}

template<typename ValueType>
DenseMtx<ValueType>::~DenseMtx() {
  delete[] values;
}


#endif //BENCHMARK_MATRIX_UTILS_H

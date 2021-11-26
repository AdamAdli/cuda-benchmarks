//
// Created by lwilkinson on 11/25/21.
//

#ifndef BENCHMARK_CODLET_MULTIPLY_CUH
#define BENCHMARK_CODLET_MULTIPLY_CUH

#include "common/utils/matrix_utils.h"
#include "synthetic_codlets.cuh"

namespace CodeletMultiply {
    typedef struct {
        int *col_pattern;
        int col_pattern_len;
        int *rows;
        int num_rows;
        float *row_segment_values;

        float coeff(int row, int col_pattern_idx) const {
          return row_segment_values[row * col_pattern_len + col_pattern_idx];
        }
    } Block;


    constexpr int TILE_K = 32;
    constexpr int VECTOR_WIDTH = 4;
    constexpr int MAX_COLS_PER_BLOCK = 32;
    constexpr int MAX_ROWS_PER_BLOCK = 8;

    constexpr int BLOCK_Y_DIM = 32 / (TILE_K / VECTOR_WIDTH);
    constexpr int BLOCK_X_DIM = TILE_K / VECTOR_WIDTH;

    int codelet_multiply(cudaStream_t &stream, cudaEvent_t &start, cudaEvent_t &stop, const Block *blocks,
                         size_t num_blocks,
                         const CSR<float> &A_h, const CSR<float> &A, const Dense &B, Dense &C);

}
#endif //BENCHMARK_CODLET_MULTIPLY_CUH

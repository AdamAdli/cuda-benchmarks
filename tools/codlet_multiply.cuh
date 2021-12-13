//
// Created by lwilkinson on 11/25/21.
//

#ifndef BENCHMARK_CODLET_MULTIPLY_CUH
#define BENCHMARK_CODLET_MULTIPLY_CUH

#include "common/utils/matrix_utils.h"

#include "dense_matrix.cuh"

namespace CodeletMultiply {
    typedef struct {
        int *col_pattern;
        int col_pattern_len;
        int *rows;
        int num_rows;
        int batch_size;
        float *row_segment_values;

        float coeff(int row, int col_pattern_idx) const {
          return row_segment_values[row * col_pattern_len + col_pattern_idx];
        }
    } Block;


    namespace codelet_4x8x8 {
        constexpr int TILE_K = 32;
        constexpr int VECTOR_WIDTH = 4;
        constexpr int MAX_COLS_PER_BLOCK = 8;
        constexpr int MAX_ROWS_PER_BLOCK = 4;
        constexpr int BATCH_SIZE = 8;

        constexpr int BLOCK_Y_DIM = 4;
        constexpr int BLOCK_X_DIM = 8;

        int codelet_multiply(cudaStream_t &stream, cudaEvent_t &start, cudaEvent_t &stop, const Block *blocks,
                             size_t num_blocks,
                             const CSR<float> &A_h, const CSR<float> &A, const Dense &B, Dense &C);
    }

    namespace codelet_8x32x1 {
        constexpr int TILE_K = 32;
        constexpr int VECTOR_WIDTH = 4;
        constexpr int MAX_COLS_PER_BLOCK = 32;
        constexpr int MAX_ROWS_PER_BLOCK = 8;
        constexpr int BATCH_SIZE = 1;

        constexpr int BLOCK_Y_DIM = 4;
        constexpr int BLOCK_X_DIM = 8;

        int codelet_multiply(cudaStream_t &stream, cudaEvent_t &start, cudaEvent_t &stop, const Block *blocks,
                             size_t num_blocks,
                             const CSR<float> &A_h, const CSR<float> &A, const Dense &B, Dense &C);
    }

    namespace codelet_8x8x1 {
        constexpr int TILE_K = 32;
        constexpr int VECTOR_WIDTH = 4;
        constexpr int BLOCK_COLS = 8;
        constexpr int BLOCK_ROWS = 8;
        constexpr int BATCH_SIZE = 4;

        constexpr int BLOCK_Y_DIM = 4;
        constexpr int BLOCK_X_DIM = 8;

        int codelet_multiply(cudaStream_t &stream, cudaEvent_t &start, cudaEvent_t &stop, const Block *blocks,
                             size_t num_blocks,
                             const CSR<float> &A_h, const CSR<float> &A, const Dense &B, Dense &C);
    }

    namespace codelet_4x16x4 {
        constexpr int TILE_K = 32;
        constexpr int VECTOR_WIDTH = 4;
        constexpr int BLOCK_ROWS = 4;
        constexpr int BLOCK_COLS = 16;
        constexpr int BATCH_SIZE = 4;

        constexpr int BLOCK_Y_DIM = 4;
        constexpr int BLOCK_X_DIM = 8;

        int codelet_multiply(cudaStream_t &stream, cudaEvent_t &start, cudaEvent_t &stop, const Block *blocks,
                             size_t num_blocks,
                             const CSR<float> &A_h, const CSR<float> &A, const Dense &B, Dense &C);
    }


    namespace codelet_4x32x1 {
        constexpr int TILE_K = 32;
        constexpr int VECTOR_WIDTH = 4;
        constexpr int MAX_ROWS_PER_BLOCK = 4;
        constexpr int MAX_COLS_PER_BLOCK = 32;
        constexpr int BATCH_SIZE = 1;

        constexpr int BLOCK_Y_DIM = 4;
        constexpr int BLOCK_X_DIM = 8;

        int codelet_multiply(cudaStream_t &stream, cudaEvent_t &start, cudaEvent_t &stop, const Block *blocks,
                             size_t num_blocks,
                             const CSR<float> &A_h, const CSR<float> &A, const Dense &B, Dense &C);
    }
}
#endif //BENCHMARK_CODLET_MULTIPLY_CUH

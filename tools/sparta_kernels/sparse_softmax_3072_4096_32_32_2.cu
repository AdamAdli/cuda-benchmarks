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

using namespace std;

#define FULL_MASK 0xffffffff

const int H = 3072;
const int W = 4096;
const int block_h = 32;
const int block_w = 32;
const int row_tile = 2;

__global__ void SPARSE_SOFTMAX(
        float* in_val,
        int* row_ptr,
        int* col_idx,
        int* mask,
        float* out_val
) {
    int num_nnz = row_ptr[H / block_h];
    in_val += blockIdx.y * num_nnz * block_h * block_w;

    uint blk_row_idx = blockIdx.x / (block_h/row_tile) ;
    int block_inter_row = (blockIdx.x % (block_h/row_tile)) * row_tile;
    uint bm = threadIdx.x / 32;
    uint bn = threadIdx.x % 32;
    float regC = 0.0f;
    float regSum = 0.0f;
    float regMax = -100000.0;
    int block_seq_start = row_ptr[blk_row_idx];
    int block_seq_end = row_ptr[blk_row_idx+1];

    uint index_list[W / 32];
    int val_num = 0;
    for (int block_inter_col = bn; block_inter_col < block_w; block_inter_col += 32) {
        for (int block_seq = block_seq_start; block_seq < block_seq_end; block_seq++) {

            uint index = block_h * block_w * block_seq + (block_inter_row + bm) * block_w + block_inter_col;

            if (mask[index]) {
                index_list[val_num++] = index;
            }
        }
    }

    for (int k = 0; k < val_num; k++) {
        uint index = index_list[k];
        regMax = max(regMax, in_val[index]);
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        regMax = max(regMax, __shfl_down_sync(FULL_MASK, regMax, offset));
    }
    regMax = __shfl_sync(FULL_MASK, regMax, 0);

    for (int k = 0; k < val_num; k++) {
        uint index = index_list[k];
        regC = expf(in_val[index] - regMax);
        regSum += regC;
    }

    for (int offset = 16; offset > 0; offset >>= 1) {
        regSum += __shfl_down_sync(FULL_MASK, regSum, offset);
    }
    regSum = __shfl_sync(FULL_MASK, regSum, 0);

    for (int k = 0; k < val_num; k++) {
        uint index = index_list[k];
        out_val[index] = expf(in_val[index] - regMax) / regSum;
    }
}


int main() {
    // TODO: anything?
    return 0;
}
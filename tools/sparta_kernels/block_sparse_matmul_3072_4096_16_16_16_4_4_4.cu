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

const int K = 3072;
const int N = 4096;
const int BM = 16;
const int BK = 16;
const int BN = 16;
const int TM = 4;
const int TK = 4;
const int TN = 4;

__global__ void BLOCK_SPARSE_MATMUL(
        float* input_A_val,
        int* input_A_block_ptr,
        int* input_A_block_idx,
        float* input_B,
        float* input_bias,
        float* output_C
) {
    float * A_val = reinterpret_cast<float*>(input_A_val);
    int * A_block_ptr = reinterpret_cast<int*>(input_A_block_ptr);
    int * A_block_idx = reinterpret_cast<int*>(input_A_block_idx);
    float * B = reinterpret_cast<float*>(input_B);
    float * bias = reinterpret_cast<float*>(input_bias);
    float * C = reinterpret_cast<float*>(output_C);

    int by = blockIdx.y;
    int bx = blockIdx.x;
    int ty = threadIdx.y;
    int tx = threadIdx.x;

    __shared__ float As[BM * BK];
    __shared__ float Bs[BN * BK];

    float accum[TN][TM] = {0};
    float a_frag[TM][TK];
    float b_frag[TN][TK];

    int A_THREAD_PER_ROW = BK / 4;
    int B_THREAD_PER_ROW = BK / 4;

    int bszy = BM / TM;
    int bszx = BN / TN;

    int THREADS_PER_BLOCK = bszy * bszx;

    int A_TILE_ROW_STRIDE = THREADS_PER_BLOCK / A_THREAD_PER_ROW;
    int B_TILE_ROW_STRIDE = THREADS_PER_BLOCK / B_THREAD_PER_ROW;

    int tid = ty * bszx + tx;

    int A_BLOCK_ROW_START = tid / A_THREAD_PER_ROW;
    int B_BLOCK_ROW_START = tid / B_THREAD_PER_ROW;

    int A_BLOCK_COL_START = tid % A_THREAD_PER_ROW * 4;
    int B_BLOCK_COL_START = tid % B_THREAD_PER_ROW * 4;

    int index_start = A_block_ptr[by], index_end = A_block_ptr[by+1];

    const int vBLOCK_SIZE_M = BM / TM;
    const int vBLOCK_SIZE_N = BN / TN;
    float4 tmp_float4;
    for (int tile_block_idx = index_start; tile_block_idx < index_end; tile_block_idx += 1) {
        int tile_idx = A_block_idx[tile_block_idx] * BK;
#pragma unroll
        for (int k = 0; k < BM; k += A_TILE_ROW_STRIDE) {
            *((float4 *)(&As[(k+A_BLOCK_ROW_START) * BK + A_BLOCK_COL_START])) =

                    *((float4 *)(&A_val[tile_block_idx * BM * BK + (k+A_BLOCK_ROW_START) * BK + A_BLOCK_COL_START]));

        }

        // Kazem: Storing B into Bs (from the way the indirection is being done here)
        // Maryam: Compare this generated kernel with 2 other equivalent libraries.
        // Kazem: Try running this with/without #pragma unroll
#pragma unroll
        for (int k = 0; k < BN; k += B_TILE_ROW_STRIDE) {

            tmp_float4 = (reinterpret_cast<float4*>(&B[(bx*BN + B_BLOCK_ROW_START+k) * K + tile_idx + B_BLOCK_COL_START]))[0];
            Bs[(B_BLOCK_COL_START+0) * BN + k+B_BLOCK_ROW_START] = tmp_float4.x;
            Bs[(B_BLOCK_COL_START+1) * BN + k+B_BLOCK_ROW_START] = tmp_float4.y;
            Bs[(B_BLOCK_COL_START+2) * BN + k+B_BLOCK_ROW_START] = tmp_float4.z;
            Bs[(B_BLOCK_COL_START+3) * BN + k+B_BLOCK_ROW_START] = tmp_float4.w;

        }

        __syncthreads();

#pragma unroll
        for (int k = 0; k < BK; k += TK) {
#pragma unroll
            for (int i = 0; i < TK; i++) {
#pragma unroll
                for (int j = 0; j < TM; j += 1) {
                    a_frag[j][i] = As[(ty + vBLOCK_SIZE_M * j) * BK + k + i];
                }
            }

#pragma unroll
            for (int i = 0; i < TK; i++) {
#pragma unroll
                for (int j = 0; j < TN; j += 1) {
                    b_frag[j][i] = Bs[(k + i) * BN + tx + vBLOCK_SIZE_N * j];
                }
            }

#pragma unroll
            for (int i = 0; i < TN; i++) {
#pragma unroll
                for (int j = 0; j < TM; j++) {
#pragma unroll
                    for (int k_in = 0; k_in < TK; k_in++) {
                        accum[i][j] += a_frag[j][k_in] * b_frag[i][k_in];
                    }
                }
            }
        }

        __syncthreads();
    }


    float bias_local[TN];
    for (int thread_x = 0; thread_x < TN; thread_x++) {
        bias_local[thread_x] = bias[BN * bx + tx + thread_x * vBLOCK_SIZE_N];
    }


#pragma unroll
    for (int thread_x = 0; thread_x < TN; thread_x++) {
#pragma unroll
        for (int thread_y = 0; thread_y < TM; thread_y+=1) {
            C[(BM * by + ty + thread_y * vBLOCK_SIZE_M) * N + BN * bx + tx + thread_x * vBLOCK_SIZE_N] =
                    (accum[thread_x][thread_y]) + bias_local[thread_x];
        }
    }
}

int main() {
    // TODO: anything?
    return 0;
}
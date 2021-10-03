#!/bin/bash

# CUDA_ARCH_GENCODE="arch=compute_60,code=sm_60"
CUDA_ARCH_GENCODE="arch=compute_75,code=sm_75"
NVCC_ARGS=""

cd ASpT_SpMM_GPU
nvcc -std=c++14 -O3 ${NVCC_ARGS} -gencode ${CUDA_ARCH_GENCODE} sspmm_32.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o sspmm_32
nvcc -std=c++14 -O3 ${NVCC_ARGS} -gencode ${CUDA_ARCH_GENCODE} sspmm_128.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o sspmm_128
nvcc -std=c++14 -O3 ${NVCC_ARGS} -gencode ${CUDA_ARCH_GENCODE} dspmm_32.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o dspmm_32
nvcc -std=c++14 -O3 ${NVCC_ARGS} -gencode ${CUDA_ARCH_GENCODE} dspmm_128.cu --use_fast_math -Xptxas "-v -dlcm=ca" -o dspmm_128
cd ..


//
// Created by lwilkinson on 11/25/21.
//

#ifndef BENCHMARK_CUDA_UTILS_H
#define BENCHMARK_CUDA_UTILS_H

#include "cuda_runtime.h"
#include "cublas_v2.h"

// cuBLAS API errors
constexpr const char *_cudaGetErrorEnum(cublasStatus_t error)
{
  switch (error)
  {
    case CUBLAS_STATUS_SUCCESS:
      return "CUBLAS_STATUS_SUCCESS";

    case CUBLAS_STATUS_NOT_INITIALIZED:
      return "CUBLAS_STATUS_NOT_INITIALIZED";

    case CUBLAS_STATUS_ALLOC_FAILED:
      return "CUBLAS_STATUS_ALLOC_FAILED";

    case CUBLAS_STATUS_INVALID_VALUE:
      return "CUBLAS_STATUS_INVALID_VALUE";

    case CUBLAS_STATUS_ARCH_MISMATCH:
      return "CUBLAS_STATUS_ARCH_MISMATCH";

    case CUBLAS_STATUS_MAPPING_ERROR:
      return "CUBLAS_STATUS_MAPPING_ERROR";

    case CUBLAS_STATUS_EXECUTION_FAILED:
      return "CUBLAS_STATUS_EXECUTION_FAILED";

    case CUBLAS_STATUS_INTERNAL_ERROR:
      return "CUBLAS_STATUS_INTERNAL_ERROR";
  }

  return "<unknown>";
}

#define CHECK_CUDA(func)                                                        \
    {                                                                           \
        cudaError_t status = (func);                                            \
        if (status != cudaSuccess)                                              \
        {                                                                       \
            printf("CUDA API failed at %s:%d with error: %s (%d)\n",            \
                   __FILE__, __LINE__, cudaGetErrorString(status), status);     \
            return EXIT_FAILURE;                                                \
        }                                                                       \
    }

#define CHECK_CUSPARSE(func)                                                    \
    {                                                                           \
        cusparseStatus_t status = (func);                                       \
        if (status != CUSPARSE_STATUS_SUCCESS)                                  \
        {                                                                       \
            printf("CUSPARSE API failed at %s:%d with error: %s (%d)\n",        \
                   __FILE__, __LINE__, cusparseGetErrorString(status), status); \
            return EXIT_FAILURE;                                                \
        }                                                                       \
    }

#define CHECK_CUBLAS(func)                                                    \
    {                                                                           \
        cublasStatus_t status = (func);                                         \
        if (status != CUBLAS_STATUS_SUCCESS)                                    \
        {                                                                       \
            printf("CUSPARSE API failed at %s:%d with error: %s (%d)\n",        \
                   __FILE__, __LINE__, _cudaGetErrorEnum(status), status);       \
            return EXIT_FAILURE;                                                \
        }                                                                       \
    }

#endif //BENCHMARK_CUDA_UTILS_H

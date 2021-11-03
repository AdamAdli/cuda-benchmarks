#include <algorithm>
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cstdio>
#include <chrono>

#include "common/kernel_wrappers.h"

namespace py = pybind11;


/**
 * @brief This function time the cuSparse sparse-dense matrix multiplication: A @ B where A is of
 * shape m * k and B is of shape k * n
 * @return float : the time in millisecond
 */

float py_test_cusparse_gemm(int m, int n, int k, int A_nnz, py::array_t<int> A_csr_offsets,
                         py::array_t<int> A_csr_columns, py::array_t<float> A_csr_values, py::array_t<float> arr_B)
{
    //Get the array from the input
    py::buffer_info buf_A_csr_offsets = A_csr_offsets.request();
    py::buffer_info buf_A_csr_columns = A_csr_columns.request();
    py::buffer_info buf_A_csr_values = A_csr_values.request();
    py::buffer_info buf_B_values = arr_B.request();

    int *hA_csr_offsets = (int *)buf_A_csr_offsets.ptr;
    int *hA_csr_columns = (int *)buf_A_csr_columns.ptr;
    float *hA_csr_values = (float *)buf_A_csr_values.ptr;
    float *hB_values = (float *)buf_B_values.ptr;

    return test_cusparse_gemm(m, k, n, A_nnz, hA_csr_values, hA_csr_offsets, hA_csr_columns, hB_values);
}

float py_test_cublas_sgemm(int m, int n, int k, py::array_t<float> arr_A, py::array_t<float> arr_B)
{
    //remember the mtx is col based!!!
    //init the variables

    float *A, *B;
#ifdef DEBUG
    //define the output variable C
    float *C;
    C = (float *)malloc(sizeof(float) * m * n);
#endif

    // get the elements inside the numpy passed in array
    py::buffer_info buf_A = arr_A.request();
    py::buffer_info buf_B = arr_B.request();
    A = (float *)buf_A.ptr;
    B = (float *)buf_B.ptr;

    return test_cublas_sgemm(m, k, n, A, B);
}

float py_test_sgk_spmm(
        int m, int n, int k, int nonzeros,
        py::array_t<float> A_values, py::array_t<int> A_row_offsets, py::array_t<int> A_col_indices,
        py::array_t<float> B_values)
{
    //Get the array from the input
    py::buffer_info buf_A_value = A_values.request();
    py::buffer_info buf_A_row_offsets = A_row_offsets.request();
    py::buffer_info buf_A_col_indices = A_col_indices.request();
    py::buffer_info buf_B_values = B_values.request();

    float *h_values = (float *)buf_A_value.ptr;
    int *h_row_offsets = (int *)buf_A_row_offsets.ptr;
    int *h_col_indices = (int *)buf_A_col_indices.ptr;
    float *h_dense_matrix = (float *)buf_B_values.ptr;

    return test_sgk_spmm(m, k, n, nonzeros, h_values, h_row_offsets, h_col_indices, h_dense_matrix);
}

float py_test_sgk_spmm_custom_row_order(
    int m, int n, int k, int nonzeros,
    py::array_t<float> A_value, py::array_t<int> A_row_idex, py::array_t<int> A_row_offsets, py::array_t<int> A_col_indices,
    py::array_t<float> B_values)
{

    typedef std::chrono::steady_clock Clock;
    typedef std::chrono::milliseconds nanoseconds;

    //Get the array from the input
    py::buffer_info buf_A_value = A_value.request();
    py::buffer_info buf_A_row_idex = A_row_idex.request();
    py::buffer_info buf_A_row_offsets = A_row_offsets.request();
    py::buffer_info buf_A_col_indices = A_col_indices.request();
    py::buffer_info buf_B_values = B_values.request();

    float *h_values = (float *)buf_A_value.ptr;
    int *h_row_indices = (int *)buf_A_row_idex.ptr;
    int *h_row_offsets = (int *)buf_A_row_offsets.ptr;
    int *h_col_indices = (int *)buf_A_col_indices.ptr;
    float *h_dense_matrix = (float *)buf_B_values.ptr;

    return test_sgk_spmm_custom_row_order(m, k, n, nonzeros,
                                          h_values, h_row_offsets, h_col_indices, h_row_indices, h_dense_matrix);
}



py:

    std::size_t const size = calculateSize();
    ndarray::Array<double, 2, 1> array = ndarray::allocate(size, size);
    array.deep() = 0;  // initialise
    for (std::size_t ii = 0; ii < size; ++ii) {
    array[ii][ndarray::view(ii, ii + 1)] = 1.0;
    }
    return array;

//Pybind call
PYBIND11_MODULE(kernel_python_bindings, m)
{
    m.def("cuBLAS", &py_test_cublas_sgemm,
          "the function returning the RT of cuBLAS");
    m.def("cuSPARSE", &py_test_cusparse_gemm,
          "the function returning the RT of cuSPARSE");
    m.def("sgk", &py_test_sgk_spmm,
          "the function returning the RT of sgk");
    m.def("sgk_custom_row_order", &py_test_sgk_spmm_custom_row_order,
          "the function returning the RT of sgk, with custom row swizzle");
}
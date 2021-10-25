from scipy.sparse.csr import csr_matrix
import numpy as np


def scipy_spmm_kernel_decorator(func):
    def wrapper(A_mtx: csr_matrix, B_mtx: np.matrix):
        assert(A_mtx.shape[1] == B_mtx.shape[0])

        m, k = A_mtx.shape
        n = B_mtx.shape[1]

        return func(m, n, k, A_mtx.nnz,    # m, n, k, nnz
                 A_mtx.data.astype(np.float32), A_mtx.indptr.astype(np.int32), A_mtx.indices.astype(np.int32),
                 B_mtx.flatten().astype(np.float32))
    return wrapper
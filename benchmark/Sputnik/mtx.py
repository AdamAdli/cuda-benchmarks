import numpy  as np
import matplotlib.pyplot as plt

def get_mtx(path):
    with open(path, 'r') as f:
        contents = f.readlines()
        contents = [i.strip() for i in contents]
        info, row_pt, col_indx = contents
        row_pt = row_pt.split(" ")
        row_pt = [int(i) for i in row_pt]
        col_indx = col_indx.split(" ")
        col_indx = [int(i) for i in col_indx]

        info = info.split(", ")
        info = [int(i) for i in info]
        n_row, n_col, nnz = info
        result = np.zeros((n_row, n_col))
        for i in range(len(row_pt) - 1):
            start, end = row_pt[i], row_pt[i+1]
            for j in range(start, end):
                try:
                    result[i, col_indx[j]] = 1
                except:
                    raise ValueError("error!!! i: {}, j: {}".format(i, col_indx[j]))
                    
    return result

 

def dense_to_sparse(matrix):
  """Converts dense numpy matrix to a csr sparse matrix."""
  assert len(matrix.shape) == 2

  # Extract the nonzero values.
  values = matrix.compress((matrix != 0).flatten())

  # Calculate the offset of each row.
  mask = (matrix != 0).astype(np.int32)
  row_offsets = np.concatenate(([0], np.cumsum(np.add.reduce(mask, axis=1))),
                               axis=0)

  # Create the row indices and sort them.
  row_indices = np.argsort(-1 * np.diff(row_offsets))

  # Extract the column indices for the nonzero values.
  x = mask * (np.arange(matrix.shape[1]) + 1)
  column_indices = x.compress((x != 0).flatten())
  column_indices = column_indices - 1

  # Cast the desired precision.
  values = values.astype(np.float32)
  row_indices, row_offsets, column_indices = [
      x.astype(np.uint32) for x in
      [row_indices, row_offsets, column_indices]
  ]
  return values, row_indices, row_offsets, column_indices


if __name__ == "__main__":
    path = '/mnt/benchmark/dlmc/rn50/extended_magnitude_pruning/0.8/bottleneck_1_block_group_projection_block_group1.smtx'
    mtx = get_mtx(path)
    plt.imshow(mtx)
    plt.savefig("./shit.pdf")

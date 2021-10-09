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

    

if __name__ == "__main__":
    path = '/mnt/benchmark/dlmc/rn50/extended_magnitude_pruning/0.8/bottleneck_1_block_group_projection_block_group1.smtx'
    mtx = get_mtx(path)
    plt.imshow(mtx)
    plt.savefig("./shit.pdf")

# Sputnik Implementation Notes
## TODOs

* Explain predicates (binary vectors to filter dense matrix loading)
* Explain `ComputeUtil`
* Explain final aggregation and output

## Constants
Template parameters for `SpmmConfig`:
```cpp
template <typename ScalarValue_,    // Scalar data type for all operands.
          typename SparseValue_,    // Vector data type for the sparse matrix.
          typename DenseValue_,     // Vector data type for the dense operands.
          int kBlockItemsY_,        // Tile size in the m-dimension.
          int kBlockItemsK_,        // Tile size in the k-dimension.
          int kBlockItemsX_,        // Tile size in the n-dimension.
          int kBlockWidth_,         // Threadblock width.
          int kResidueUnroll_ = 4,  // Number of unroll steps in the residue.
          int kPredicateLoads_ = true,  // Whether to predicate loads or not.
          bool kLaunchBounds_ = false,  // Whether or not to set launch bounds.
          int kMinOccupancy_ = 8>       // Minimum occupancy to target.
```

We multiply matrices of size `(m,k)` with `(k,n)`.

## Work Distribution
**`IndexM`**: divides rows of the sparse matrix, defined as: `blockIdx.x * kBlockItemsY + threadIdx.y`. Within each block, `threadIdx.y` locates the row responsible within the row block of size `kBlockItemsY`. 

**`IndexN`**: divices columns of the dense matrix, defined as: `blockIdx.y * kBlockItemsX`.

Each thread block (defined by its 2D index) is responsible for computing a `kBlockItemsY` by `kBlockItemsX` rectangle in the final matrix at one time. 

## Collaborative Loading
### Sparse Matrix (`sparse_tile.h`)
Array of size `kTileSize` maintained in shared memory within each thread block. `kTileSize` is `kBlockItemsK * kBlockItemsY`. `k` elements per row and `y` rows. 

Each thread (`0 <= threadIdx.x <= kBlockWidth`) processes 1D tiles in the sparse matrix. 

* `values_tile` to store the values and
* `column_indices_tile` to store their corresponding indices.

`sparse_tile_loader` initialized to collaboratively load sparse tiles. During its initialization, the following values are populated:

* `values_`, `column_idxs_`:
    Offsetted by `col_values+row_offset+thread_idx_x`, stores the (strided) sparse matrix values.
* `values_tile_base_`, `column_idxs_tile_base_`: same as above, but stored in shared memory. These will be the targets of loads. 

Each thread is responsible for loading `kThreadItemsK` which is `kBlocksItemsK` spread evenly among `kBlockWidth` blocks and `kValuesPerLoad_` (the number of `ScalarValues` that could fit in `Value`). 

Every time `Load` is called, we load `kBlockWidth` values collaborately each time into shared memory (hence the offset by `threadIdx.x` when `sparse_tile_loader` is constructed). After each collaborative load, we move on to the next block `kBlockWidth` values away.

### Dense Matrix (`dense_tile.h`)
This is slightly more complex than loading sparse matrices since we need to stripmine a 2D tile from the dense matrix.

Outputs a dense matrix fragment of size `kElementsPerScalar * kBlockItemsK * kBlockItemsX / kBlockWidth`.

For every 1D tile of `kBlockItemsK` from the sparse matrix, we need to load `kBlockItemsX` items from the dense matrix. Each thread is responsible for `kThreadItemsX` of them (`kBlockItemsX` items evenly spread throughout `kBlockWidth` loads with `kValuesPerLoad` values per load).

Accepts:

* `rhs_columns`: n
* `offset`: n_index
* `thread_idx_x`: threadIdx.x
* `matrix`: dense_matrix
* `row_offsets`: column_indices_tile, where `row_offsets[i]` is the row we need to read from in the dense matrix. 

Writes into:

* `matrix_fragment`: dense_matrix_fragment

```cpp
matrix_base = matrix + offset + threadIdx.x;
// Loop over rows in the dense matrix corresponding to each sparse entry.
for (int k_item_idx = 0; k_item_idx < kBlockItemsK; ++k_item_idx) {
    // We can shove multiple (e.g. 4 with float4) values into one element, each
    // entry in row_indices may correspond to multiple actual row indices.
    scaled_indices = Convert(row_indices);

    // Since we're using vector types (e.g. float4), we can stuff multiple elements
    // in each scalar. We loop over those. 
    for (int elt_idx = 0; elt_idx < kElementsPerScalar; ++elt_idx) {
        const Value *matrix = matrix_base + scaled_indices[elt_idx];

        for (int x_item_idx = 0; x_item_idx < kThreadItemsX_; ++x_item_idx) {
            // locate the beginning of the k'th row
            int offset1 = k_item_idx * kThreadItemsX_ * kElementsPerScalar;

            // Each entry in row_indices correspond to multiple actual indices.
            // We consider the position of one of them here.
            int offset2 = elt_idx * kThreadItemsX_;

            fragment[offset1 + offset2 + x_item_idx] = Load(matrix);

            // Go on to load for the next tile.
            matrix += kBlockWidth;
        }
    }
    ++ row_indices;
}
```

If we make the simplifying assumption to ignore vector types, the above would look something like this:
```cpp
matrix_base = matrix + offset + threadIdx.x;
for (int k_item_idx = 0; k_item_idx < kBlockItemsK; ++k_item_idx) {
    // Instead here, row_indices point to exactly one element.
    const Value *matrix = matrix_base + *row_indices;
    for (int x_item_idx = 0; x_item_idx < kThreadItemsX_; ++x_item_idx) {
        // We no longer care about indexing for vector types
        int offset1 = k_item_idx * kThreadItemsX_;
        fragment[offset1 + x_item_idx] = Load(matrix);
        matrix += kBlockWidth;
    }
    ++ row_indices;
}
```

## Computing Partial Matmuls (`compute_utils.h`)
TODO. 

Essentially just computes the indices and calls `VectorCompute<Value>::FMA` and writes into `output_fragment`.
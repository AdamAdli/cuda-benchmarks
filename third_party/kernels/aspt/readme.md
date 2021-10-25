# Some notes on ASpT
## Implementation
### `ready` function
Responsible for reading the matrix files. Does the following:

#### Initialize sparse matrix
Read the first line, sets `sflag` for symmetric and `nflag` for pattern/complex/other. If complex, each element consists of 2 floating point numbers, the second (imaginary) part is dropped. 

Skips all lines starting with `%`, considering them comments.

Variables:
* `nr`: number of rows
* `nc`: number of columns
* `ne`: number of non-zero elements
* `csr_v`: CSR row offsets
* `csr_e0`: column number of the i'th non-zero element
* `csr_ev0`: value of the i'th non-zero element

If symmetric, `ne` is doubled and the existing elements are duplicated with row/col swapped.

nflag: `pattern` means random, `other` means use whatever is in the file.

#### Initialize dense matrix
Dense input matrix initialized as `vin` with dimensions `nc`*`sc`. The parameter `sc` is passed in as `argv[2]`.

### `process` function
None of the `cudaMemcpy`s and `cudaMalloc`s occurred during sparse matrix copy are accounted for.

`ptot_ms` (L1022-L1141):
* Preprocessing time

`tot_ms` (L1157-L1211):
* Actual kernel time
* TODO: Check heuristics

## Variants
TODO. Let's test both for now.
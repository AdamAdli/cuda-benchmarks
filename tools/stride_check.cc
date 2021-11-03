//
// Created by lwilkinson on 10/27/21.
//

#include <boost/dynamic_bitset.hpp>
#include <boost/ptr_container/ptr_vector.hpp>

#include "common/utils/smtx_io.h"

typedef struct {
    int stride = -1;
    int phase = -1;
    float score = 0.0;
    float pct = 0.0;
    int matching = 0;
    int nnz_row = 0;
} best_stride_t;

int main(int argc, char* argv[]) {
  printf("argc %d %s\n", argc, argv[0]);

  if (argc != 2) {
    printf("Usage: row_similarity <path to smtx file>");
    return -1;
  }

  std::string file_path(argv[1]);

  auto A = load_smtx(file_path);
  boost::ptr_vector<boost::dynamic_bitset<>> row_patterns;
  row_patterns.reserve(A->rows);

  for (int i = 0; i < A->rows; i++) {
    row_patterns.push_back(new boost::dynamic_bitset<>(A->cols));
    for (int p = A->row_offsets[i]; p < A->row_offsets[i+1]; p++) {
      row_patterns[i].set(A->col_indices[p]);
    }
  }

  std::vector<best_stride_t> best_strides(A->rows);
  for (int stride = 0; stride < A->cols / 32; stride++) {
    for (int phase = 0; phase < stride; phase++) {
      for (int i = 0; i < A->rows; i++) {

      }
    }
  }
  for (int i = 0; i < A->rows; i++) {
    printf("%3d: ", A->row_offsets[i+1] - A->row_offsets[i]);
    float best_pct = 0.f;
    int best_ii = 0;
    int best_p = 0;

    for (int ii = 0; ii < A->rows; ii++) {
      if (ii == i) continue;
      for (int p = 0; p < 10; p++) {
        auto num_different = (row_patterns[i] ^ (row_patterns[ii] << p)).count();
        auto total_nnz = row_patterns[i].count() + row_patterns[ii].count();
        float pct = (total_nnz - num_different) / float(total_nnz);
        if (pct > best_pct) {
          best_pct = pct;
          best_p = p;
          best_ii = ii;
        }
      }
      //printf("%f, ", (total_nnz - num_different) / float(total_nnz) );
    }
    printf("%f (%d, %d)\n", best_pct, best_p, best_ii);
  }

  return 0;
}
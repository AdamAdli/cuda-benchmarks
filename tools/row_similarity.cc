//
// Created by lwilkinson on 10/27/21.
//

#include <boost/dynamic_bitset.hpp>
#include <boost/ptr_container/ptr_vector.hpp>

#include "common/utils/smtx_io.h"

int main(int argc, char* argv[]) {
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


  printf("%d: ", A->rows);
  for (int i = 0; i < A->rows; i++) {
    printf("%d, ", A->row_offsets[i+1] - A->row_offsets[i]);
  }
  printf("\n");

  for (int i = 0; i < A->rows; i++) {

    float best_pct = 0.f;
    int best_ii = 0;
    int best_p = 0;

    for (int ii = 0; ii < A->rows; ii++) {
      auto union_count = (row_patterns[i] & row_patterns[ii]).count();
      auto total_nnz = row_patterns[i].count();
      float pct = (total_nnz) / float(total_nnz);
      printf("%f, ", (union_count) / float(total_nnz) );
    }
    printf("\n");
  }

  return 0;
}
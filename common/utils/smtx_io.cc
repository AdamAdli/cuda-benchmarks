//
// Created by lwilkinson on 10/26/21.
//

#include "smtx_io.h"

#include <algorithm>
#include <fstream>
#include <sstream>
#include <iostream>

CSR<float>* load_smtx(const std::string& filepath) {
  std::ifstream in (filepath);
  std::string line;
  int i;

  std::getline(in, line);
  std::replace( line.begin(), line.end(), ',', ' ');
  std::istringstream first_line(line);

  int rows, cols, nnz;
  first_line >> rows;
  first_line >> cols;
  first_line >> nnz;

  CSR<float>* csr = new CSR<float>(rows, cols, nnz);

  for (int i = 0; i < csr->rows + 1; i++) {
    in >> csr->row_offsets[i];
  }

  char next;
  while(in.get(next))
  {
    if (next == '\n')  // If the file has been opened in
    {    break;        // text mode then it will correctly decode the
    }                  // platform specific EOL marker into '\n'
  }

  for (int i = 0; i < csr->nnz; i++) {
    in >> csr->col_indices[i];
  }

  for (i = 0; i < csr->nnz; i++) { csr->values[i] = 1.0f; }

  return csr;
}
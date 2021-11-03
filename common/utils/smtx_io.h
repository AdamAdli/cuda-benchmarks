//
// Created by lwilkinson on 10/26/21.
//

#ifndef BENCHMARK_SMTX_IO_H
#define BENCHMARK_SMTX_IO_H

#include "matrix_utils.h"
#include <string>

CSR<float>* load_smtx(const std::string& filepath);

#endif //BENCHMARK_SMTX_IO_H

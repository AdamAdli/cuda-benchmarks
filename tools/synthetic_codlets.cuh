//
// Created by lwilkinson on 11/4/21.
//

#ifndef BENCHMARK_SYNTHETIC_CODLETS_CUH
#define BENCHMARK_SYNTHETIC_CODLETS_CUH

typedef struct Dense {
    int rows;
    int cols;
    float * values;

    Dense(int m, int n, float fill_val): rows(m), cols(n) {
      values = new float[rows * cols];
      for (int i = 0; i < rows * cols; i++) values[i] = fill_val;
    }

    Dense(int m, int n, float * values): rows(m), cols(n), values(values) {}

    float coeff(int i, int j) const {
      return values[i * cols + j];
    }

    inline void set_coeff(int i, int j, float val) {
      values[i * cols + j] = val;
    }

    void fill(float fill_val) {
      for (int i = 0; i < rows * cols; i++) values[i] = fill_val;
    }

} Dense;

#endif //BENCHMARK_SYNTHETIC_CODLETS_CUH

//
// Created by lwilkinson on 8/20/21.
//

#ifndef ITER_SOLVER_TEST_HARNESS_CSV_LOG_IO_H
#define ITER_SOLVER_TEST_HARNESS_CSV_LOG_IO_H

#include <map>
#include <set>
#include <fstream>
#include <sstream>

namespace test_harness {

typedef std::map<std::string, std::string> csv_row_t;
void write_csv_row(std::string filepath, csv_row_t &csv_row);

template<typename Value>
void csv_row_insert(csv_row_t &csv_row, std::string prefix, std::string name, Value value);

void csv_row_insert(csv_row_t &csv_row, std::string name, double value);
void csv_row_insert(csv_row_t &csv_row, std::string name, int value);
void csv_row_insert(csv_row_t &csv_row, std::string name, size_t value);
void csv_row_insert(csv_row_t &csv_row, std::string name, std::ptrdiff_t value);
void csv_row_insert(csv_row_t &csv_row, std::string name, std::string value);

template <typename T>
std::string to_string_with_precision(const T a_value, const int n = 20);


//
//  Template Definitions
//


template<typename Value>
void csv_row_insert(csv_row_t &csv_row, std::string prefix, std::string name, Value value) {
  auto qualified_name = prefix + " " + name;
  csv_row_insert(csv_row, qualified_name, value);
}

} // namespace test_harness

#endif //ITER_SOLVER_TEST_HARNESS_CSV_LOG_IO_H

//
// Created by lwilkinson on 8/20/21.
//

#include "csv_log_io.h"

#include <iostream>
#include <range/v3/view.hpp>

namespace test_harness {

//
//  Utils
//

template <typename T>
std::string to_string_with_precision(const T a_value, const int n)
{
  std::ostringstream out;
  out.precision(n);
  out << std::fixed << a_value;
  return out.str();
}

typedef std::map<std::string, std::pair<std::ofstream, std::set<std::string>>> csv_files_cache_t;
static csv_files_cache_t  csv_files_cache;

void write_csv_row(
    std::string filepath,
    std::map<std::string, std::string> &log
) {
  auto file_cache = csv_files_cache.find(filepath);
  bool write_header = false;

  auto keys_view = ranges::views::keys(log);
  auto column_names = std::set(keys_view.begin(), keys_view.end());

  if (file_cache == csv_files_cache.end()) {
    // First time we have seen this file so we need to open up a `ofstream` and initialize the column names
    csv_files_cache[filepath] = std::make_pair(std::ofstream(filepath), std::move(column_names));
    file_cache = csv_files_cache.find(filepath);
    write_header = true;
  } else {
    auto& column_names_written = file_cache->second.second;
    assert(column_names_written == column_names && "Column names do not match the first row written");
  }

  auto& file = file_cache->second.first;

  if (write_header) {
    for (auto it = log.cbegin(); it != log.cend(); ++it)
      file << it->first << ",";
    file << "\n";
  }

  for (auto it = log.cbegin(); it != log.cend(); ++it)
    file << it->second << ",";
  file << "\n";

  file.flush();
}

void csv_row_insert(csv_row_t &csv_row, std::string name, double value) {
  csv_row[name] = to_string_with_precision(value);
}

void csv_row_insert(csv_row_t &csv_row, std::string name, int value) {
  csv_row[name] = std::to_string(value);
}

void csv_row_insert(csv_row_t &csv_row, std::string name, std::ptrdiff_t value) {
  csv_row[name] = std::to_string(value);
}

void csv_row_insert(csv_row_t &csv_row, std::string name, size_t value) {
  csv_row[name] = std::to_string(value);
}

void csv_row_insert(csv_row_t &csv_row, std::string name, std::string value) {
  csv_row[name] = value;
}
} // namespace test_harness
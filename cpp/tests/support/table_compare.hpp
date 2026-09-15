#pragma once

#include <cstddef>
#include <set>
#include <string>
#include <vector>

#include "guiding/table/csv.hpp"

namespace guiding::testing {

struct TableComparison {
  bool header_equal = true;
  bool row_count_equal = true;
  std::size_t numeric_cells = 0;
  std::size_t bit_identical_cells = 0;
  std::size_t mismatches = 0;
  double max_relative_difference = 0.0;
  std::vector<std::string> messages;

  [[nodiscard]] bool ok() const { return header_equal && row_count_equal && mismatches == 0; }
  [[nodiscard]] std::string describe() const;
};

// Compares CSV tables cell by cell: numeric cells within a relative tolerance
// (NaN equals NaN), everything else as exact text.
[[nodiscard]] TableComparison compare_tables(const table::CsvTable& actual, const table::CsvTable& expected,
                                             double relative_tolerance,
                                             const std::set<std::string>& ignored_columns = {});

}  // namespace guiding::testing

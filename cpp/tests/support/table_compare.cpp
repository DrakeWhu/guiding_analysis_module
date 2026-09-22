#include "support/table_compare.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>

#include <fmt/format.h>
#include <fmt/ranges.h>

namespace guiding::testing {

std::string TableComparison::describe() const {
  std::string text = fmt::format("header_equal={} row_count_equal={} numeric_cells={} bit_identical={} "
                                 "mismatches={} max_rel_diff={:.3g}",
                                 header_equal, row_count_equal, numeric_cells, bit_identical_cells, mismatches,
                                 max_relative_difference);
  const std::size_t shown = std::min<std::size_t>(messages.size(), 12);
  for (std::size_t i = 0; i < shown; ++i) {
    text += "\n  " + messages[i];
  }
  if (messages.size() > shown) {
    text += fmt::format("\n  ... {} more", messages.size() - shown);
  }
  return text;
}

TableComparison compare_tables(const table::CsvTable& actual, const table::CsvTable& expected,
                               double relative_tolerance, const std::set<std::string>& ignored_columns) {
  TableComparison result;
  if (actual.columns != expected.columns) {
    result.header_equal = false;
    result.messages.push_back(fmt::format("columns differ: actual [{}] expected [{}]", fmt::join(actual.columns, ","),
                                          fmt::join(expected.columns, ",")));
    return result;
  }
  if (actual.rows.size() != expected.rows.size()) {
    result.row_count_equal = false;
    result.messages.push_back(fmt::format("row count {} != {}", actual.rows.size(), expected.rows.size()));
    return result;
  }
  for (std::size_t row = 0; row < actual.rows.size(); ++row) {
    for (std::size_t col = 0; col < actual.columns.size(); ++col) {
      const std::string& name = actual.columns[col];
      if (ignored_columns.contains(name)) {
        continue;
      }
      const std::string a = col < actual.rows[row].size() ? actual.rows[row][col] : std::string();
      const std::string e = col < expected.rows[row].size() ? expected.rows[row][col] : std::string();
      const auto a_number = table::parse_number(a);
      const auto e_number = table::parse_number(e);
      if (a_number && e_number) {
        ++result.numeric_cells;
        const double x = *a_number;
        const double y = *e_number;
        if (std::memcmp(&x, &y, sizeof(double)) == 0) {
          ++result.bit_identical_cells;
          continue;
        }
        if (std::isnan(x) && std::isnan(y)) {
          continue;
        }
        const double scale = std::max(std::abs(x), std::abs(y));
        const double relative = scale == 0.0 ? 0.0 : std::abs(x - y) / scale;
        if (std::isfinite(relative)) {
          result.max_relative_difference = std::max(result.max_relative_difference, relative);
        }
        if (!(relative <= relative_tolerance)) {
          ++result.mismatches;
          result.messages.push_back(
              fmt::format("row {} column {}: actual {} expected {} (rel {:.3g})", row, name, a, e, relative));
        }
        continue;
      }
      if (a != e) {
        ++result.mismatches;
        result.messages.push_back(fmt::format("row {} column {}: actual '{}' expected '{}'", row, name, a, e));
      }
    }
  }
  return result;
}

}  // namespace guiding::testing

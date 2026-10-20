#pragma once

#include <algorithm>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "guiding/table/csv.hpp"

namespace guiding::table {

// An insertion-ordered key/value row with Python dict semantics: assigning an
// existing key keeps its original position ({**base, key: value}).
class Record {
 public:
  Record() = default;
  Record(std::initializer_list<std::pair<std::string, Cell>> items) {
    for (const auto& [key, value] : items) {
      set(key, value);
    }
  }

  void set(const std::string& key, Cell value) {
    const auto it = std::find_if(items_.begin(), items_.end(), [&](const auto& item) { return item.first == key; });
    if (it != items_.end()) {
      it->second = std::move(value);
    } else {
      items_.emplace_back(key, std::move(value));
    }
  }

  void merge(const Record& other) {
    for (const auto& [key, value] : other.items_) {
      set(key, value);
    }
  }

  [[nodiscard]] std::optional<Cell> get(std::string_view key) const {
    const auto it = std::find_if(items_.begin(), items_.end(), [&](const auto& item) { return item.first == key; });
    if (it == items_.end()) {
      return std::nullopt;
    }
    return it->second;
  }

  [[nodiscard]] const std::vector<std::pair<std::string, Cell>>& items() const noexcept { return items_; }

 private:
  std::vector<std::pair<std::string, Cell>> items_;
};

// pandas.DataFrame(records).to_csv(index=False) for records that share their
// value type per column: columns in order of first appearance, missing values
// empty.
[[nodiscard]] inline std::string format_records_pandas(std::span<const Record> records) {
  std::vector<std::string> columns;
  for (const auto& record : records) {
    for (const auto& [key, value] : record.items()) {
      if (std::find(columns.begin(), columns.end(), key) == columns.end()) {
        columns.push_back(key);
      }
    }
  }
  const auto dialect = CsvDialect::pandas();
  std::string out;
  append_csv_header(out, columns, dialect);
  std::vector<Cell> cells(columns.size());
  for (const auto& record : records) {
    for (std::size_t i = 0; i < columns.size(); ++i) {
      cells[i] = record.get(columns[i]).value_or(Cell{});
    }
    append_csv_record(out, cells, dialect);
  }
  return out;
}

}  // namespace guiding::table

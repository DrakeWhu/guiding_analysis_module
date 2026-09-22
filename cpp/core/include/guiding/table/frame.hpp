#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <variant>
#include <vector>

#include <fmt/format.h>

#include "guiding/table/csv.hpp"
#include "guiding/table/py_format.hpp"
#include "guiding/table/record.hpp"

namespace guiding::table {

// A column-typed table standing in for the pandas DataFrames of the reference
// pipeline: int64, float64 or str columns, written like DataFrame.to_csv.
class Frame {
 public:
  using Integers = std::vector<std::int64_t>;
  using Doubles = std::vector<double>;
  using Strings = std::vector<std::string>;
  using Column = std::variant<Integers, Doubles, Strings>;

  [[nodiscard]] std::size_t row_count() const noexcept { return rows_; }
  [[nodiscard]] const std::vector<std::string>& names() const noexcept { return names_; }
  [[nodiscard]] bool has(std::string_view name) const noexcept { return find(name) != kNpos; }

  [[nodiscard]] const Column& column(std::string_view name) const {
    const std::size_t index = find(name);
    if (index == kNpos) {
      throw std::out_of_range(fmt::format("table has no column '{}'", name));
    }
    return columns_[index];
  }

  // Series.to_numpy(float): integers widen, strings parse like
  // pandas.to_numeric(errors="coerce").
  [[nodiscard]] Doubles doubles(std::string_view name) const {
    const Column& values = column(name);
    if (const auto* reals = std::get_if<Doubles>(&values)) {
      return *reals;
    }
    if (const auto* integers = std::get_if<Integers>(&values)) {
      return Doubles(integers->begin(), integers->end());
    }
    Doubles out;
    for (const auto& text : std::get<Strings>(values)) {
      out.push_back(parse_pandas_number(text));
    }
    return out;
  }

  [[nodiscard]] const Strings& strings(std::string_view name) const {
    const auto* texts = std::get_if<Strings>(&column(name));
    if (texts == nullptr) {
      throw std::invalid_argument(fmt::format("column '{}' does not hold strings", name));
    }
    return *texts;
  }

  // df[name] = values: replaces an existing column in place, else appends.
  void set(const std::string& name, Column values) {
    check_length(name, values);
    const std::size_t index = find(name);
    if (index != kNpos) {
      columns_[index] = std::move(values);
      return;
    }
    names_.push_back(name);
    columns_.push_back(std::move(values));
  }

  // DataFrame.insert(position, name, values)
  void insert(std::size_t position, const std::string& name, Column values) {
    if (has(name)) {
      throw std::invalid_argument(fmt::format("cannot insert {}, already exists", name));
    }
    check_length(name, values);
    position = std::min(position, names_.size());
    names_.insert(names_.begin() + static_cast<std::ptrdiff_t>(position), name);
    columns_.insert(columns_.begin() + static_cast<std::ptrdiff_t>(position), std::move(values));
  }

  void rename(const std::string& from, const std::string& to) {
    const std::size_t index = find(from);
    if (index == kNpos) {
      throw std::out_of_range(fmt::format("table has no column '{}'", from));
    }
    names_[index] = to;
  }

  // df.iloc[rows]
  [[nodiscard]] Frame take(std::span<const std::size_t> rows) const {
    Frame out;
    out.rows_ = rows.size();
    for (std::size_t c = 0; c < names_.size(); ++c) {
      out.names_.push_back(names_[c]);
      out.columns_.push_back(std::visit(
          [&](const auto& values) -> Column {
            std::decay_t<decltype(values)> picked;
            picked.reserve(rows.size());
            for (std::size_t row : rows) {
              picked.push_back(values.at(row));
            }
            return picked;
          },
          columns_[c]));
    }
    return out;
  }

  // df[names]
  [[nodiscard]] Frame select(std::span<const std::string> names) const {
    Frame out;
    out.rows_ = rows_;
    for (const auto& name : names) {
      out.names_.push_back(name);
      out.columns_.push_back(column(name));
    }
    return out;
  }

  // DataFrame.to_csv(index=False)
  [[nodiscard]] std::string to_pandas_csv() const {
    const auto dialect = CsvDialect::pandas();
    std::string out;
    append_csv_header(out, names_, dialect);
    std::vector<Cell> cells(columns_.size());
    for (std::size_t row = 0; row < rows_; ++row) {
      for (std::size_t c = 0; c < columns_.size(); ++c) {
        cells[c] = std::visit([&](const auto& values) -> Cell { return values[row]; }, columns_[c]);
      }
      append_csv_record(out, cells, dialect);
    }
    return out;
  }

 private:
  static constexpr std::size_t kNpos = static_cast<std::size_t>(-1);

  [[nodiscard]] std::size_t find(std::string_view name) const noexcept {
    for (std::size_t i = 0; i < names_.size(); ++i) {
      if (names_[i] == name) {
        return i;
      }
    }
    return kNpos;
  }

  static std::size_t length(const Column& values) {
    return std::visit([](const auto& v) { return v.size(); }, values);
  }

  void check_length(const std::string& name, const Column& values) {
    const std::size_t n = length(values);
    if (names_.empty() && rows_ == 0) {
      rows_ = n;
      return;
    }
    if (n != rows_) {
      throw std::invalid_argument(
          fmt::format("column '{}' has {} values but the table has {} rows", name, n, rows_));
    }
  }

  std::vector<std::string> names_;
  std::vector<Column> columns_;
  std::size_t rows_ = 0;
};

// dtype inference of pandas.read_csv's C engine: int64, then float64, then
// bool (written back as True/False), otherwise str with NA fields empty.
[[nodiscard]] inline Frame::Column infer_pandas_column(const std::vector<std::string>& cells) {
  if (cells.empty()) {
    return Frame::Strings{};
  }
  Frame::Integers integers;
  integers.reserve(cells.size());
  for (const auto& cell : cells) {
    const auto value = pandas_int64_field(cell);
    if (!value) {
      break;
    }
    integers.push_back(*value);
  }
  if (integers.size() == cells.size()) {
    return integers;
  }
  Frame::Doubles reals;
  reals.reserve(cells.size());
  for (const auto& cell : cells) {
    const auto value = pandas_float_field(cell);
    if (!value) {
      break;
    }
    reals.push_back(*value);
  }
  if (reals.size() == cells.size()) {
    return reals;
  }
  auto bool_word = [](std::string_view text) -> std::optional<bool> {
    if (text == "True" || text == "TRUE" || text == "true") {
      return true;
    }
    if (text == "False" || text == "FALSE" || text == "false") {
      return false;
    }
    return std::nullopt;
  };
  const bool all_bool = std::all_of(cells.begin(), cells.end(), [&](const auto& c) { return bool_word(c).has_value(); });
  Frame::Strings texts;
  texts.reserve(cells.size());
  for (const auto& cell : cells) {
    if (all_bool) {
      texts.emplace_back(py_bool(*bool_word(cell)));
    } else {
      texts.push_back(is_pandas_na(cell) ? std::string() : cell);
    }
  }
  return texts;
}

// pandas.read_csv(path) as a Frame.
[[nodiscard]] inline Frame frame_from_csv(const CsvTable& table) {
  Frame frame;
  for (std::size_t c = 0; c < table.columns.size(); ++c) {
    std::vector<std::string> cells;
    cells.reserve(table.rows.size());
    for (const auto& row : table.rows) {
      cells.push_back(c < row.size() ? row[c] : std::string());
    }
    frame.set(table.columns[c], infer_pandas_column(cells));
  }
  return frame;
}

// pandas.DataFrame(list_of_dicts): columns in order of first appearance; a
// column with any str or bool value becomes str, with float or missing values
// float64, otherwise int64.
[[nodiscard]] inline Frame frame_from_records(std::span<const Record> records) {
  std::vector<std::string> names;
  for (const auto& record : records) {
    for (const auto& item : record.items()) {
      if (std::find(names.begin(), names.end(), item.first) == names.end()) {
        names.push_back(item.first);
      }
    }
  }
  Frame frame;
  for (const auto& name : names) {
    bool textual = false;
    bool real = false;
    bool missing = false;
    for (const auto& record : records) {
      const auto cell = record.get(name);
      if (!cell || std::holds_alternative<std::monostate>(*cell)) {
        missing = true;
      } else if (std::holds_alternative<std::string>(*cell) || std::holds_alternative<bool>(*cell)) {
        textual = true;
      } else if (std::holds_alternative<double>(*cell)) {
        real = true;
      }
    }
    if (textual) {
      Frame::Strings texts;
      for (const auto& record : records) {
        const Cell cell = record.get(name).value_or(Cell{});
        if (const auto* text = std::get_if<std::string>(&cell)) {
          texts.push_back(*text);
        } else if (const auto* flag = std::get_if<bool>(&cell)) {
          texts.emplace_back(py_bool(*flag));
        } else if (const auto* integer = std::get_if<std::int64_t>(&cell)) {
          texts.push_back(std::to_string(*integer));
        } else if (const auto* value = std::get_if<double>(&cell)) {
          texts.push_back(std::isnan(*value) ? std::string() : py_float_repr(*value));
        } else {
          texts.emplace_back();
        }
      }
      frame.set(name, std::move(texts));
    } else if (real || missing) {
      Frame::Doubles values;
      for (const auto& record : records) {
        const Cell cell = record.get(name).value_or(Cell{});
        if (const auto* integer = std::get_if<std::int64_t>(&cell)) {
          values.push_back(static_cast<double>(*integer));
        } else if (const auto* value = std::get_if<double>(&cell)) {
          values.push_back(*value);
        } else {
          values.push_back(std::numeric_limits<double>::quiet_NaN());
        }
      }
      frame.set(name, std::move(values));
    } else {
      Frame::Integers values;
      for (const auto& record : records) {
        values.push_back(std::get<std::int64_t>(*record.get(name)));
      }
      frame.set(name, std::move(values));
    }
  }
  return frame;
}

}  // namespace guiding::table

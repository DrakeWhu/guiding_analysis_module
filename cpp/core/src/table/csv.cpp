#include "guiding/table/csv.hpp"

#include <fast_float/fast_float.h>
#include <fmt/format.h>

#include <atomic>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <system_error>

#ifndef _WIN32
#include <unistd.h>
#endif

#include "guiding/table/py_format.hpp"

namespace guiding::table {
namespace {

bool needs_quotes(std::string_view field, const CsvDialect& dialect) {
  for (char c : field) {
    if (c == ',' || c == '"' || c == '\r' || c == '\n' ||
        dialect.line_terminator.find(c) != std::string::npos) {
      return true;
    }
  }
  return false;
}

void append_field(std::string& out, std::string_view field, const CsvDialect& dialect) {
  if (!needs_quotes(field, dialect)) {
    out.append(field);
    return;
  }
  out.push_back('"');
  for (char c : field) {
    if (c == '"') {
      out.push_back('"');
    }
    out.push_back(c);
  }
  out.push_back('"');
}

std::string_view trim_ascii_space(std::string_view text) {
  const auto first = text.find_first_not_of(" \t\r\n");
  if (first == std::string_view::npos) {
    return {};
  }
  const auto last = text.find_last_not_of(" \t\r\n");
  return text.substr(first, last - first + 1);
}

}  // namespace

void append_csv_record(std::string& out, std::span<const Cell> cells, const CsvDialect& dialect) {
  if (cells.size() == 1 && std::holds_alternative<std::string>(cells[0]) &&
      std::get<std::string>(cells[0]).empty()) {
    // csv.writer quotes a lone empty field so the row is not read back as blank.
    out += "\"\"";
    out += dialect.line_terminator;
    return;
  }
  bool first = true;
  for (const Cell& cell : cells) {
    if (!first) {
      out.push_back(',');
    }
    first = false;
    if (const auto* integer = std::get_if<std::int64_t>(&cell)) {
      out += std::to_string(*integer);
    } else if (const auto* real = std::get_if<double>(&cell)) {
      if (std::isnan(*real) && dialect.nan_as_empty) {
        continue;
      }
      append_py_float_repr(out, *real);
    } else if (const auto* flag = std::get_if<bool>(&cell)) {
      out += py_bool(*flag);
    } else if (const auto* text = std::get_if<std::string>(&cell)) {
      append_field(out, *text, dialect);
    }
  }
  out += dialect.line_terminator;
}

void append_csv_header(std::string& out, std::span<const std::string> columns, const CsvDialect& dialect) {
  std::vector<Cell> cells(columns.begin(), columns.end());
  append_csv_record(out, cells, dialect);
}

void write_file_atomically(const std::filesystem::path& path, std::string_view content) {
  static std::atomic<unsigned long> counter{0};
  if (path.has_parent_path()) {
    std::filesystem::create_directories(path.parent_path());
  }
#ifndef _WIN32
  const auto pid = static_cast<unsigned long>(::getpid());
#else
  const unsigned long pid = 0;
#endif
  auto temporary = path;
  temporary += fmt::format(".tmp.{}.{}", pid, counter.fetch_add(1));
  {
    std::ofstream stream(temporary, std::ios::binary | std::ios::trunc);
    if (!stream) {
      throw std::runtime_error(fmt::format("cannot open {} for writing", temporary.string()));
    }
    stream.write(content.data(), static_cast<std::streamsize>(content.size()));
    stream.flush();
    if (!stream) {
      throw std::runtime_error(fmt::format("failed writing {}", temporary.string()));
    }
  }
  std::error_code error;
  std::filesystem::rename(temporary, path, error);
  if (error) {
    std::filesystem::remove(temporary);
    throw std::runtime_error(fmt::format("cannot move {} into place: {}", path.string(), error.message()));
  }
}

std::optional<std::size_t> CsvTable::column_index(std::string_view name) const {
  for (std::size_t i = 0; i < columns.size(); ++i) {
    if (columns[i] == name) {
      return i;
    }
  }
  return std::nullopt;
}

std::vector<double> CsvTable::numeric_column(std::string_view name) const {
  const auto index = column_index(name);
  if (!index) {
    throw std::out_of_range(fmt::format("CSV has no column '{}'", name));
  }
  std::vector<double> values;
  values.reserve(rows.size());
  for (const auto& row : rows) {
    const std::string_view cell = *index < row.size() ? std::string_view(row[*index]) : std::string_view();
    values.push_back(parse_number(cell).value_or(std::numeric_limits<double>::quiet_NaN()));
  }
  return values;
}

std::vector<std::string> CsvTable::string_column(std::string_view name) const {
  const auto index = column_index(name);
  if (!index) {
    throw std::out_of_range(fmt::format("CSV has no column '{}'", name));
  }
  std::vector<std::string> values;
  values.reserve(rows.size());
  for (const auto& row : rows) {
    values.push_back(*index < row.size() ? row[*index] : std::string());
  }
  return values;
}

std::optional<double> parse_number(std::string_view text) {
  text = trim_ascii_space(text);
  if (text.empty()) {
    return std::nullopt;
  }
  if (text.front() == '+') {
    text.remove_prefix(1);
  }
  double value = 0.0;
  const auto result = fast_float::from_chars(text.data(), text.data() + text.size(), value);
  if (result.ec != std::errc() || result.ptr != text.data() + text.size()) {
    return std::nullopt;
  }
  return value;
}

CsvTable parse_csv(std::string_view text) {
  if (text.size() >= 3 && static_cast<unsigned char>(text[0]) == 0xEF &&
      static_cast<unsigned char>(text[1]) == 0xBB && static_cast<unsigned char>(text[2]) == 0xBF) {
    text.remove_prefix(3);
  }

  std::vector<std::vector<std::string>> records;
  std::vector<std::string> record;
  std::string field;
  bool in_quotes = false;
  bool field_started = false;

  auto end_field = [&] {
    record.push_back(std::move(field));
    field.clear();
    field_started = false;
  };
  auto end_record = [&] {
    end_field();
    if (!(record.size() == 1 && record.front().empty())) {
      records.push_back(std::move(record));
    }
    record.clear();
  };

  for (std::size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    if (in_quotes) {
      if (c == '"') {
        if (i + 1 < text.size() && text[i + 1] == '"') {
          field.push_back('"');
          ++i;
        } else {
          in_quotes = false;
        }
      } else {
        field.push_back(c);
      }
      continue;
    }
    if (c == '"' && !field_started) {
      in_quotes = true;
      field_started = true;
    } else if (c == ',') {
      end_field();
    } else if (c == '\r') {
      if (i + 1 < text.size() && text[i + 1] == '\n') {
        ++i;
      }
      end_record();
    } else if (c == '\n') {
      end_record();
    } else {
      field.push_back(c);
      field_started = true;
    }
  }
  if (in_quotes) {
    throw std::runtime_error("CSV ends inside a quoted field");
  }
  if (field_started || !field.empty() || !record.empty()) {
    end_record();
  }

  CsvTable table;
  if (records.empty()) {
    return table;
  }
  table.columns = std::move(records.front());
  table.rows.assign(std::make_move_iterator(records.begin() + 1), std::make_move_iterator(records.end()));
  return table;
}

CsvTable read_csv_file(const std::filesystem::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error(fmt::format("cannot open {}", path.string()));
  }
  std::string content((std::istreambuf_iterator<char>(stream)), std::istreambuf_iterator<char>());
  return parse_csv(content);
}

}  // namespace guiding::table

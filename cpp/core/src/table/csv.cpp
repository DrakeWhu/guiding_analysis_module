#include "guiding/table/csv.hpp"

#include <fast_float/fast_float.h>
#include <fmt/format.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cctype>
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

constexpr bool is_space_ascii(char c) {
  return c == ' ' || static_cast<unsigned>(c) - static_cast<unsigned>('\t') < 5U;
}

constexpr bool is_digit_ascii(char c) {
  return static_cast<unsigned>(c) - static_cast<unsigned>('0') < 10U;
}

bool equals_ignore_case(std::string_view a, std::string_view b) {
  return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) {
           return std::tolower(static_cast<unsigned char>(x)) == std::tolower(static_cast<unsigned char>(y));
         });
}

// The e[] table of pandas' tokenizer: correctly rounded literals 1e0 ... 1e308.
const std::array<double, 309>& powers_of_ten() {
  static const std::array<double, 309> powers = [] {
    std::array<double, 309> table{};
    for (std::size_t i = 0; i < table.size(); ++i) {
      const std::string literal = fmt::format("1e{}", i);
      fast_float::from_chars(literal.data(), literal.data() + literal.size(), table[i]);
    }
    return table;
  }();
  return powers;
}

// precise_xstrtod from pandas/_libs/src/parser/tokenizer.c (pandas 3.0.5) with
// decimal='.', sci='E', no thousands separator and skip_trailing=1. Returns
// nullopt where pandas reports an error or leaves characters unparsed.
std::optional<double> precise_xstrtod(std::string_view text) {
  constexpr long kMaxDigits = 17;
  const char* p = text.data();
  const char* const end = text.data() + text.size();

  while (p < end && is_space_ascii(*p)) {
    ++p;
  }
  bool negative = false;
  if (p < end && (*p == '-' || *p == '+')) {
    negative = *p == '-';
    ++p;
  }

  double number = 0.0;
  long exponent = 0;
  long num_digits = 0;
  long num_decimals = 0;

  while (p < end && is_digit_ascii(*p)) {
    if (num_digits < kMaxDigits) {
      number = number * 10. + (*p - '0');
      ++num_digits;
    } else {
      ++exponent;
    }
    ++p;
  }
  if (p < end && *p == '.') {
    ++p;
    while (num_digits < kMaxDigits && p < end && is_digit_ascii(*p)) {
      number = number * 10. + (*p - '0');
      ++p;
      ++num_digits;
      ++num_decimals;
    }
    if (num_digits >= kMaxDigits) {
      while (p < end && is_digit_ascii(*p)) {
        ++p;
      }
    }
    exponent -= num_decimals;
  }
  if (num_digits == 0) {
    return std::nullopt;
  }
  if (negative) {
    number = -number;
  }

  if (p < end && (*p == 'e' || *p == 'E')) {
    ++p;
    // long n = strtol(p, &tmp_ptr, 10)
    const char* q = p;
    while (q < end && is_space_ascii(*q)) {
      ++q;
    }
    bool exponent_negative = false;
    if (q < end && (*q == '+' || *q == '-')) {
      exponent_negative = *q == '-';
      ++q;
    }
    const char* const digits_begin = q;
    long n = 0;
    bool overflow = false;
    while (q < end && is_digit_ascii(*q)) {
      const long digit = *q - '0';
      if (!overflow && n > (std::numeric_limits<long>::max() - digit) / 10) {
        overflow = true;
      }
      if (!overflow) {
        n = n * 10 + digit;
      }
      ++q;
    }
    if (q == digits_begin) {
      --p;  // no exponent digits: un-consume the 'e'
    } else {
      if (overflow) {
        exponent = exponent_negative ? std::numeric_limits<long>::min() : std::numeric_limits<long>::max();
      } else {
        n = exponent_negative ? -n : n;
        const bool add_overflows = (n > 0 && exponent > std::numeric_limits<long>::max() - n) ||
                                   (n < 0 && exponent < std::numeric_limits<long>::min() - n);
        exponent = add_overflows ? n : exponent + n;
      }
      p = q;
    }
  }

  const auto& e = powers_of_ten();
  if (exponent > 308) {
    number = number == 0 ? 0.0 : (number < 0 ? -HUGE_VAL : HUGE_VAL);
  } else if (exponent > 0) {
    number *= e[static_cast<std::size_t>(exponent)];
  } else if (exponent < -308) {
    if (exponent < -616) {
      number = 0.;
    } else {
      number /= e[static_cast<std::size_t>(-308 - exponent)];
      number /= e[308];
    }
  } else {
    number /= e[static_cast<std::size_t>(-exponent)];
  }

  // pandas 3.0.5 returns +-HUGE_VAL on overflow without reporting an error.
  while (p < end && is_space_ascii(*p)) {
    ++p;
  }
  if (p != end) {
    return std::nullopt;
  }
  return number;
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
    values.push_back(parse_pandas_number(cell));
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

bool is_pandas_na(std::string_view text) {
  static constexpr std::array<std::string_view, 19> kDefaultNaValues = {
      "",     "#N/A", "#N/A N/A", "#NA", "-1.#IND", "-1.#QNAN", "-NaN", "-nan", "1.#IND", "1.#QNAN",
      "<NA>", "N/A",  "NA",       "NULL", "NaN",    "None",     "n/a",  "nan",  "null"};
  return std::find(kDefaultNaValues.begin(), kDefaultNaValues.end(), text) != kDefaultNaValues.end();
}

std::optional<double> pandas_float_field(std::string_view text) {
  constexpr double kInf = std::numeric_limits<double>::infinity();
  if (is_pandas_na(text)) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (const auto value = precise_xstrtod(text)) {
    return value;
  }
  if (equals_ignore_case(text, "inf") || equals_ignore_case(text, "+inf") || equals_ignore_case(text, "infinity") ||
      equals_ignore_case(text, "+infinity")) {
    return kInf;
  }
  if (equals_ignore_case(text, "-inf") || equals_ignore_case(text, "-infinity")) {
    return -kInf;
  }
  return std::nullopt;
}

double parse_pandas_number(std::string_view text) {
  return pandas_float_field(text).value_or(std::numeric_limits<double>::quiet_NaN());
}

std::optional<std::int64_t> pandas_int64_field(std::string_view text) {
  const char* p = text.data();
  const char* const end = text.data() + text.size();
  while (p < end && is_space_ascii(*p)) {
    ++p;
  }
  bool negative = false;
  if (p < end && (*p == '-' || *p == '+')) {
    negative = *p == '-';
    ++p;
  }
  if (p == end || !is_digit_ascii(*p)) {
    return std::nullopt;
  }
  // Accumulate as a negative magnitude so INT64_MIN stays representable.
  std::int64_t value = 0;
  const std::int64_t limit = std::numeric_limits<std::int64_t>::min();
  while (p < end && is_digit_ascii(*p)) {
    const int digit = *p - '0';
    if (value < (limit + digit) / 10) {
      return std::nullopt;
    }
    value = value * 10 - digit;
    ++p;
  }
  while (p < end && is_space_ascii(*p)) {
    ++p;
  }
  if (p != end) {
    return std::nullopt;
  }
  if (!negative) {
    if (value == limit) {
      return std::nullopt;
    }
    value = -value;
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

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace guiding::table {

// How a writer in the Python reference renders cells.
struct CsvDialect {
  std::string line_terminator;
  bool nan_as_empty = false;

  // csv.DictWriter defaults (metrics.py, particles.py, campaign.py): CRLF rows,
  // floats via repr, NaN as "nan".
  static CsvDialect python_csv() { return {"\r\n", false}; }
  // pandas.DataFrame.to_csv defaults on POSIX: LF rows, NaN as an empty field.
  static CsvDialect pandas() { return {"\n", true}; }
};

// None, int, float, bool, str — the value types the reference writes.
using Cell = std::variant<std::monostate, std::int64_t, double, bool, std::string>;

// Appends one record using csv.QUOTE_MINIMAL rules.
void append_csv_record(std::string& out, std::span<const Cell> cells, const CsvDialect& dialect);
void append_csv_header(std::string& out, std::span<const std::string> columns, const CsvDialect& dialect);

// Writes through a temporary sibling and renames it into place, so concurrent
// readers on shared file systems never observe a partially written file.
void write_file_atomically(const std::filesystem::path& path, std::string_view content);

class CsvTable {
 public:
  std::vector<std::string> columns;
  std::vector<std::vector<std::string>> rows;

  [[nodiscard]] std::optional<std::size_t> column_index(std::string_view name) const;
  [[nodiscard]] bool has_column(std::string_view name) const { return column_index(name).has_value(); }
  // Values as pandas sees them after read_csv + to_numeric(errors="coerce"),
  // see parse_pandas_number.
  [[nodiscard]] std::vector<double> numeric_column(std::string_view name) const;
  [[nodiscard]] std::vector<std::string> string_column(std::string_view name) const;
};

// RFC 4180 parsing (quoted fields, CRLF or LF); a UTF-8 BOM is dropped.
[[nodiscard]] CsvTable parse_csv(std::string_view text);
[[nodiscard]] CsvTable read_csv_file(const std::filesystem::path& path);

// Correctly rounded full-string parse ("nan", "inf", exponents); nullopt otherwise.
[[nodiscard]] std::optional<double> parse_number(std::string_view text);

// One cell of pandas.read_csv (C engine, default float_precision) followed by
// pandas.to_numeric(errors="coerce"): default NA strings give NaN, numbers go
// through pandas' precise_xstrtod (17 significant digits, not always correctly
// rounded), inf-like words give infinities, anything else NaN.
[[nodiscard]] double parse_pandas_number(std::string_view text);

// pandas' default na_values ("", "NA", "nan", "null", ...).
[[nodiscard]] bool is_pandas_na(std::string_view text);

// The C parser's float conversion of one field: NaN for NA strings, nullopt
// when the field is not a number (the column would not be float64).
[[nodiscard]] std::optional<double> pandas_float_field(std::string_view text);

// pandas' str_to_int64 for one field (surrounding spaces allowed); nullopt when
// the field is not an integer that fits in int64.
[[nodiscard]] std::optional<std::int64_t> pandas_int64_field(std::string_view text);

}  // namespace guiding::table

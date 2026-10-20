#include "guiding/table/py_format.hpp"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string_view>

#include <fast_float/fast_float.h>

namespace guiding::table {

void append_py_float_repr(std::string& out, double value) {
  if (std::isnan(value)) {
    out += "nan";
    return;
  }
  if (std::isinf(value)) {
    out += value < 0 ? "-inf" : "inf";
    return;
  }

  // Shortest round-trip digits in scientific form: [-]d[.ddd]e(+|-)XX
  char buffer[64];
  const auto result = std::to_chars(buffer, buffer + sizeof(buffer), value, std::chars_format::scientific);
  std::string_view text(buffer, static_cast<std::size_t>(result.ptr - buffer));

  if (text.front() == '-') {
    out.push_back('-');
    text.remove_prefix(1);
  }
  const std::size_t e_pos = text.find('e');
  const std::string_view mantissa = text.substr(0, e_pos);
  const std::string_view exponent_text = text.substr(e_pos + 1);

  int exponent = 0;
  const char* exponent_begin = exponent_text.data() + (exponent_text.front() == '+' ? 1 : 0);
  std::from_chars(exponent_begin, exponent_text.data() + exponent_text.size(), exponent);

  std::string digits;
  digits.reserve(mantissa.size());
  for (char c : mantissa) {
    if (c != '.') {
      digits.push_back(c);
    }
  }
  if (digits == "0") {
    out += "0.0";
    return;
  }

  // Python's float_repr_style 'short': decpt counts digits before the point.
  const int decpt = exponent + 1;
  const int n_digits = static_cast<int>(digits.size());
  if (decpt <= -4 || decpt > 16) {
    out.push_back(digits.front());
    if (n_digits > 1) {
      out.push_back('.');
      out.append(digits, 1, std::string::npos);
    }
    const int e10 = decpt - 1;
    out.push_back('e');
    out.push_back(e10 < 0 ? '-' : '+');
    const int magnitude = std::abs(e10);
    if (magnitude < 10) {
      out.push_back('0');
    }
    out += std::to_string(magnitude);
  } else if (decpt <= 0) {
    out += "0.";
    out.append(static_cast<std::size_t>(-decpt), '0');
    out += digits;
  } else if (decpt >= n_digits) {
    out += digits;
    out.append(static_cast<std::size_t>(decpt - n_digits), '0');
    out += ".0";
  } else {
    out.append(digits, 0, static_cast<std::size_t>(decpt));
    out.push_back('.');
    out.append(digits, static_cast<std::size_t>(decpt), std::string::npos);
  }
}

std::string py_float_repr(double value) {
  std::string out;
  append_py_float_repr(out, value);
  return out;
}

std::string py_str_repr(std::string_view text) {
  const bool use_double = text.find('\'') != std::string_view::npos && text.find('"') == std::string_view::npos;
  const char quote = use_double ? '"' : '\'';
  std::string out(1, quote);
  for (char c : text) {
    if (c == '\\' || c == quote) {
      out.push_back('\\');
      out.push_back(c);
    } else if (c == '\n') {
      out += "\\n";
    } else if (c == '\t') {
      out += "\\t";
    } else if (c == '\r') {
      out += "\\r";
    } else {
      out.push_back(c);
    }
  }
  out.push_back(quote);
  return out;
}

std::string python_list_repr(std::span<const std::string> items) {
  std::string out = "[";
  for (std::size_t i = 0; i < items.size(); ++i) {
    if (i > 0) {
      out += ", ";
    }
    out += py_str_repr(items[i]);
  }
  return out + "]";
}

std::string python_path_string(const std::filesystem::path& path) {
  const std::string text = path.generic_string();
  if (text.empty()) {
    return ".";
  }
  std::string out = text.front() == '/' ? "/" : "";
  bool first = true;
  std::size_t start = 0;
  while (start <= text.size()) {
    const std::size_t end = text.find('/', start);
    const std::string_view part(text.data() + start, (end == std::string::npos ? text.size() : end) - start);
    if (!part.empty() && part != ".") {
      if (!first) {
        out.push_back('/');
      }
      out.append(part);
      first = false;
    }
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return out.empty() ? "." : out;
}

std::string python_float_list_repr(std::span<const double> values) {
  std::string out = "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (i != 0) {
      out += ", ";
    }
    append_py_float_repr(out, values[i]);
  }
  out += "]";
  return out;
}

namespace {

std::string_view strip_py_whitespace(std::string_view text) {
  const auto first = text.find_first_not_of(" \t\n\r\v\f");
  if (first == std::string_view::npos) {
    return {};
  }
  return text.substr(first, text.find_last_not_of(" \t\n\r\v\f") - first + 1);
}

bool iequals(std::string_view a, std::string_view b) {
  return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) {
           return std::tolower(static_cast<unsigned char>(x)) == std::tolower(static_cast<unsigned char>(y));
         });
}

bool is_digit(char c) { return c >= '0' && c <= '9'; }

// Removes "_" separators; nullopt unless every "_" sits between two digits.
std::optional<std::string> remove_digit_separators(std::string_view text) {
  std::string out;
  out.reserve(text.size());
  for (std::size_t i = 0; i < text.size(); ++i) {
    if (text[i] == '_') {
      if (i == 0 || i + 1 == text.size() || !is_digit(text[i - 1]) || !is_digit(text[i + 1])) {
        return std::nullopt;
      }
      continue;
    }
    out.push_back(text[i]);
  }
  return out;
}

}  // namespace

std::optional<double> parse_py_float(std::string_view text) {
  text = strip_py_whitespace(text);
  const auto cleaned = remove_digit_separators(text);
  if (!cleaned || cleaned->empty()) {
    return std::nullopt;
  }
  std::string_view body = *cleaned;
  bool negative = false;
  if (body.front() == '+' || body.front() == '-') {
    negative = body.front() == '-';
    body.remove_prefix(1);
  }
  if (iequals(body, "inf") || iequals(body, "infinity")) {
    return negative ? -std::numeric_limits<double>::infinity() : std::numeric_limits<double>::infinity();
  }
  if (iequals(body, "nan")) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  // Only digits, one '.', and an exponent remain valid; fast_float would also
  // accept "nan(...)" and "infinity" spellings that were handled above.
  if (body.empty() || !(is_digit(body.front()) || body.front() == '.')) {
    return std::nullopt;
  }
  double value = 0.0;
  const auto result = fast_float::from_chars(body.data(), body.data() + body.size(), value);
  if (result.ec != std::errc() || result.ptr != body.data() + body.size()) {
    return std::nullopt;
  }
  return negative ? -value : value;
}

std::optional<std::int64_t> parse_py_int(std::string_view text) {
  text = strip_py_whitespace(text);
  const auto cleaned = remove_digit_separators(text);
  if (!cleaned || cleaned->empty()) {
    return std::nullopt;
  }
  std::string_view body = *cleaned;
  const bool negative = body.front() == '-';
  if (body.front() == '+' || body.front() == '-') {
    body.remove_prefix(1);
  }
  if (body.empty() || !std::all_of(body.begin(), body.end(), is_digit)) {
    return std::nullopt;
  }
  std::string digits = negative ? "-" : "";
  digits.append(body);
  std::int64_t value = 0;
  const auto result = std::from_chars(digits.data(), digits.data() + digits.size(), value);
  if (result.ec != std::errc() || result.ptr != digits.data() + digits.size()) {
    return std::nullopt;
  }
  return value;
}

}  // namespace guiding::table

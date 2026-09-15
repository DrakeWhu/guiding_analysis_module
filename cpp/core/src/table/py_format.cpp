#include "guiding/table/py_format.hpp"

#include <charconv>
#include <cmath>
#include <cstdlib>
#include <string_view>

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

}  // namespace guiding::table

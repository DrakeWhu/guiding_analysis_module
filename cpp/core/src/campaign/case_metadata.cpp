#include "guiding/campaign/case_metadata.hpp"

#include <cctype>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "guiding/table/csv.hpp"
#include "guiding/table/py_format.hpp"

namespace guiding::campaign {
namespace {

bool is_separator(char c) {
  return c == '\\' || c == '/' || c == 's' || c == 'S' || c == '_' || c == '-';
}

bool is_digit(char c) { return c >= '0' && c <= '9'; }

std::size_t digit_run(std::string_view text, std::size_t pos) {
  std::size_t end = pos;
  while (end < text.size() && is_digit(text[end])) {
    ++end;
  }
  return end - pos;
}

// Tries to match L<digits>([p.]<digits>)?mm followed by end or separator at pos
// (which holds 'L' or 'l'); returns the matched number text.
std::optional<std::string_view> match_at(std::string_view text, std::size_t pos) {
  std::size_t cursor = pos + 1;
  const std::size_t int_digits = digit_run(text, cursor);
  if (int_digits == 0) {
    return std::nullopt;
  }
  cursor += int_digits;

  auto accept_suffix = [&](std::size_t number_end) -> bool {
    if (number_end + 2 > text.size()) {
      return false;
    }
    if (std::tolower(static_cast<unsigned char>(text[number_end])) != 'm' ||
        std::tolower(static_cast<unsigned char>(text[number_end + 1])) != 'm') {
      return false;
    }
    const std::size_t after = number_end + 2;
    return after == text.size() || is_separator(text[after]);
  };

  // The optional fraction is greedy; regex backtracking retries without it.
  if (cursor < text.size() && (text[cursor] == 'p' || text[cursor] == 'P' || text[cursor] == '.')) {
    const std::size_t frac_digits = digit_run(text, cursor + 1);
    if (frac_digits > 0 && accept_suffix(cursor + 1 + frac_digits)) {
      return text.substr(pos + 1, cursor + frac_digits - pos);
    }
  }
  if (accept_suffix(cursor)) {
    return text.substr(pos + 1, cursor - pos - 1);
  }
  return std::nullopt;
}

}  // namespace

std::optional<double> infer_plateau_length_mm_from_text(std::string_view text) {
  for (std::size_t pos = 0; pos < text.size(); ++pos) {
    if (text[pos] != 'L' && text[pos] != 'l') {
      continue;
    }
    if (pos > 0 && !is_separator(text[pos - 1])) {
      continue;
    }
    if (const auto token = match_at(text, pos)) {
      // Python: float(token.replace("p", ".")) — an uppercase 'P' makes float() fail.
      std::string number(*token);
      for (char& c : number) {
        if (c == 'p') {
          c = '.';
        }
      }
      const auto value = table::parse_number(number);
      if (!value || number.find('P') != std::string::npos) {
        throw std::invalid_argument(fmt::format("could not convert string to float: '{}'", number));
      }
      return value;
    }
  }
  return std::nullopt;
}

std::pair<double, double> plateau_window_from_length_mm(double plateau_length_mm, double ramp_up_mm) {
  if (plateau_length_mm <= 0.0) {
    throw std::invalid_argument(fmt::format("plateau_length_mm must be > 0, got {}", plateau_length_mm));
  }
  return {ramp_up_mm, ramp_up_mm + plateau_length_mm};
}

std::optional<std::pair<double, double>> infer_plateau_window_mm_from_text(std::string_view text,
                                                                          double ramp_up_mm) {
  const auto length = infer_plateau_length_mm_from_text(text);
  if (!length) {
    return std::nullopt;
  }
  return plateau_window_from_length_mm(*length, ramp_up_mm);
}

std::optional<std::pair<double, double>> infer_plateau_window_mm_from_sources(std::span<const std::string> sources,
                                                                             double ramp_up_mm) {
  std::vector<std::pair<double, double>> windows;
  for (const auto& source : sources) {
    if (const auto window = infer_plateau_window_mm_from_text(source, ramp_up_mm)) {
      windows.push_back(*window);
    }
  }
  if (windows.empty()) {
    return std::nullopt;
  }
  // Python compares {(round(start, 9), round(end, 9))}.
  auto round9 = [](double value) { return table::parse_number(fmt::format("{:.9f}", value)).value_or(value); };
  std::set<std::pair<double, double>> unique;
  for (const auto& [start, end] : windows) {
    unique.emplace(round9(start), round9(end));
  }
  if (unique.size() != 1) {
    std::string listed;
    for (const auto& [start, end] : unique) {
      listed += fmt::format("{}({}, {})", listed.empty() ? "" : ", ", table::py_float_repr(start),
                            table::py_float_repr(end));
    }
    throw std::invalid_argument(fmt::format("Inconsistent plateau windows inferred from triplet sources: [{}]", listed));
  }
  return windows.front();
}

}  // namespace guiding::campaign

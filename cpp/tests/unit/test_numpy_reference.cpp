#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <vector>

#include "guiding/campaign/case_metadata.hpp"
#include "guiding/numeric/npcompat.hpp"
#include "guiding/table/csv.hpp"
#include "guiding/table/py_format.hpp"

namespace {

namespace np = guiding::np;

struct ReferenceLine {
  std::string kind;
  std::string input;
  std::string expected;
};

const std::vector<ReferenceLine>& reference_lines() {
  static const std::vector<ReferenceLine> lines = [] {
    std::ifstream stream(std::string(GUIDING_TEST_DATA_DIR) + "/numpy_reference.tsv");
    REQUIRE(stream.good());
    std::vector<ReferenceLine> parsed;
    std::string line;
    while (std::getline(stream, line)) {
      if (line.empty() || line.front() == '#') {
        continue;
      }
      const auto first = line.find('\t');
      const auto second = line.find('\t', first + 1);
      REQUIRE(second != std::string::npos);
      parsed.push_back({line.substr(0, first), line.substr(first + 1, second - first - 1), line.substr(second + 1)});
    }
    return parsed;
  }();
  return lines;
}

std::vector<double> numbers(const std::string& text) {
  std::vector<double> values;
  std::size_t start = 0;
  while (start < text.size()) {
    auto end = text.find(' ', start);
    if (end == std::string::npos) {
      end = text.size();
    }
    const auto value = guiding::table::parse_number(std::string_view(text).substr(start, end - start));
    REQUIRE(value.has_value());
    values.push_back(*value);
    start = end + 1;
  }
  return values;
}

bool same_bits(double a, double b) { return std::memcmp(&a, &b, sizeof(double)) == 0; }

double parse_hex_float(std::string text) {
  if (text == "nan") {
    return std::numeric_limits<double>::quiet_NaN();
  }
  if (text == "inf") {
    return std::numeric_limits<double>::infinity();
  }
  if (text == "-inf") {
    return -std::numeric_limits<double>::infinity();
  }
  bool negative = false;
  if (text.front() == '-') {
    negative = true;
    text.erase(0, 1);
  }
  REQUIRE(text.rfind("0x", 0) == 0);
  text.erase(0, 2);
  double value = 0.0;
  const auto result = std::from_chars(text.data(), text.data() + text.size(), value, std::chars_format::hex);
  REQUIRE(result.ec == std::errc());
  return negative ? -value : value;
}

std::vector<float> to_float32(const std::vector<double>& values) {
  return std::vector<float>(values.begin(), values.end());
}

}  // namespace

TEST_CASE("pairwise sums and float32 means match numpy bit for bit", "[numpy]") {
  std::size_t checked = 0;
  for (const auto& line : reference_lines()) {
    if (line.kind != "sum64" && line.kind != "sum32" && line.kind != "meansq32") {
      continue;
    }
    INFO(line.kind << " with " << numbers(line.input).size() << " values");
    const auto input = numbers(line.input);
    const double expected = numbers(line.expected).at(0);
    if (line.kind == "sum64") {
      CHECK(same_bits(np::pairwise_sum<double>(input), expected));
    } else if (line.kind == "sum32") {
      CHECK(same_bits(static_cast<double>(np::pairwise_sum<float>(to_float32(input))), expected));
    } else {
      auto values = to_float32(input);
      for (float& v : values) {
        v = v * v;
      }
      CHECK(same_bits(static_cast<double>(np::mean<float>(values)), expected));
    }
    ++checked;
  }
  CHECK(checked > 30);
}

TEST_CASE("median, linspace, moving average and rounding match numpy", "[numpy]") {
  std::size_t checked = 0;
  for (const auto& line : reference_lines()) {
    INFO(line.kind);
    if (line.kind == "median") {
      CHECK(same_bits(np::median(numbers(line.input)), numbers(line.expected).at(0)));
    } else if (line.kind.rfind("movavg", 0) == 0) {
      const auto window = static_cast<std::size_t>(std::stoul(line.kind.substr(6)));
      const auto actual = np::moving_average_same(numbers(line.input), window);
      const auto expected = numbers(line.expected);
      REQUIRE(actual.size() == expected.size());
      for (std::size_t i = 0; i < actual.size(); ++i) {
        CHECK(same_bits(actual[i], expected[i]));
      }
    } else if (line.kind == "linspace") {
      const auto args = numbers(line.input);
      const auto actual = np::linspace(args.at(0), args.at(1), static_cast<std::size_t>(args.at(2)));
      const auto expected = numbers(line.expected);
      REQUIRE(actual.size() == expected.size());
      for (std::size_t i = 0; i < actual.size(); ++i) {
        CHECK(same_bits(actual[i], expected[i]));
      }
    } else if (line.kind == "round") {
      CHECK(np::round_half_even(numbers(line.input).at(0)) == std::stoll(line.expected));
    } else {
      continue;
    }
    ++checked;
  }
  CHECK(checked > 20);
}

TEST_CASE("argmax and argmin follow numpy NaN and tie rules", "[numpy]") {
  const double nan = std::numeric_limits<double>::quiet_NaN();
  const std::vector<double> ties{1.0, 3.0, 3.0, 2.0};
  CHECK(np::argmax<double>(ties) == 1);
  const std::vector<double> with_nan{1.0, nan, 3.0, nan};
  CHECK(np::argmax<double>(with_nan) == 1);
  CHECK(np::argmin<double>(with_nan) == 1);
  CHECK(std::isnan(np::max_value<double>(with_nan)));
  const std::vector<double> zeros{-0.0, 0.0, 1.0};
  CHECK(np::argmin<double>(zeros) == 0);
}

TEST_CASE("float repr matches Python", "[format]") {
  std::size_t checked = 0;
  for (const auto& line : reference_lines()) {
    if (line.kind != "repr") {
      continue;
    }
    INFO(line.input);
    CHECK(guiding::table::py_float_repr(parse_hex_float(line.input)) == line.expected);
    ++checked;
  }
  CHECK(checked > 1000);
}

TEST_CASE("plateau token parsing matches cap_guiding.case_metadata", "[metadata]") {
  std::size_t checked = 0;
  for (const auto& line : reference_lines()) {
    if (line.kind != "plateau") {
      continue;
    }
    INFO(line.input);
    if (line.expected == "ValueError") {
      CHECK_THROWS(guiding::campaign::infer_plateau_length_mm_from_text(line.input));
    } else if (line.expected == "None") {
      CHECK_FALSE(guiding::campaign::infer_plateau_length_mm_from_text(line.input).has_value());
    } else {
      const auto value = guiding::campaign::infer_plateau_length_mm_from_text(line.input);
      REQUIRE(value.has_value());
      CHECK(same_bits(*value, numbers(line.expected).at(0)));
    }
    ++checked;
  }
  CHECK(checked > 10);
}

TEST_CASE("CSV numbers are parsed exactly like pandas.read_csv + to_numeric", "[pandas]") {
  std::size_t checked = 0;
  std::size_t not_correctly_rounded = 0;
  for (const auto& line : reference_lines()) {
    if (line.kind != "pandasfloat") {
      continue;
    }
    INFO("input '" << line.input << "' expected " << line.expected);
    const double expected = numbers(line.expected).at(0);
    const double actual = guiding::table::parse_pandas_number(line.input);
    if (std::isnan(expected)) {
      CHECK(std::isnan(actual));
    } else {
      CHECK(same_bits(actual, expected));
      const auto exact = guiding::table::parse_number(line.input);
      if (exact && !same_bits(*exact, expected)) {
        ++not_correctly_rounded;
      }
    }
    ++checked;
  }
  CHECK(checked > 3000);
  // The reference set is meant to exercise pandas' non-correctly-rounded paths.
  CHECK(not_correctly_rounded > 0);
}

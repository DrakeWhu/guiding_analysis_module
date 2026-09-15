#include "guiding/numeric/npcompat.hpp"

#include <algorithm>
#include <cmath>

namespace guiding::np {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

}  // namespace

double std_dev(std::span<const double> values) {
  if (values.empty()) {
    return kNaN;
  }
  const double count = static_cast<double>(values.size());
  const double mean_value = pairwise_sum(values) / count;
  std::vector<double> squared(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    const double centered = values[i] - mean_value;
    squared[i] = centered * centered;
  }
  return std::sqrt(pairwise_sum<double>(squared) / count);
}

std::vector<double> linspace(double start, double stop, std::size_t num) {
  std::vector<double> values(num);
  if (num == 0) {
    return values;
  }
  if (num == 1) {
    values[0] = start;
    return values;
  }
  const double div = static_cast<double>(num - 1);
  const double delta = stop - start;
  const double step = delta / div;
  if (step == 0.0) {
    // numpy's denormal-safe branch: (i / div) * delta
    for (std::size_t i = 0; i < num; ++i) {
      values[i] = (static_cast<double>(i) / div) * delta + start;
    }
  } else {
    for (std::size_t i = 0; i < num; ++i) {
      values[i] = static_cast<double>(i) * step + start;
    }
  }
  values[num - 1] = stop;
  return values;
}

std::vector<double> diff(std::span<const double> values) {
  if (values.size() < 2) {
    return {};
  }
  std::vector<double> out(values.size() - 1);
  for (std::size_t i = 0; i + 1 < values.size(); ++i) {
    out[i] = values[i + 1] - values[i];
  }
  return out;
}

double median(std::span<const double> values) {
  if (values.empty()) {
    return kNaN;
  }
  std::vector<double> sorted(values.begin(), values.end());
  if (std::any_of(sorted.begin(), sorted.end(), [](double v) { return std::isnan(v); })) {
    return kNaN;
  }
  std::sort(sorted.begin(), sorted.end());
  const std::size_t n = sorted.size();
  // numpy takes the mean of the middle slice: identity 0.0 plus the elements.
  if (n % 2 == 1) {
    return (0.0 + sorted[n / 2]) / 1.0;
  }
  return ((0.0 + sorted[n / 2 - 1]) + sorted[n / 2]) / 2.0;
}

double nanmedian(std::span<const double> values) {
  std::vector<double> finite;
  finite.reserve(values.size());
  for (double v : values) {
    if (!std::isnan(v)) {
      finite.push_back(v);
    }
  }
  return median(finite);
}

std::vector<double> moving_average_same(std::span<const double> y, std::size_t window) {
  if (window == 0) {
    throw std::invalid_argument("moving average window must be positive");
  }
  if (window > y.size()) {
    throw std::invalid_argument("moving average window is longer than the signal");
  }
  const double coefficient = 1.0 / static_cast<double>(window);
  const auto n = static_cast<std::ptrdiff_t>(y.size());
  const auto left = static_cast<std::ptrdiff_t>(window / 2);
  std::vector<double> out(y.size());
  for (std::ptrdiff_t m = 0; m < n; ++m) {
    double acc = 0.0;
    for (std::ptrdiff_t j = 0; j < static_cast<std::ptrdiff_t>(window); ++j) {
      const std::ptrdiff_t t = m - left + j;
      if (t >= 0 && t < n) {
        acc += y[static_cast<std::size_t>(t)] * coefficient;
      }
    }
    out[static_cast<std::size_t>(m)] = acc;
  }
  return out;
}

std::int64_t round_half_even(double value) {
  if (!std::isfinite(value)) {
    throw std::invalid_argument("cannot round a non-finite value to an integer");
  }
  return static_cast<std::int64_t>(std::nearbyint(value));
}

}  // namespace guiding::np

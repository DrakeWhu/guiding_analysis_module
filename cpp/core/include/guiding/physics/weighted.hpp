#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <vector>

#include "guiding/numeric/npcompat.hpp"

// Weighted statistics shared by the particle modules, written to follow the
// numpy expressions of cap_guiding step by step.
namespace guiding::physics {

using Mask = std::vector<char>;

[[nodiscard]] inline std::size_t count(const Mask& mask) {
  return static_cast<std::size_t>(std::count(mask.begin(), mask.end(), char{1}));
}

[[nodiscard]] inline bool any(const Mask& mask) {
  return std::find(mask.begin(), mask.end(), char{1}) != mask.end();
}

// values[mask]
[[nodiscard]] inline std::vector<double> gather(std::span<const double> values, const Mask& mask) {
  std::vector<double> out;
  out.reserve(count(mask));
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (mask[i] != 0) {
      out.push_back(values[i]);
    }
  }
  return out;
}

// np.average(values, weights=weights): sum(values * weights) / sum(weights)
[[nodiscard]] inline double weighted_average(std::span<const double> values, std::span<const double> weights) {
  std::vector<double> product(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    product[i] = values[i] * weights[i];
  }
  const double scale = np::pairwise_sum(weights);
  if (scale == 0.0) {
    throw std::domain_error("Weights sum to zero, can't be normalized");
  }
  return np::pairwise_sum<double>(product) / scale;
}

// np.average((a - mean_a) * (b - mean_b), weights=weights)
[[nodiscard]] inline double weighted_covariance(std::span<const double> a, std::span<const double> b,
                                                std::span<const double> weights) {
  const double mean_a = weighted_average(a, weights);
  const double mean_b = weighted_average(b, weights);
  std::vector<double> product(a.size());
  for (std::size_t i = 0; i < a.size(); ++i) {
    product[i] = (a[i] - mean_a) * (b[i] - mean_b);
  }
  return weighted_average(product, weights);
}

// The cumulative-weight percentile used across cap_guiding: finite values with
// positive weights, sorted by value, first index whose cumulative weight
// reaches p/100 of the total (searchsorted side="left"), no interpolation.
[[nodiscard]] inline double weighted_percentile(std::span<const double> values, std::span<const double> weights,
                                                double percentile) {
  std::vector<std::size_t> kept;
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (std::isfinite(values[i]) && std::isfinite(weights[i]) && weights[i] > 0.0) {
      kept.push_back(i);
    }
  }
  if (kept.empty()) {
    return std::numeric_limits<double>::quiet_NaN();
  }
  std::stable_sort(kept.begin(), kept.end(), [&](std::size_t a, std::size_t b) { return values[a] < values[b]; });
  std::vector<double> cumulative(kept.size());
  double running = 0.0;
  for (std::size_t i = 0; i < kept.size(); ++i) {
    running = i == 0 ? weights[kept[i]] : running + weights[kept[i]];
    cumulative[i] = running;
  }
  const double target = percentile / 100.0 * cumulative.back();
  const auto position = std::lower_bound(cumulative.begin(), cumulative.end(), target) - cumulative.begin();
  return values[kept[static_cast<std::size_t>(position)]];
}

}  // namespace guiding::physics

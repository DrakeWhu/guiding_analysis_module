#pragma once

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

// Kernels that reproduce the numpy expressions used by the Python reference
// pipeline bit for bit (verified against numpy 2.4; see
// tools/golden/make_numpy_reference.py). Build with -ffp-contract=off.
namespace guiding::np {

// np.sum / np.add.reduce over a 1-D contiguous float array: numpy's pairwise
// summation (8 unrolled accumulators up to 128 elements, halving above).
template <typename T>
[[nodiscard]] T pairwise_sum(const T* data, std::size_t n) {
  if (n < 8) {
    T result = T(0);
    for (std::size_t i = 0; i < n; ++i) {
      result += data[i];
    }
    return result;
  }
  if (n <= 128) {
    T r[8];
    for (std::size_t k = 0; k < 8; ++k) {
      r[k] = data[k];
    }
    std::size_t i = 8;
    for (; i < n - (n % 8); i += 8) {
      for (std::size_t k = 0; k < 8; ++k) {
        r[k] += data[i + k];
      }
    }
    T result = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
    for (; i < n; ++i) {
      result += data[i];
    }
    return result;
  }
  std::size_t half = n / 2;
  half -= half % 8;
  return pairwise_sum(data, half) + pairwise_sum(data + half, n - half);
}

template <typename T>
[[nodiscard]] T pairwise_sum(std::span<const T> values) {
  return pairwise_sum(values.data(), values.size());
}

// np.mean of a 1-D contiguous array: pairwise sum divided by the count in T.
template <typename T>
[[nodiscard]] T mean(std::span<const T> values) {
  if (values.empty()) {
    return std::numeric_limits<T>::quiet_NaN();
  }
  return pairwise_sum(values) / static_cast<T>(values.size());
}

// np.std(x) with ddof=0: sqrt(sum((x - mean)**2) / n), sums pairwise.
[[nodiscard]] double std_dev(std::span<const double> values);

// np.linspace(start, stop, num) with endpoint=True.
[[nodiscard]] std::vector<double> linspace(double start, double stop, std::size_t num);

// np.diff of a 1-D array.
[[nodiscard]] std::vector<double> diff(std::span<const double> values);

// np.median of a 1-D array (NaN if empty or if any element is NaN).
[[nodiscard]] double median(std::span<const double> values);

// np.nanmedian of a 1-D array (NaN if every element is NaN).
[[nodiscard]] double nanmedian(std::span<const double> values);

// np.convolve(y, np.ones(window) / window, mode="same") for window <= len(y).
[[nodiscard]] std::vector<double> moving_average_same(std::span<const double> y, std::size_t window);

// Python round(x) for a float: ties to even. Throws for NaN or infinity.
[[nodiscard]] std::int64_t round_half_even(double value);

// libm pow(base, exponent), as Python float ** and numpy scalar ** call it.
// Kept out of line with a runtime exponent so the compiler cannot rewrite
// pow(x, 2.0) as x * x (glibc pow is not guaranteed to equal x * x).
[[nodiscard]] double c_pow(double base, double exponent);

// np.argmax / np.argmin: first occurrence; the first NaN wins when present.
template <typename T>
[[nodiscard]] std::size_t argmax(std::span<const T> values) {
  if (values.empty()) {
    throw std::invalid_argument("attempt to get argmax of an empty sequence");
  }
  std::size_t best = 0;
  if (std::isnan(values[0])) {
    return 0;
  }
  for (std::size_t i = 1; i < values.size(); ++i) {
    if (std::isnan(values[i])) {
      return i;
    }
    if (values[i] > values[best]) {
      best = i;
    }
  }
  return best;
}

template <typename T>
[[nodiscard]] std::size_t argmin(std::span<const T> values) {
  if (values.empty()) {
    throw std::invalid_argument("attempt to get argmin of an empty sequence");
  }
  std::size_t best = 0;
  if (std::isnan(values[0])) {
    return 0;
  }
  for (std::size_t i = 1; i < values.size(); ++i) {
    if (std::isnan(values[i])) {
      return i;
    }
    if (values[i] < values[best]) {
      best = i;
    }
  }
  return best;
}

// np.max / np.min: NaN when any element is NaN.
template <typename T>
[[nodiscard]] T max_value(std::span<const T> values) {
  return values[argmax(values)];
}

template <typename T>
[[nodiscard]] T min_value(std::span<const T> values) {
  return values[argmin(values)];
}

// np.maximum(a, b) for scalars: NaN-propagating, returns a on ties.
[[nodiscard]] inline double maximum(double a, double b) noexcept {
  return (a >= b || std::isnan(a)) ? a : b;
}

}  // namespace guiding::np

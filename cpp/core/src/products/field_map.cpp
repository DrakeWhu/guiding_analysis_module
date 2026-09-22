#include "guiding/products/field_map.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <tuple>

#include "guiding/io/thetamode.hpp"
#include "guiding/numeric/npcompat.hpp"

namespace guiding::products {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

std::size_t ceil_div(std::size_t a, std::size_t b) { return (a + b - 1) / b; }

}  // namespace

PooledGrid max_pool(std::span<const double> values, std::size_t rows, std::size_t cols, std::size_t out_rows,
                    std::size_t out_cols) {
  if (values.size() != rows * cols) {
    throw std::invalid_argument("max_pool: grid size mismatch");
  }
  if (rows == 0 || cols == 0) {
    return {};
  }
  out_rows = std::clamp<std::size_t>(out_rows, 1, std::max<std::size_t>(rows, 1));
  out_cols = std::clamp<std::size_t>(out_cols, 1, std::max<std::size_t>(cols, 1));
  const std::size_t block_r = ceil_div(rows, out_rows);
  const std::size_t block_c = ceil_div(cols, out_cols);
  const std::size_t result_rows = ceil_div(rows, block_r);
  const std::size_t result_cols = ceil_div(cols, block_c);
  PooledGrid out{std::vector<double>(result_rows * result_cols, kNaN), result_rows, result_cols};
  for (std::size_t i = 0; i < rows; ++i) {
    for (std::size_t j = 0; j < cols; ++j) {
      const double value = values[i * cols + j];
      double& cell = out.values[(i / block_r) * result_cols + j / block_c];
      if (!std::isnan(value) && (std::isnan(cell) || value > cell)) {
        cell = value;
      }
    }
  }
  return out;
}

FieldMap load_field_map(const std::filesystem::path& file_path, std::int64_t iteration, const FieldMapOptions& options) {
  auto file = h5::File::open_read_only(file_path);
  file.set_raw_reads_enabled(options.raw_reads);

  io::PositiveRSlice ex;
  io::PositiveRSlice ey;
  {
    const auto e_r = io::read_thetamode_component(file, iteration, "E", "r");
    const auto e_t = io::read_thetamode_component(file, iteration, "E", "t");
    std::tie(ex, ey) = io::cartesian_xy(e_r, e_t);
  }
  const auto ez = io::component_slice(io::read_thetamode_component(file, iteration, "E", "z"));

  FieldMap map;
  map.iteration = iteration;
  map.time_fs = std::isfinite(ex.time_s) ? ex.time_s * 1.0e15 : kNaN;
  map.laser = physics::weighted_laser_metrics(ex, ey, options.params.smooth_um, options.params.lambda0_m);
  map.wake = physics::wake_metrics(ez, map.laser.z_peak_um, options.params.wake_behind_um, options.params.wake_gap_um);

  const std::size_t n_rows = ex.n_rows;
  const std::size_t n_z = ex.n_z;
  std::vector<double> intensity(n_rows * n_z);
  for (std::size_t k = 0; k < intensity.size(); ++k) {
    intensity[k] = ex.values[k] * ex.values[k] + ey.values[k] * ey.values[k];
  }

  // Lineouts (metrics.weighted_laser_metrics).
  map.z_um = ex.z_um;
  map.I_z.assign(n_z, 0.0);
  for (std::size_t i = 0; i < n_rows; ++i) {
    const double weight = std::max(ex.r_um[i] * 1.0e-6, 0.0);
    for (std::size_t j = 0; j < n_z; ++j) {
      map.I_z[j] += intensity[i * n_z + j] * weight;
    }
  }
  const double dz_um = n_z > 1 ? np::median(np::diff(ex.z_um)) : 1.0;
  const std::int64_t smooth_cells = std::max<std::int64_t>(1, np::round_half_even(options.params.smooth_um / dz_um));
  map.I_z_smooth = smooth_cells <= 1 || static_cast<std::size_t>(smooth_cells) > n_z
                       ? map.I_z
                       : np::moving_average_same(map.I_z, static_cast<std::size_t>(smooth_cells));

  // Rows sorted by r (the slice keeps openpmd-viewer's signed order).
  std::vector<std::size_t> order(n_rows);
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return ex.r_um[a] < ex.r_um[b]; });
  map.r_um.reserve(n_rows);
  map.P_r.reserve(n_rows);
  for (std::size_t i : order) {
    double profile = 0.0;
    for (std::size_t j = 0; j < n_z; ++j) {
      if (std::abs(ex.z_um[j] - map.laser.z_peak_um) <= 5.0) {
        profile += intensity[i * n_z + j];
      }
    }
    map.r_um.push_back(ex.r_um[i]);
    map.P_r.push_back(profile);
  }

  std::size_t axis_row = 0;
  for (std::size_t i = 1; i < ez.n_rows; ++i) {
    if (std::abs(ez.r_um[i]) < std::abs(ez.r_um[axis_row])) {
      axis_row = i;
    }
  }
  map.Ez_axis_GVm.resize(ez.n_z);
  for (std::size_t j = 0; j < ez.n_z; ++j) {
    map.Ez_axis_GVm[j] = ez.values[axis_row * ez.n_z + j] / 1.0e9;
  }

  // Image: largest r on top.
  std::vector<double> image(n_rows * n_z);
  for (std::size_t out = 0; out < n_rows; ++out) {
    const std::size_t source = order[n_rows - 1 - out];
    std::copy_n(intensity.begin() + static_cast<std::ptrdiff_t>(source * n_z), n_z,
                image.begin() + static_cast<std::ptrdiff_t>(out * n_z));
  }
  auto pooled = max_pool(image, n_rows, n_z, options.max_r_cells, options.max_z_cells);
  map.intensity = std::move(pooled.values);
  map.rows = pooled.rows;
  map.cols = pooled.cols;
  map.intensity_max = 0.0;
  for (double value : map.intensity) {
    if (std::isfinite(value)) {
      map.intensity_max = std::max(map.intensity_max, value);
    }
  }
  map.z_min_um = np::min_value<double>(ex.z_um);
  map.z_max_um = np::max_value<double>(ex.z_um);
  map.r_min_um = map.r_um.empty() ? 0.0 : map.r_um.front();
  map.r_max_um = map.r_um.empty() ? 0.0 : map.r_um.back();
  return map;
}

}  // namespace guiding::products

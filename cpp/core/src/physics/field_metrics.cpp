#include "guiding/physics/field_metrics.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "guiding/numeric/npcompat.hpp"

namespace guiding::physics {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

}  // namespace

double eperp_peak_to_a0(double eperp_peak_vm, double lambda0_m) {
  if (!std::isfinite(eperp_peak_vm) || eperp_peak_vm <= 0.0) {
    return kNaN;
  }
  if (!std::isfinite(lambda0_m) || lambda0_m <= 0.0) {
    return kNaN;
  }
  const double omega0 = 2.0 * kPi * kSpeedOfLightMPerS / lambda0_m;
  return kElementaryChargeC * eperp_peak_vm / (kElectronMassKg * kSpeedOfLightMPerS * omega0);
}

LaserMetrics weighted_laser_metrics(const io::PositiveRSlice& ex, const io::PositiveRSlice& ey, double smooth_um,
                                    double lambda0_m) {
  if (ex.n_rows != ey.n_rows || ex.n_z != ey.n_z) {
    throw std::runtime_error("Ex and Ey slices have different shapes");
  }
  const std::size_t n_rows = ex.n_rows;
  const std::size_t n_z = ex.n_z;
  if (n_rows == 0 || n_z == 0) {
    throw std::runtime_error("empty positive-r field slice");
  }

  // I = Ex**2 + Ey**2 ; radial_weight = np.maximum(r_m, 0.0)
  std::vector<double> intensity(n_rows * n_z);
  for (std::size_t i = 0; i < intensity.size(); ++i) {
    intensity[i] = ex.values[i] * ex.values[i] + ey.values[i] * ey.values[i];
  }
  std::vector<double> r_m(n_rows);
  std::vector<double> weight(n_rows);
  for (std::size_t i = 0; i < n_rows; ++i) {
    r_m[i] = ex.r_um[i] * 1.0e-6;
    weight[i] = np::maximum(r_m[i], 0.0);
  }
  std::vector<double> z_m(n_z);
  for (std::size_t j = 0; j < n_z; ++j) {
    z_m[j] = ex.z_um[j] * 1.0e-6;
  }

  // I_z = np.sum(I * w[:, None], axis=0): numpy accumulates row by row here.
  std::vector<double> weighted(n_rows * n_z);
  std::vector<double> i_z(n_z, 0.0);
  for (std::size_t i = 0; i < n_rows; ++i) {
    for (std::size_t j = 0; j < n_z; ++j) {
      const double value = intensity[i * n_z + j] * weight[i];
      weighted[i * n_z + j] = value;
      i_z[j] += value;
    }
  }

  const double dz_um = np::median(np::diff(ex.z_um));
  const std::int64_t smooth_cells = std::max<std::int64_t>(1, np::round_half_even(smooth_um / dz_um));
  const std::vector<double> i_z_smooth =
      smooth_cells <= 1 ? i_z : np::moving_average_same(i_z, static_cast<std::size_t>(smooth_cells));

  const std::size_t j_peak = np::argmax<double>(i_z_smooth);
  const double z_peak_um = ex.z_um[j_peak];

  const double z_window_um = 5.0;
  std::vector<std::size_t> window;
  for (std::size_t j = 0; j < n_z; ++j) {
    if (std::abs(ex.z_um[j] - z_peak_um) <= z_window_um) {
      window.push_back(j);
    }
  }
  if (window.empty()) {
    window.push_back(j_peak);
  }

  // profile = np.sum(I[:, z_mask], axis=1): accumulated column by column.
  std::vector<double> profile_weight(n_rows);
  std::vector<double> second_moment(n_rows);
  for (std::size_t i = 0; i < n_rows; ++i) {
    double profile = 0.0;
    for (std::size_t j : window) {
      profile += intensity[i * n_z + j];
    }
    profile_weight[i] = profile * weight[i];
    second_moment[i] = (r_m[i] * r_m[i]) * profile * weight[i];
  }

  const double denom = np::pairwise_sum<double>(profile_weight);
  double waist_um = kNaN;
  if (denom > 0) {
    const double r2_mean = np::pairwise_sum<double>(second_moment) / denom;
    waist_um = std::sqrt(2.0 * r2_mean) * 1.0e6;
  }

  const double dr_m = np::median(np::diff(r_m));
  const double dz_m = np::median(np::diff(z_m));
  // np.sum over the whole C-contiguous array is one pairwise pass.
  const double energy_proxy = np::pairwise_sum<double>(weighted) * dr_m * dz_m;

  const double peak_i_proxy = np::max_value<double>(intensity);
  const double eperp_peak_vm = std::sqrt(peak_i_proxy);
  const double a0_peak = eperp_peak_to_a0(eperp_peak_vm, lambda0_m);

  return LaserMetrics{
      .z_peak_um = z_peak_um,
      .front_margin_um = np::max_value<double>(ex.z_um) - z_peak_um,
      .back_margin_um = z_peak_um - np::min_value<double>(ex.z_um),
      .waist_um = waist_um,
      .peak_I_proxy = peak_i_proxy,
      .Eperp_peak_Vm = eperp_peak_vm,
      .a0_peak = a0_peak,
      .energy_proxy = energy_proxy,
  };
}

WakeMetrics wake_metrics(const io::PositiveRSlice& ez, double z_peak_um, double wake_behind_um, double wake_gap_um) {
  if (ez.n_rows == 0) {
    throw std::runtime_error("empty positive-r Ez slice");
  }
  std::vector<double> abs_r(ez.n_rows);
  for (std::size_t i = 0; i < ez.n_rows; ++i) {
    abs_r[i] = std::abs(ez.r_um[i]);
  }
  const std::size_t i_axis = np::argmin<double>(abs_r);

  const double z1 = z_peak_um - wake_behind_um;
  const double z2 = z_peak_um - wake_gap_um;
  std::vector<double> values;
  std::vector<double> z_values;
  for (std::size_t j = 0; j < ez.n_z; ++j) {
    const double z = ez.z_um[j];
    if (z >= z1 && z <= z2) {
      values.push_back(ez.values[i_axis * ez.n_z + j]);
      z_values.push_back(z);
    }
  }
  if (values.empty()) {
    return WakeMetrics{kNaN, kNaN, kNaN, kNaN, kNaN, kNaN};
  }

  std::vector<double> magnitudes(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    magnitudes[i] = std::abs(values[i]);
  }
  const std::size_t k = np::argmax<double>(magnitudes);

  double rms = kNaN;
  if (ez.float32) {
    // Ez_wake stays float32: square, mean and sqrt all run in float32.
    std::vector<float> squared(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
      const auto v = static_cast<float>(values[i]);
      squared[i] = v * v;
    }
    rms = static_cast<double>(std::sqrt(np::mean<float>(squared)));
  } else {
    std::vector<double> squared(values.size());
    for (std::size_t i = 0; i < values.size(); ++i) {
      squared[i] = values[i] * values[i];
    }
    rms = std::sqrt(np::mean<double>(squared));
  }

  return WakeMetrics{
      .Ez_wake_max = np::max_value<double>(values),
      .Ez_wake_min = np::min_value<double>(values),
      .Ez_wake_absmax = np::max_value<double>(magnitudes),
      .Ez_wake_rms = rms,
      .z_Ez_absmax_um = z_values[k],
      .z_Ez_absmax_rel_um = z_values[k] - z_peak_um,
  };
}

}  // namespace guiding::physics

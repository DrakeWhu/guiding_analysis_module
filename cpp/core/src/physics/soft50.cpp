#include "guiding/physics/soft50.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

#include "guiding/numeric/npcompat.hpp"
#include "guiding/physics/field_metrics.hpp"
#include "guiding/physics/weighted.hpp"

namespace guiding::physics {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Python max(value, 0.0)
double at_least_zero(double value) { return (0.0 > value) ? 0.0 : value; }

double normalized_emittance_um_rad(std::span<const double> position_m, std::span<const double> momentum,
                                   std::span<const double> weights) {
  const double q_var = weighted_covariance(position_m, position_m, weights);
  const double u_var = weighted_covariance(momentum, momentum, weights);
  const double qu_cov = weighted_covariance(position_m, momentum, weights);
  double determinant = q_var * u_var - qu_cov * qu_cov;
  if (determinant < 0.0 && std::abs(determinant) <= 1.0e-24) {
    determinant = 0.0;
  }
  return std::sqrt(at_least_zero(determinant)) * 1.0e6;
}

void set_nan(table::Record& row, std::initializer_list<const char*> keys) {
  for (const char* key : keys) {
    row.set(key, kNaN);
  }
}

}  // namespace

void Soft50Config::validate() const {
  const std::array<double, 4> values{energy_low_mev, energy_target_mev, reliability_floor, effective_count_reference};
  if (!std::all_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); })) {
    throw std::invalid_argument("Soft50Config values must be finite");
  }
  if (energy_low_mev < 0.0) {
    throw std::invalid_argument("energy_low_mev must be non-negative");
  }
  if (energy_target_mev <= energy_low_mev) {
    throw std::invalid_argument("energy_target_mev must be greater than energy_low_mev");
  }
  if (!(0.0 <= reliability_floor && reliability_floor <= 1.0)) {
    throw std::invalid_argument("reliability_floor must be within [0, 1]");
  }
  if (effective_count_reference <= 0.0) {
    throw std::invalid_argument("effective_count_reference must be positive");
  }
}

std::vector<double> smooth_energy_acceptance(std::span<const double> energy_mev, double energy_low_mev,
                                             double energy_target_mev) {
  if (!std::isfinite(energy_low_mev) || !std::isfinite(energy_target_mev) || energy_target_mev <= energy_low_mev) {
    throw std::invalid_argument("energy acceptance requires finite target > low");
  }
  std::vector<double> accepted(energy_mev.size());
  for (std::size_t i = 0; i < energy_mev.size(); ++i) {
    const double ratio = (energy_mev[i] - energy_low_mev) / (energy_target_mev - energy_low_mev);
    // np.clip(ratio, 0, 1) keeps NaN and returns the bound on ties.
    const double lower = std::isnan(ratio) ? ratio : ((ratio > 0.0) ? ratio : 0.0);
    const double u = std::isnan(lower) ? lower : ((lower < 1.0) ? lower : 1.0);
    accepted[i] = std::isfinite(energy_mev[i]) ? 3.0 * u * u - 2.0 * u * u * u : 0.0;
  }
  return accepted;
}

double effective_sample_size(std::span<const double> weights) {
  std::vector<double> selected;
  for (double w : weights) {
    if (std::isfinite(w) && w > 0.0) {
      selected.push_back(w);
    }
  }
  if (selected.empty()) {
    return 0.0;
  }
  std::vector<double> squared(selected.size());
  for (std::size_t i = 0; i < selected.size(); ++i) {
    squared[i] = selected[i] * selected[i];
  }
  const double denominator = np::pairwise_sum<double>(squared);
  if (denominator <= 0.0) {
    return 0.0;
  }
  return np::c_pow(np::pairwise_sum<double>(selected), 2.0) / denominator;
}

double soft_reliability(double n_effective, double reliability_floor, double effective_count_reference) {
  const double n_eff = at_least_zero(n_effective);
  if (!std::isfinite(n_eff) || !std::isfinite(reliability_floor) || !std::isfinite(effective_count_reference)) {
    throw std::invalid_argument("reliability inputs must be finite");
  }
  if (!(0.0 <= reliability_floor && reliability_floor <= 1.0) || effective_count_reference <= 0.0) {
    throw std::invalid_argument("invalid reliability floor/reference");
  }
  return reliability_floor + (1.0 - reliability_floor) * (1.0 - std::exp(-n_eff / effective_count_reference));
}

table::Record summarize_soft50_metrics(const io::ParticleDump& dump, std::span<const double> energy,
                                       const Soft50Config& config, Longitudinal longitudinal, bool forward_only,
                                       std::optional<double> exit_window_mm) {
  config.validate();
  const std::size_t n = dump.size();
  const auto& weights = dump.w;
  const auto& q_long = longitudinal == Longitudinal::Z ? dump.z_m : longitudinal == Longitudinal::X ? dump.x_m : dump.y_m;
  const auto& u_long = longitudinal == Longitudinal::Z ? dump.uz : longitudinal == Longitudinal::X ? dump.ux : dump.uy;

  Mask base(n);
  for (std::size_t i = 0; i < n; ++i) {
    bool ok = std::isfinite(energy[i]) && std::isfinite(weights[i]) && weights[i] > 0.0 && std::isfinite(q_long[i]) &&
              std::isfinite(u_long[i]);
    if (forward_only) {
      ok = ok && u_long[i] > 0.0;
    }
    base[i] = ok ? 1 : 0;
  }
  if (exit_window_mm) {
    const double window_m = *exit_window_mm * 1.0e-3;
    if (!std::isfinite(window_m) || window_m <= 0.0) {
      throw std::invalid_argument("exit_window_mm must be finite and positive");
    }
    if (any(base)) {
      const double q_max = np::max_value<double>(gather(q_long, base));
      for (std::size_t i = 0; i < n; ++i) {
        base[i] = (base[i] != 0 && q_long[i] >= q_max - window_m) ? 1 : 0;
      }
    }
  }

  const auto acceptance = smooth_energy_acceptance(energy, config.energy_low_mev, config.energy_target_mev);
  std::vector<double> soft_weights(n);
  Mask soft(n);
  Mask hard(n);
  for (std::size_t i = 0; i < n; ++i) {
    soft_weights[i] = base[i] != 0 ? weights[i] * acceptance[i] : 0.0;
    soft[i] = soft_weights[i] > 0.0 ? 1 : 0;
    hard[i] = (base[i] != 0 && energy[i] >= config.energy_target_mev) ? 1 : 0;
  }

  const double weight_soft = np::pairwise_sum<double>(soft_weights);
  const double hard_weight = any(hard) ? np::pairwise_sum<double>(gather(weights, hard)) : 0.0;
  const double n_effective = effective_sample_size(soft_weights);
  const double reliability = soft_reliability(n_effective, config.reliability_floor, config.effective_count_reference);

  table::Record row;
  row.set("soft50_schema_version", std::string(kSoft50SchemaVersion));
  row.set("soft50_energy_low_MeV", config.energy_low_mev);
  row.set("soft50_energy_target_MeV", config.energy_target_mev);
  row.set("soft50_reliability_floor", config.reliability_floor);
  row.set("soft50_effective_count_reference", config.effective_count_reference);
  row.set("soft50_status", std::string(any(soft) ? "ok" : "no_accepted_particles"));
  row.set("n_macroparticles_soft50", static_cast<std::int64_t>(count(soft)));
  row.set("weight_soft50", weight_soft);
  row.set("charge_soft50_pC", weight_soft * kElementaryChargeC / 1.0e-12);
  row.set("n_effective_soft50", n_effective);
  row.set("reliability_soft50", reliability);
  row.set("charge_Ege50MeV_pC", hard_weight * kElementaryChargeC / 1.0e-12);
  row.set("n_macroparticles_Ege50MeV", static_cast<std::int64_t>(count(hard)));

  if (!any(soft)) {
    set_nan(row, {"energy_mean_soft50_MeV", "energy_spread_rms_soft50_MeV", "energy_relative_spread_rms_soft50",
                  "energy_p10_soft50_MeV", "energy_p50_soft50_MeV", "energy_p90_soft50_MeV", "energy_p95_soft50_MeV",
                  "theta_r_p90_soft50_mrad", "theta_r_p95_soft50_mrad", "emitn_x_soft50_um_rad",
                  "emitn_y_soft50_um_rad", "emitn_xy_soft50_um_rad"});
    return row;
  }

  const auto selected_energy = gather(energy, soft);
  const auto selected_weights = gather(soft_weights, soft);
  const double energy_mean = weighted_average(selected_energy, selected_weights);
  std::vector<double> squared_deviation(selected_energy.size());
  for (std::size_t i = 0; i < selected_energy.size(); ++i) {
    const double deviation = selected_energy[i] - energy_mean;
    squared_deviation[i] = deviation * deviation;
  }
  const double energy_variance = weighted_average(squared_deviation, selected_weights);
  const double energy_spread_rms = std::sqrt(at_least_zero(energy_variance));

  row.set("energy_mean_soft50_MeV", energy_mean);
  row.set("energy_spread_rms_soft50_MeV", energy_spread_rms);
  row.set("energy_relative_spread_rms_soft50", energy_mean > 0.0 ? energy_spread_rms / energy_mean : kNaN);
  row.set("energy_p10_soft50_MeV", weighted_percentile(energy, soft_weights, 10.0));
  row.set("energy_p50_soft50_MeV", weighted_percentile(energy, soft_weights, 50.0));
  row.set("energy_p90_soft50_MeV", weighted_percentile(energy, soft_weights, 90.0));
  row.set("energy_p95_soft50_MeV", weighted_percentile(energy, soft_weights, 95.0));

  if (longitudinal != Longitudinal::Z) {
    set_nan(row, {"theta_r_p90_soft50_mrad", "theta_r_p95_soft50_mrad", "emitn_x_soft50_um_rad",
                  "emitn_y_soft50_um_rad", "emitn_xy_soft50_um_rad"});
    return row;
  }

  Mask valid_transverse(n);
  for (std::size_t i = 0; i < n; ++i) {
    valid_transverse[i] = (soft[i] != 0 && std::isfinite(dump.x_m[i]) && std::isfinite(dump.y_m[i]) &&
                           std::isfinite(dump.ux[i]) && std::isfinite(dump.uy[i]) && std::isfinite(dump.uz[i]))
                              ? 1
                              : 0;
  }
  if (!any(valid_transverse)) {
    row.set("soft50_status", std::string("no_transverse_particles"));
    set_nan(row, {"theta_r_p90_soft50_mrad", "theta_r_p95_soft50_mrad", "emitn_x_soft50_um_rad",
                  "emitn_y_soft50_um_rad", "emitn_xy_soft50_um_rad"});
    return row;
  }

  const auto w = gather(soft_weights, valid_transverse);
  const auto x = gather(dump.x_m, valid_transverse);
  const auto y = gather(dump.y_m, valid_transverse);
  const auto ux = gather(dump.ux, valid_transverse);
  const auto uy = gather(dump.uy, valid_transverse);
  const auto uz = gather(dump.uz, valid_transverse);
  std::vector<double> theta_r_mrad(w.size());
  for (std::size_t i = 0; i < w.size(); ++i) {
    const double theta_x = std::atan2(ux[i], uz[i]);
    const double theta_y = std::atan2(uy[i], uz[i]);
    theta_r_mrad[i] = std::sqrt(theta_x * theta_x + theta_y * theta_y) * 1.0e3;
  }
  const double emit_x = normalized_emittance_um_rad(x, ux, w);
  const double emit_y = normalized_emittance_um_rad(y, uy, w);
  row.set("theta_r_p90_soft50_mrad", weighted_percentile(theta_r_mrad, w, 90.0));
  row.set("theta_r_p95_soft50_mrad", weighted_percentile(theta_r_mrad, w, 95.0));
  row.set("emitn_x_soft50_um_rad", emit_x);
  row.set("emitn_y_soft50_um_rad", emit_y);
  row.set("emitn_xy_soft50_um_rad", std::sqrt(at_least_zero(emit_x) * at_least_zero(emit_y)));
  return row;
}

std::vector<table::Record> summarize_soft50_curve(const io::ParticleDump& dump, std::span<const double> energy,
                                                  std::span<const double> energy_low_values_mev,
                                                  double energy_target_mev, double reliability_floor,
                                                  double effective_count_reference, Longitudinal longitudinal,
                                                  bool forward_only, std::optional<double> exit_window_mm,
                                                  const table::Record& metadata) {
  const std::set<double> values(energy_low_values_mev.begin(), energy_low_values_mev.end());
  if (values.empty()) {
    throw std::invalid_argument("energy_low_values_mev must not be empty");
  }
  std::vector<table::Record> rows;
  for (double energy_low : values) {
    const Soft50Config config{energy_low, energy_target_mev, reliability_floor, effective_count_reference};
    table::Record row = metadata;
    row.merge(summarize_soft50_metrics(dump, energy, config, longitudinal, forward_only, exit_window_mm));
    row.set("iteration", dump.iteration);
    row.set("longitudinal_coordinate", std::string(longitudinal_name(longitudinal)));
    row.set("forward_only", forward_only);
    row.set("exit_window_mm", exit_window_mm ? table::Cell{*exit_window_mm} : table::Cell{std::string()});
    rows.push_back(std::move(row));
  }
  return rows;
}

const std::vector<std::string>& soft50_curve_columns() {
  static const std::vector<std::string> columns{
      "case_id",
      "case_name",
      "species_scope",
      "iteration",
      "selection_mode",
      "selected_particle_iteration",
      "soft50_schema_version",
      "soft50_energy_low_MeV",
      "soft50_energy_target_MeV",
      "soft50_reliability_floor",
      "soft50_effective_count_reference",
      "soft50_status",
      "n_macroparticles_soft50",
      "weight_soft50",
      "charge_soft50_pC",
      "n_effective_soft50",
      "reliability_soft50",
      "energy_mean_soft50_MeV",
      "energy_spread_rms_soft50_MeV",
      "energy_relative_spread_rms_soft50",
      "energy_p10_soft50_MeV",
      "energy_p50_soft50_MeV",
      "energy_p90_soft50_MeV",
      "energy_p95_soft50_MeV",
      "theta_r_p90_soft50_mrad",
      "theta_r_p95_soft50_mrad",
      "emitn_x_soft50_um_rad",
      "emitn_y_soft50_um_rad",
      "emitn_xy_soft50_um_rad",
      "charge_Ege50MeV_pC",
      "n_macroparticles_Ege50MeV",
      "longitudinal_coordinate",
      "forward_only",
      "exit_window_mm"};
  return columns;
}

}  // namespace guiding::physics

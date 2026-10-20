#include "guiding/products/particle_view.hpp"

#include <algorithm>
#include <cmath>
#include <limits>

#include <fmt/format.h>

#include "guiding/physics/field_metrics.hpp"

namespace guiding::products {
namespace {

double charge_pC(double weight) { return weight * physics::kElementaryChargeC / 1.0e-12; }

// Finite extent of the selected values; degenerate ranges get a unit width.
std::pair<double, double> extent(std::span<const double> values, const physics::Mask& mask) {
  double low = std::numeric_limits<double>::infinity();
  double high = -std::numeric_limits<double>::infinity();
  for (std::size_t i = 0; i < values.size(); ++i) {
    if (mask[i] != 0 && std::isfinite(values[i])) {
      low = std::min(low, values[i]);
      high = std::max(high, values[i]);
    }
  }
  if (!(low <= high)) {
    return {0.0, 1.0};
  }
  if (high == low) {
    return {low - 0.5, high + 0.5};
  }
  return {low, high};
}

std::size_t bin_index(double value, double low, double high, std::size_t bins) {
  const double position = (value - low) / (high - low) * static_cast<double>(bins);
  return std::min(bins - 1, static_cast<std::size_t>(std::max(0.0, position)));
}

}  // namespace

Histogram2D charge_histogram2d(std::span<const double> x, std::span<const double> y, std::span<const double> weights,
                               const physics::Mask& mask, std::size_t nx, std::size_t ny) {
  Histogram2D histogram;
  histogram.nx = std::max<std::size_t>(nx, 1);
  histogram.ny = std::max<std::size_t>(ny, 1);
  std::tie(histogram.x_min, histogram.x_max) = extent(x, mask);
  std::tie(histogram.y_min, histogram.y_max) = extent(y, mask);
  histogram.charge_pC.assign(histogram.nx * histogram.ny, 0.0);
  for (std::size_t i = 0; i < weights.size(); ++i) {
    if (mask[i] == 0 || !std::isfinite(x[i]) || !std::isfinite(y[i]) || !std::isfinite(weights[i])) {
      continue;
    }
    const std::size_t ix = bin_index(x[i], histogram.x_min, histogram.x_max, histogram.nx);
    const std::size_t iy = histogram.ny - 1 - bin_index(y[i], histogram.y_min, histogram.y_max, histogram.ny);
    histogram.charge_pC[iy * histogram.nx + ix] += charge_pC(weights[i]);
  }
  histogram.max_charge_pC = *std::max_element(histogram.charge_pC.begin(), histogram.charge_pC.end());
  return histogram;
}

ParticleView build_particle_view(const io::ParticleDump& dump, const std::string& species_scope,
                                 const ParticleViewOptions& options) {
  ParticleView view;
  view.species_scope = species_scope;
  view.iteration = dump.iteration;
  view.time_fs = dump.time_fs;
  view.n_total = dump.size();

  physics::SummaryOptions summary_options;
  summary_options.hot_energy_mev = options.hot_energy_mev;
  summary_options.longitudinal = options.longitudinal;
  summary_options.exit_window_mm = options.exit_window_mm;
  summary_options.forward_only = options.forward_only;
  summary_options.soft50 = options.soft50;
  summary_options.species_scope = species_scope;
  view.summary = physics::summarize_dump(dump, summary_options);

  const auto energy = physics::kinetic_energy_mev(dump);
  const physics::Mask hot = physics::select_hot_electrons(dump, energy, options.hot_energy_mev, options.forward_only,
                                                          options.longitudinal, options.exit_window_mm);
  view.n_hot = physics::count(hot);

  // Spectra.
  physics::Mask valid(dump.size());
  for (std::size_t i = 0; i < dump.size(); ++i) {
    valid[i] = std::isfinite(energy[i]) && std::isfinite(dump.w[i]) && dump.w[i] > 0.0 ? 1 : 0;
  }
  const std::size_t bins = std::max<std::size_t>(options.spectrum_bins, 1);
  const auto [e_low, e_high_raw] = extent(energy, valid);
  const double e_low_edge = std::min(0.0, e_low);
  const double e_high = std::max(e_high_raw, e_low_edge + 1.0);
  view.energy_edges_mev.resize(bins + 1);
  for (std::size_t b = 0; b <= bins; ++b) {
    view.energy_edges_mev[b] = e_low_edge + (e_high - e_low_edge) * static_cast<double>(b) / static_cast<double>(bins);
  }
  const double bin_width = (e_high - e_low_edge) / static_cast<double>(bins);
  view.spectrum_all.assign(bins, 0.0);
  view.spectrum_hot.assign(bins, 0.0);
  for (std::size_t i = 0; i < dump.size(); ++i) {
    if (valid[i] == 0) {
      continue;
    }
    const std::size_t b = bin_index(energy[i], e_low_edge, e_high, bins);
    view.spectrum_all[b] += charge_pC(dump.w[i]) / bin_width;
    if (hot[i] != 0) {
      view.spectrum_hot[b] += charge_pC(dump.w[i]) / bin_width;
    }
  }

  // Acceptance grid (same rows as particle_acceptance_curves.csv).
  physics::AcceptanceOptions acceptance_options;
  acceptance_options.species_scope = species_scope;
  acceptance_options.theta_cuts_mrad = options.theta_cuts_mrad;
  acceptance_options.energy_min_mev = options.energy_cuts_mev;
  acceptance_options.longitudinal = options.longitudinal;
  acceptance_options.forward_only = options.forward_only;
  for (const auto& row : physics::summarize_acceptance_curves(dump, acceptance_options)) {
    const double theta = std::get<double>(*row.get("theta_cut_mrad"));
    const double e_min = std::get<double>(*row.get("E_min_MeV"));
    if (view.theta_cuts_mrad.empty() || view.theta_cuts_mrad.back() != theta) {
      if (std::find(view.theta_cuts_mrad.begin(), view.theta_cuts_mrad.end(), theta) == view.theta_cuts_mrad.end()) {
        view.theta_cuts_mrad.push_back(theta);
      }
    }
    if (std::find(view.energy_cuts_mev.begin(), view.energy_cuts_mev.end(), e_min) == view.energy_cuts_mev.end()) {
      view.energy_cuts_mev.push_back(e_min);
    }
    view.accepted_pC.push_back(std::get<double>(*row.get("accepted_charge_pC")));
  }

  // Phase spaces of the hot electrons.
  const auto& q_long = options.longitudinal == physics::Longitudinal::Z   ? dump.z_m
                       : options.longitudinal == physics::Longitudinal::X ? dump.x_m
                                                                          : dump.y_m;
  const auto& u_long = options.longitudinal == physics::Longitudinal::Z   ? dump.uz
                       : options.longitudinal == physics::Longitudinal::X ? dump.ux
                                                                          : dump.uy;
  const char* axis = physics::longitudinal_name(options.longitudinal);
  std::vector<double> q_um(dump.size());
  for (std::size_t i = 0; i < dump.size(); ++i) {
    q_um[i] = q_long[i] * 1.0e6;
  }
  const std::size_t nb = std::max<std::size_t>(options.phase_bins, 2);
  auto add = [&](std::string title, std::string x_label, std::string y_label, std::span<const double> x,
                 std::span<const double> y) {
    auto histogram = charge_histogram2d(x, y, dump.w, hot, nb, nb);
    histogram.title = std::move(title);
    histogram.x_label = std::move(x_label);
    histogram.y_label = std::move(y_label);
    view.phase_spaces.push_back(std::move(histogram));
  };
  add("longitudinal phase space", fmt::format("{} [um]", axis), fmt::format("u{}", axis), q_um, u_long);
  add("energy vs position", fmt::format("{} [um]", axis), "E [MeV]", q_um, energy);
  if (options.longitudinal == physics::Longitudinal::Z) {
    std::vector<double> x_um(dump.size());
    std::vector<double> y_um(dump.size());
    std::vector<double> theta_x(dump.size());
    std::vector<double> theta_y(dump.size());
    for (std::size_t i = 0; i < dump.size(); ++i) {
      x_um[i] = dump.x_m[i] * 1.0e6;
      y_um[i] = dump.y_m[i] * 1.0e6;
      theta_x[i] = std::atan2(dump.ux[i], dump.uz[i]) * 1.0e3;
      theta_y[i] = std::atan2(dump.uy[i], dump.uz[i]) * 1.0e3;
    }
    add("x - theta_x", "x [um]", "theta_x [mrad]", x_um, theta_x);
    add("y - theta_y", "y [um]", "theta_y [mrad]", y_um, theta_y);
    add("theta_x - theta_y", "theta_x [mrad]", "theta_y [mrad]", theta_x, theta_y);
  }
  return view;
}

}  // namespace guiding::products

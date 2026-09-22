#include "guiding/physics/particles.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <set>
#include <stdexcept>

#include <fmt/format.h>

#include "guiding/numeric/npcompat.hpp"
#include "guiding/physics/field_metrics.hpp"
#include "guiding/physics/transverse.hpp"

namespace guiding::physics {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

const std::vector<double>& longitudinal_positions(const io::ParticleDump& dump, Longitudinal axis) {
  switch (axis) {
    case Longitudinal::X: return dump.x_m;
    case Longitudinal::Y: return dump.y_m;
    case Longitudinal::Z: return dump.z_m;
  }
  return dump.z_m;
}

const std::vector<double>& longitudinal_momenta(const io::ParticleDump& dump, Longitudinal axis) {
  switch (axis) {
    case Longitudinal::X: return dump.ux;
    case Longitudinal::Y: return dump.uy;
    case Longitudinal::Z: return dump.uz;
  }
  return dump.uz;
}

// np.allclose(a, b): |a - b| <= atol + rtol * |b|
bool allclose(double a, double b) { return std::abs(a - b) <= 1.0e-8 + 1.0e-5 * std::abs(b); }

}  // namespace

Longitudinal parse_longitudinal(std::string_view name) {
  if (name == "z") {
    return Longitudinal::Z;
  }
  if (name == "x") {
    return Longitudinal::X;
  }
  if (name == "y") {
    return Longitudinal::Y;
  }
  throw std::invalid_argument("longitudinal must be one of: x, y, z");
}

const char* longitudinal_name(Longitudinal axis) noexcept {
  switch (axis) {
    case Longitudinal::X: return "x";
    case Longitudinal::Y: return "y";
    case Longitudinal::Z: return "z";
  }
  return "z";
}

std::vector<double> kinetic_energy_mev(const io::ParticleDump& dump) {
  std::vector<double> energy(dump.size());
  for (std::size_t i = 0; i < energy.size(); ++i) {
    const double gamma = std::sqrt(1.0 + dump.ux[i] * dump.ux[i] + dump.uy[i] * dump.uy[i] + dump.uz[i] * dump.uz[i]);
    energy[i] = (gamma - 1.0) * kElectronRestEnergyMeV;
  }
  return energy;
}

io::ParticleDump concatenate_particle_dumps(std::span<const io::ParticleDump> dumps) {
  if (dumps.empty()) {
    throw std::invalid_argument("At least one ParticleDump is required");
  }
  std::set<std::int64_t> iterations;
  std::vector<double> finite_times;
  for (const auto& dump : dumps) {
    iterations.insert(dump.iteration);
    if (std::isfinite(dump.time_fs)) {
      finite_times.push_back(dump.time_fs);
    }
  }
  if (iterations.size() != 1) {
    std::string listed;
    for (std::int64_t iteration : iterations) {
      listed += fmt::format("{}{}", listed.empty() ? "" : ", ", iteration);
    }
    throw std::invalid_argument(fmt::format("Cannot combine different iterations: [{}]", listed));
  }
  for (double time : finite_times) {
    if (!allclose(time, finite_times.front())) {
      throw std::invalid_argument("Cannot combine ParticleDump objects with different times");
    }
  }

  io::ParticleDump combined;
  combined.iteration = *iterations.begin();
  combined.time_fs = finite_times.empty() ? kNaN : finite_times.front();
  for (const auto& dump : dumps) {
    combined.x_m.insert(combined.x_m.end(), dump.x_m.begin(), dump.x_m.end());
    combined.y_m.insert(combined.y_m.end(), dump.y_m.begin(), dump.y_m.end());
    combined.z_m.insert(combined.z_m.end(), dump.z_m.begin(), dump.z_m.end());
    combined.ux.insert(combined.ux.end(), dump.ux.begin(), dump.ux.end());
    combined.uy.insert(combined.uy.end(), dump.uy.begin(), dump.uy.end());
    combined.uz.insert(combined.uz.end(), dump.uz.begin(), dump.uz.end());
    combined.w.insert(combined.w.end(), dump.w.begin(), dump.w.end());
  }
  return combined;
}

Mask select_hot_electrons(const io::ParticleDump& dump, std::span<const double> energy, double hot_energy_mev,
                          bool forward_only, Longitudinal longitudinal, std::optional<double> exit_window_mm) {
  const auto& q_long = longitudinal_positions(dump, longitudinal);
  const auto& u_long = longitudinal_momenta(dump, longitudinal);
  Mask mask(dump.size());
  for (std::size_t i = 0; i < mask.size(); ++i) {
    bool selected = std::isfinite(energy[i]) && std::isfinite(dump.w[i]) && dump.w[i] > 0.0;
    selected = selected && energy[i] >= hot_energy_mev;
    selected = selected && std::isfinite(q_long[i]) && std::isfinite(u_long[i]);
    if (forward_only) {
      selected = selected && u_long[i] > 0.0;
    }
    mask[i] = selected ? 1 : 0;
  }
  if (exit_window_mm) {
    const double window_m = *exit_window_mm * 1.0e-3;
    if (window_m <= 0.0) {
      throw std::invalid_argument("exit_window_mm must be positive when provided");
    }
    const double q_max = any(mask) ? np::max_value<double>(gather(q_long, mask)) : kNaN;
    if (std::isfinite(q_max)) {
      for (std::size_t i = 0; i < mask.size(); ++i) {
        mask[i] = (mask[i] != 0 && q_long[i] >= q_max - window_m) ? 1 : 0;
      }
    }
  }
  return mask;
}

table::Record summarize_dump(const io::ParticleDump& dump, const SummaryOptions& options) {
  const auto energy = kinetic_energy_mev(dump);
  const auto& w = dump.w;
  const std::size_t n = w.size();

  Mask finite(n);
  for (std::size_t i = 0; i < n; ++i) {
    finite[i] = (std::isfinite(energy[i]) && std::isfinite(w[i]) && w[i] > 0.0) ? 1 : 0;
  }
  const Mask hot = select_hot_electrons(dump, energy, options.hot_energy_mev, options.forward_only,
                                        options.longitudinal, options.exit_window_mm);
  const auto& q_long = longitudinal_positions(dump, options.longitudinal);
  const auto& u_long = longitudinal_momenta(dump, options.longitudinal);

  const auto hot_w = gather(w, hot);
  const auto finite_w = gather(w, finite);
  const double hot_weight = any(hot) ? np::pairwise_sum<double>(hot_w) : 0.0;
  const double total_weight = any(finite) ? np::pairwise_sum<double>(finite_w) : 0.0;

  table::Record row;
  row.set("species_scope", options.species_scope);
  row.set("iteration", dump.iteration);
  row.set("time_fs", dump.time_fs);
  row.set("n_macroparticles_total", static_cast<std::int64_t>(n));
  row.set("n_macroparticles_valid", static_cast<std::int64_t>(count(finite)));
  row.set("n_macroparticles_hot", static_cast<std::int64_t>(count(hot)));
  row.set("weight_total", total_weight);
  row.set("weight_hot", hot_weight);
  row.set("charge_hot_pC", hot_weight * kElementaryChargeC / 1.0e-12);
  row.set("hot_energy_threshold_MeV", options.hot_energy_mev);
  row.set("forward_only", options.forward_only);
  row.set("longitudinal_coordinate", std::string(longitudinal_name(options.longitudinal)));
  row.set("exit_window_mm", options.exit_window_mm ? table::Cell{*options.exit_window_mm} : table::Cell{std::string()});

  const auto finite_energy = gather(energy, finite);
  const bool has_finite = any(finite);
  row.set("Emax_MeV", has_finite ? np::max_value<double>(finite_energy) : kNaN);
  const WeightedSample valid_sample(finite_energy, finite_w);
  row.set("E99_MeV", has_finite ? valid_sample.percentile(99.0) : kNaN);
  row.set("E95_MeV", has_finite ? valid_sample.percentile(95.0) : kNaN);
  row.set("E90_MeV", has_finite ? valid_sample.percentile(90.0) : kNaN);

  if (any(hot)) {
    const auto hot_energy = gather(energy, hot);
    const auto hot_q = gather(q_long, hot);
    const auto hot_u = gather(u_long, hot);
    row.set("Emax_hot_MeV", np::max_value<double>(hot_energy));
    row.set("Emean_hot_MeV", weighted_average(hot_energy, hot_w));
    const WeightedSample hot_sample(hot_energy, hot_w);
    row.set("E99_hot_MeV", hot_sample.percentile(99.0));
    row.set("E95_hot_MeV", hot_sample.percentile(95.0));
    row.set("q_long_mean_hot_mm", weighted_average(hot_q, hot_w) * 1.0e3);
    row.set("q_long_min_hot_mm", np::min_value<double>(hot_q) * 1.0e3);
    row.set("q_long_max_hot_mm", np::max_value<double>(hot_q) * 1.0e3);
    row.set("u_long_mean_hot", weighted_average(hot_u, hot_w));
    row.set("u_long_max_hot", np::max_value<double>(hot_u));
  } else {
    for (const char* key : {"Emax_hot_MeV", "Emean_hot_MeV", "E99_hot_MeV", "E95_hot_MeV", "q_long_mean_hot_mm",
                            "q_long_min_hot_mm", "q_long_max_hot_mm", "u_long_mean_hot", "u_long_max_hot"}) {
      row.set(key, kNaN);
    }
  }

  table::Record out = add_beamlike_metrics(row, options.beamlike);
  out.merge(summarize_transverse_metrics(dump, hot, options.longitudinal));
  if (options.soft50) {
    out.merge(summarize_soft50_metrics(dump, energy, *options.soft50, options.longitudinal, options.forward_only,
                                       options.exit_window_mm));
  }
  return out;
}

std::vector<table::Record> summarize_acceptance_curves(const io::ParticleDump& dump, const AcceptanceOptions& options) {
  std::vector<double> theta_cuts = options.theta_cuts_mrad;
  std::vector<double> energy_cuts = options.energy_min_mev;
  if (theta_cuts.empty()) {
    throw std::invalid_argument("theta_cuts_mrad must not be empty");
  }
  if (energy_cuts.empty()) {
    throw std::invalid_argument("e_min_mev must not be empty");
  }
  if (std::any_of(theta_cuts.begin(), theta_cuts.end(), [](double v) { return !std::isfinite(v) || v <= 0.0; })) {
    throw std::invalid_argument("theta_cuts_mrad values must be finite and positive");
  }
  if (std::any_of(energy_cuts.begin(), energy_cuts.end(), [](double v) { return !std::isfinite(v) || v < 0.0; })) {
    throw std::invalid_argument("e_min_mev values must be finite and non-negative");
  }
  std::sort(theta_cuts.begin(), theta_cuts.end());
  std::sort(energy_cuts.begin(), energy_cuts.end());

  const auto energy = kinetic_energy_mev(dump);
  const auto& weights = dump.w;
  // theta_r = sqrt(atan2(u_t1, u_long)^2 + atan2(u_t2, u_long)^2) * 1e3
  const std::vector<double>* u_long = &dump.uz;
  const std::vector<double>* u_t1 = &dump.ux;
  const std::vector<double>* u_t2 = &dump.uy;
  if (options.longitudinal == Longitudinal::X) {
    u_long = &dump.ux;
    u_t1 = &dump.uy;
    u_t2 = &dump.uz;
  } else if (options.longitudinal == Longitudinal::Y) {
    u_long = &dump.uy;
    u_t1 = &dump.ux;
    u_t2 = &dump.uz;
  }

  const std::size_t n = dump.size();
  std::vector<double> theta_r(n);
  Mask valid(n);
  for (std::size_t i = 0; i < n; ++i) {
    const double theta_1 = std::atan2((*u_t1)[i], (*u_long)[i]);
    const double theta_2 = std::atan2((*u_t2)[i], (*u_long)[i]);
    theta_r[i] = std::sqrt(theta_1 * theta_1 + theta_2 * theta_2) * 1.0e3;
    bool ok = std::isfinite(energy[i]) && std::isfinite(theta_r[i]) && std::isfinite(weights[i]) && weights[i] > 0.0;
    if (options.forward_only) {
      ok = ok && std::isfinite((*u_long)[i]) && (*u_long)[i] > 0.0;
    }
    valid[i] = ok ? 1 : 0;
  }

  const std::int64_t selected_iteration = options.selected_particle_iteration.value_or(dump.iteration);
  std::vector<table::Record> rows;
  Mask accepted(n);
  for (double theta_cut : theta_cuts) {
    for (double energy_cut : energy_cuts) {
      for (std::size_t i = 0; i < n; ++i) {
        accepted[i] = (valid[i] != 0 && theta_r[i] <= theta_cut && energy[i] >= energy_cut) ? 1 : 0;
      }
      const double accepted_weight = any(accepted) ? np::pairwise_sum<double>(gather(weights, accepted)) : 0.0;
      table::Record row;
      row.set("case_id", options.case_id);
      row.set("case_name", options.case_name);
      row.set("species_scope", options.species_scope);
      row.set("iteration", dump.iteration);
      row.set("selection_mode", options.selection_mode);
      row.set("selected_particle_iteration", selected_iteration);
      row.set("theta_cut_mrad", theta_cut);
      row.set("E_min_MeV", energy_cut);
      row.set("accepted_charge_pC", accepted_weight * kElementaryChargeC / 1.0e-12);
      row.set("accepted_weight", accepted_weight);
      row.set("accepted_n_macroparticles", static_cast<std::int64_t>(count(accepted)));
      row.set("longitudinal_coordinate", std::string(longitudinal_name(options.longitudinal)));
      row.set("forward_only", options.forward_only);
      rows.push_back(std::move(row));
    }
  }
  return rows;
}

const std::vector<std::string>& particle_acceptance_columns() {
  static const std::vector<std::string> columns{"case_id",
                                                "case_name",
                                                "species_scope",
                                                "iteration",
                                                "selection_mode",
                                                "selected_particle_iteration",
                                                "theta_cut_mrad",
                                                "E_min_MeV",
                                                "accepted_charge_pC",
                                                "accepted_weight",
                                                "accepted_n_macroparticles",
                                                "longitudinal_coordinate",
                                                "forward_only"};
  return columns;
}

}  // namespace guiding::physics

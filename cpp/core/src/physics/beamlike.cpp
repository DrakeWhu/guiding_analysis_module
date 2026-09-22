#include "guiding/physics/beamlike.hpp"

#include <cmath>
#include <limits>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "guiding/numeric/npcompat.hpp"
#include "guiding/table/csv.hpp"
#include "guiding/table/py_format.hpp"

namespace guiding::physics {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// beamlike._format_threshold: "100" for integral values, else "{:g}".
std::string format_threshold(double value) {
  if (std::isfinite(value) && std::floor(value) == value) {
    return std::to_string(static_cast<long long>(value));
  }
  return fmt::format("{:g}", value);
}

// clamp(value, low, high) = max(low, min(high, value)), non-finite -> low
double clamp(double value, double low = 0.0, double high = 1.0) {
  if (!std::isfinite(value)) {
    return low;
  }
  const double upper = (value < high) ? value : high;
  return (upper > low) ? upper : low;
}

double log_component(double value, double minimum, double reference) {
  if (!std::isfinite(value) || value <= minimum) {
    return 0.0;
  }
  if (reference <= minimum) {
    return 1.0;
  }
  return clamp(std::log10(value / minimum) / std::log10(reference / minimum));
}

double linear_component(double value, double minimum, double reference) {
  if (!std::isfinite(value) || value <= minimum) {
    return 0.0;
  }
  if (reference <= minimum) {
    return 1.0;
  }
  return clamp((value - minimum) / (reference - minimum));
}

double mono_proxy_from_energies(double e95_mev, double emax_mev) {
  if (!std::isfinite(e95_mev) || !std::isfinite(emax_mev) || emax_mev <= 0.0) {
    return kNaN;
  }
  return e95_mev / emax_mev;
}

// Python max(value, 0.0)
double at_least_zero(double value) { return (0.0 > value) ? 0.0 : value; }

std::string join(const std::vector<std::string>& parts) {
  std::string out;
  for (const auto& part : parts) {
    out += (out.empty() ? "" : ";") + part;
  }
  return out;
}

}  // namespace

double finite_float(const table::Record& row, const std::string& key) {
  const auto cell = row.get(key);
  double value = kNaN;
  if (cell) {
    if (const auto* real = std::get_if<double>(&*cell)) {
      value = *real;
    } else if (const auto* integer = std::get_if<std::int64_t>(&*cell)) {
      value = static_cast<double>(*integer);
    } else if (const auto* flag = std::get_if<bool>(&*cell)) {
      value = *flag ? 1.0 : 0.0;
    } else if (const auto* text = std::get_if<std::string>(&*cell)) {
      value = table::parse_py_float(*text).value_or(kNaN);
    }
  }
  return std::isfinite(value) ? value : kNaN;
}

std::pair<double, std::string> divergence_component_from_row(const table::Record& row) {
  double divergence = finite_float(row, "divergence_rms_mrad");
  std::string source = "divergence_rms_mrad";
  if (!std::isfinite(divergence)) {
    divergence = finite_float(row, "theta_rms_mrad");
    source = "theta_rms_mrad";
  }
  if (!std::isfinite(divergence)) {
    const double theta_x = finite_float(row, "theta_x_rms_mrad");
    const double theta_y = finite_float(row, "theta_y_rms_mrad");
    if (std::isfinite(theta_x) && std::isfinite(theta_y)) {
      divergence = std::sqrt(theta_x * theta_x + theta_y * theta_y);
      source = "theta_x/y_rms_mrad";
    }
  }
  if (!std::isfinite(divergence)) {
    return {1.0, "not_available"};
  }
  return {clamp(1.0 / (1.0 + divergence / 8.0)), source};
}

table::Record score_particle_summary_row(const table::Record& row, const BeamlikeConfig& cfg) {
  const double charge = finite_float(row, "charge_hot_pC");
  const double n_hot = finite_float(row, "n_macroparticles_hot");
  const double e95 = finite_float(row, "E95_hot_MeV");
  const double emean = finite_float(row, "Emean_hot_MeV");
  const double emax = finite_float(row, "Emax_hot_MeV");
  const double q_long_min = finite_float(row, "q_long_min_hot_mm");
  const double q_long_max = finite_float(row, "q_long_max_hot_mm");

  std::vector<std::string> hard_rejection_reasons;
  std::vector<std::string> quality_reasons;
  if (!std::isfinite(n_hot) || n_hot < cfg.min_hot_macroparticles) {
    hard_rejection_reasons.push_back(fmt::format("low_n_hot<{}", format_threshold(cfg.min_hot_macroparticles)));
  }
  if (!std::isfinite(charge) || charge < cfg.min_hot_charge_pC) {
    hard_rejection_reasons.push_back(fmt::format("low_charge<{}pC", format_threshold(cfg.min_hot_charge_pC)));
  }
  if (!std::isfinite(e95) || e95 < cfg.min_hot_E95_MeV) {
    hard_rejection_reasons.push_back(fmt::format("low_E95<{}MeV", format_threshold(cfg.min_hot_E95_MeV)));
  }
  const bool eligible = hard_rejection_reasons.empty();

  const double charge_component = log_component(charge, cfg.min_hot_charge_pC, cfg.charge_ref_pC);
  const double statistics_component = log_component(n_hot, cfg.min_hot_macroparticles, cfg.n_hot_ref);
  const double robust_energy = 0.65 * e95 + 0.30 * emean + 0.05 * emax;
  const double energy_component = linear_component(robust_energy, cfg.min_hot_E95_MeV, cfg.energy_ref_MeV);

  const double mono_proxy = mono_proxy_from_energies(e95, emax);
  const double mono_component = linear_component(mono_proxy, cfg.min_mono_proxy, cfg.mono_ref);
  if (std::isfinite(mono_proxy) && mono_proxy < cfg.min_mono_proxy) {
    quality_reasons.push_back(fmt::format("broad_proxy_E95_over_Emax<{}", format_threshold(cfg.min_mono_proxy)));
  }

  double z_span_hot_mm = kNaN;
  double z_compact_component = 1.0;
  if (std::isfinite(q_long_min) && std::isfinite(q_long_max)) {
    z_span_hot_mm = at_least_zero(q_long_max - q_long_min);
    z_compact_component = 1.0 / (1.0 + z_span_hot_mm / cfg.z_span_ref_mm);
  }

  const auto [divergence_component, divergence_source] = divergence_component_from_row(row);

  double beamlike_score = 0.0;
  double beam_yield_score = 0.0;
  if (eligible) {
    const double z_factor = 1.0 - cfg.z_compact_weight + cfg.z_compact_weight * z_compact_component;
    const double divergence_factor = 1.0 - cfg.divergence_weight + cfg.divergence_weight * divergence_component;
    beamlike_score = cfg.score_scale * np::c_pow(charge_component, cfg.charge_exponent) *
                     np::c_pow(statistics_component, cfg.statistics_exponent) *
                     np::c_pow(energy_component, cfg.energy_exponent) *
                     np::c_pow(mono_component, cfg.mono_exponent) * z_factor * divergence_factor;
    beam_yield_score = std::log10(1.0 + at_least_zero(charge)) * std::log10(1.0 + at_least_zero(n_hot)) *
                       at_least_zero(e95);
  }

  std::vector<std::string> tags;
  if (eligible) {
    tags.emplace_back("beamlike_candidate");
  }
  if (charge >= 1000.0) {
    tags.emplace_back("nC_class");
  } else if (charge >= 300.0) {
    tags.emplace_back("high_charge");
  } else if (charge >= cfg.min_hot_charge_pC) {
    tags.emplace_back("usable_charge");
  }
  if (n_hot >= 1000.0) {
    tags.emplace_back("good_statistics");
  } else if (n_hot >= cfg.min_hot_macroparticles) {
    tags.emplace_back("usable_statistics");
  }
  if (e95 >= 180.0) {
    tags.emplace_back("high_E95");
  } else if (e95 >= 100.0) {
    tags.emplace_back("medium_E95");
  }
  if (emax >= 300.0) {
    tags.emplace_back("high_Emax");
  }
  if (std::isfinite(mono_proxy)) {
    if (mono_proxy >= 0.60) {
      tags.emplace_back("compact_spectrum_proxy");
    } else if (mono_proxy < 0.35) {
      tags.emplace_back("broad_spectrum_proxy");
    }
  }
  if (divergence_source == "not_available") {
    tags.emplace_back("no_divergence_metric");
  }

  std::string status;
  if (!std::isfinite(n_hot) || n_hot < cfg.min_hot_macroparticles) {
    status = "insufficient_hot_electron_statistics";
  } else if (!std::isfinite(charge) || charge < cfg.min_hot_charge_pC) {
    status = "insufficient_hot_charge";
  } else if (!std::isfinite(e95) || e95 < cfg.min_hot_E95_MeV) {
    status = "insufficient_hot_energy";
  } else if (!quality_reasons.empty()) {
    status = "eligible_with_quality_flags";
  } else {
    status = "eligible_beamlike";
  }

  std::vector<std::string> reasons = hard_rejection_reasons;
  reasons.insert(reasons.end(), quality_reasons.begin(), quality_reasons.end());

  table::Record out;
  out.set("eligible_beamlike", eligible);
  out.set("beamlike_status", status);
  out.set("beamlike_score", beamlike_score);
  out.set("beam_yield_score", beam_yield_score);
  out.set("beamlike_rejection_reasons", join(reasons));
  out.set("beamlike_tags", join(tags));
  out.set("mono_proxy_E95_over_Emax",
          std::isfinite(mono_proxy) ? table::Cell{mono_proxy} : table::Cell{std::string()});
  out.set("z_span_hot_mm", std::isfinite(z_span_hot_mm) ? table::Cell{z_span_hot_mm} : table::Cell{std::string()});
  out.set("statistics_component", statistics_component);
  out.set("charge_component", charge_component);
  out.set("energy_component", energy_component);
  out.set("mono_component", mono_component);
  out.set("z_compact_component", z_compact_component);
  out.set("divergence_component", divergence_component);
  out.set("divergence_source", divergence_source);
  for (const char* key : {"theta_x_rms_mrad", "theta_y_rms_mrad", "theta_rms_mrad", "theta_x_p95_mrad",
                          "theta_y_p95_mrad", "emit_x_norm_mm_mrad", "emit_y_norm_mm_mrad"}) {
    out.set(key, std::string());
  }
  return out;
}

table::Record add_beamlike_metrics(const table::Record& row, const BeamlikeConfig& config) {
  table::Record out = row;
  out.merge(score_particle_summary_row(out, config));
  return out;
}

}  // namespace guiding::physics

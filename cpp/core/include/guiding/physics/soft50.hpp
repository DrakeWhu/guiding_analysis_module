#pragma once

#include <optional>
#include <span>
#include <string>
#include <vector>

#include "guiding/io/particle_reader.hpp"
#include "guiding/physics/longitudinal.hpp"
#include "guiding/table/record.hpp"

// Port of cap_guiding/soft50.py (soft energy acceptance around a target energy).
namespace guiding::physics {

inline constexpr const char* kSoft50SchemaVersion = "soft50_v2";

struct Soft50Config {
  double energy_low_mev = 10.0;
  double energy_target_mev = 50.0;
  double reliability_floor = 0.05;
  double effective_count_reference = 200.0;

  // Soft50Config.__post_init__ checks; throws std::invalid_argument.
  void validate() const;
};

// Cubic smoothstep: 0 below low, 1 at and above target, 0 for non-finite energy.
[[nodiscard]] std::vector<double> smooth_energy_acceptance(std::span<const double> energy_mev, double energy_low_mev,
                                                           double energy_target_mev);

// (sum w)^2 / sum w^2 over finite positive weights.
[[nodiscard]] double effective_sample_size(std::span<const double> weights);

// floor + (1 - floor) * (1 - exp(-n_eff / reference))
[[nodiscard]] double soft_reliability(double n_effective, double reliability_floor, double effective_count_reference);

// summarize_soft50_metrics(dump, config, longitudinal, forward_only, exit_window_mm);
// energy_mev must be kinetic_energy_mev(dump).
[[nodiscard]] table::Record summarize_soft50_metrics(const io::ParticleDump& dump, std::span<const double> energy_mev,
                                                     const Soft50Config& config, Longitudinal longitudinal,
                                                     bool forward_only, std::optional<double> exit_window_mm);

// summarize_soft50_curve: one row per distinct energy_low value (sorted).
[[nodiscard]] std::vector<table::Record> summarize_soft50_curve(
    const io::ParticleDump& dump, std::span<const double> energy_mev, std::span<const double> energy_low_values_mev,
    double energy_target_mev, double reliability_floor, double effective_count_reference, Longitudinal longitudinal,
    bool forward_only, std::optional<double> exit_window_mm, const table::Record& metadata);

[[nodiscard]] const std::vector<std::string>& soft50_curve_columns();

}  // namespace guiding::physics

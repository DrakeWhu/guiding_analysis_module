#pragma once

#include <string>
#include <utility>

#include "guiding/table/record.hpp"

// Port of cap_guiding/beamlike.py.
namespace guiding::physics {

struct BeamlikeConfig {
  double min_hot_macroparticles = 200.0;
  double min_hot_charge_pC = 100.0;
  double min_hot_E95_MeV = 50.0;
  double min_mono_proxy = 0.30;

  double charge_ref_pC = 1200.0;
  double n_hot_ref = 1000.0;
  double energy_ref_MeV = 220.0;
  double mono_ref = 0.65;
  double z_span_ref_mm = 0.50;

  double score_scale = 1000.0;

  double charge_exponent = 0.90;
  double statistics_exponent = 1.00;
  double energy_exponent = 1.20;
  double mono_exponent = 0.75;

  double z_compact_weight = 0.15;
  double divergence_weight = 0.20;
};

// beamlike._finite_float: float(row[key]) when finite, else NaN.
[[nodiscard]] double finite_float(const table::Record& row, const std::string& key);

// (component, source column) as divergence_component_from_row
[[nodiscard]] std::pair<double, std::string> divergence_component_from_row(const table::Record& row);

[[nodiscard]] table::Record score_particle_summary_row(const table::Record& row, const BeamlikeConfig& config = {});

// Copy of row with the beamlike columns merged in (dict.update semantics).
[[nodiscard]] table::Record add_beamlike_metrics(const table::Record& row, const BeamlikeConfig& config = {});

}  // namespace guiding::physics

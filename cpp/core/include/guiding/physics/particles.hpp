#pragma once

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "guiding/io/particle_reader.hpp"
#include "guiding/physics/beamlike.hpp"
#include "guiding/physics/longitudinal.hpp"
#include "guiding/physics/soft50.hpp"
#include "guiding/physics/weighted.hpp"
#include "guiding/table/record.hpp"

// Port of cap_guiding/particles.py (particle summaries and acceptance curves).
namespace guiding::physics {

inline constexpr double kElectronRestEnergyMeV = 0.51099895;
inline constexpr std::array<double, 5> kDefaultAcceptanceThetaCutsMrad{2.0, 5.0, 10.0, 20.0, 50.0};
inline constexpr std::array<double, 8> kDefaultAcceptanceEnergyMinMeV{10.0,  25.0,  50.0,  100.0,
                                                                      150.0, 200.0, 250.0, 300.0};

// (gamma - 1) * m_e c^2 with gamma = sqrt(1 + ux^2 + uy^2 + uz^2), in MeV.
[[nodiscard]] std::vector<double> kinetic_energy_mev(const io::ParticleDump& dump);

// Joins species dumps of one iteration (same iteration, times within np.allclose).
[[nodiscard]] io::ParticleDump concatenate_particle_dumps(std::span<const io::ParticleDump> dumps);

[[nodiscard]] Mask select_hot_electrons(const io::ParticleDump& dump, std::span<const double> energy_mev,
                                        double hot_energy_mev, bool forward_only, Longitudinal longitudinal,
                                        std::optional<double> exit_window_mm);

struct SummaryOptions {
  double hot_energy_mev = 10.0;
  Longitudinal longitudinal = Longitudinal::Z;
  std::optional<double> exit_window_mm;
  bool forward_only = true;
  BeamlikeConfig beamlike;
  std::optional<Soft50Config> soft50;
  std::string species_scope;
};

// summarize_dump(...): one particle_summary.csv row (beamlike, transverse and
// soft50 metrics included) in the reference key order.
[[nodiscard]] table::Record summarize_dump(const io::ParticleDump& dump, const SummaryOptions& options);

struct AcceptanceOptions {
  std::string case_id;
  std::string case_name;
  std::string species_scope;
  std::string selection_mode;
  std::optional<std::int64_t> selected_particle_iteration;
  std::vector<double> theta_cuts_mrad{kDefaultAcceptanceThetaCutsMrad.begin(), kDefaultAcceptanceThetaCutsMrad.end()};
  std::vector<double> energy_min_mev{kDefaultAcceptanceEnergyMinMeV.begin(), kDefaultAcceptanceEnergyMinMeV.end()};
  Longitudinal longitudinal = Longitudinal::Z;
  bool forward_only = true;
};

// Q(E >= E_min, theta_r <= theta_cut) for every cut pair.
[[nodiscard]] std::vector<table::Record> summarize_acceptance_curves(const io::ParticleDump& dump,
                                                                     const AcceptanceOptions& options);

[[nodiscard]] const std::vector<std::string>& particle_acceptance_columns();

}  // namespace guiding::physics

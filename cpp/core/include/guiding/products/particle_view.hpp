#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "guiding/io/particle_reader.hpp"
#include "guiding/physics/particles.hpp"
#include "guiding/table/record.hpp"

// Data behind the dashboard's particle view: the reference summary row plus
// the spectra, acceptance grid and phase-space histograms of one dump.
namespace guiding::products {

struct ParticleViewOptions {
  double hot_energy_mev = 10.0;
  physics::Longitudinal longitudinal = physics::Longitudinal::Z;
  bool forward_only = true;
  std::optional<double> exit_window_mm;
  physics::Soft50Config soft50;
  std::vector<double> theta_cuts_mrad{physics::kDefaultAcceptanceThetaCutsMrad.begin(),
                                      physics::kDefaultAcceptanceThetaCutsMrad.end()};
  std::vector<double> energy_cuts_mev{physics::kDefaultAcceptanceEnergyMinMeV.begin(),
                                      physics::kDefaultAcceptanceEnergyMinMeV.end()};
  std::size_t spectrum_bins = 200;
  std::size_t phase_bins = 160;
};

struct Histogram2D {
  std::string title;
  std::string x_label;
  std::string y_label;
  std::size_t nx = 0;
  std::size_t ny = 0;
  double x_min = 0.0;
  double x_max = 1.0;
  double y_min = 0.0;
  double y_max = 1.0;
  std::vector<double> charge_pC;  // [iy * nx + ix], iy = 0 at y_max (top row)
  double max_charge_pC = 0.0;
};

struct ParticleView {
  std::string species_scope;
  std::int64_t iteration = 0;
  double time_fs = 0.0;
  std::size_t n_total = 0;
  std::size_t n_hot = 0;
  table::Record summary;  // particle_summary.csv row (beamlike, transverse, soft50 included)

  // Charge spectrum dQ/dE [pC/MeV] of valid particles and of the hot selection.
  std::vector<double> energy_edges_mev;
  std::vector<double> spectrum_all;
  std::vector<double> spectrum_hot;

  // Accepted charge [pC] for sorted theta cuts x sorted energy cuts.
  std::vector<double> theta_cuts_mrad;
  std::vector<double> energy_cuts_mev;
  std::vector<double> accepted_pC;  // [theta_index * n_energy + energy_index]

  std::vector<Histogram2D> phase_spaces;  // hot electrons
};

[[nodiscard]] ParticleView build_particle_view(const io::ParticleDump& dump, const std::string& species_scope,
                                               const ParticleViewOptions& options = {});

// Weighted 2D histogram over the finite selected points; the range is the data
// extent. Weights are converted to charge (w * e / 1e-12).
[[nodiscard]] Histogram2D charge_histogram2d(std::span<const double> x, std::span<const double> y,
                                             std::span<const double> weights, const physics::Mask& mask,
                                             std::size_t nx, std::size_t ny);

}  // namespace guiding::products

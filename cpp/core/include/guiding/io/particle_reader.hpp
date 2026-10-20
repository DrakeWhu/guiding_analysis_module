#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include "guiding/io/h5.hpp"
#include "guiding/io/openpmd_series.hpp"

namespace guiding::io {

// Particle arrays of one species at one iteration, as
// cap_guiding.particles.read_particle_dump returns them: positions in metres
// (position + positionOffset), normalised momenta u = p / (m c), weights.
struct ParticleDump {
  std::int64_t iteration = 0;
  double time_fs = 0.0;
  std::vector<double> x_m;
  std::vector<double> y_m;
  std::vector<double> z_m;
  std::vector<double> ux;
  std::vector<double> uy;
  std::vector<double> uz;
  std::vector<double> w;

  [[nodiscard]] std::size_t size() const noexcept { return w.size(); }
};

// What openpmd-viewer learns from the first file of a series (read_openPMD_params).
struct ParticleSeriesInfo {
  std::optional<std::vector<std::string>> fields;   // nullopt without a meshes group
  std::optional<std::vector<std::string>> species;  // nullopt without particles
  // Record components per species with openpmd-viewer's short names (x, ux, w, ...).
  std::map<std::string, std::vector<std::string>> record_components;
  bool ed_pic = false;
};

[[nodiscard]] ParticleSeriesInfo read_particle_series_info(const FileSeries& series);

// time * timeUnitSI of /data/<iteration>
[[nodiscard]] double iteration_time_s(const h5::File& file, std::int64_t iteration);

// openpmd-viewer h5py read_species_data for one component: x/y/z include
// positionOffset, ux/uy/uz are divided by m c, macro-weighted records are
// divided by w**weightingPower (ED-PIC), unitSI applied.
[[nodiscard]] std::vector<double> read_species_component(const h5::File& file, std::int64_t iteration,
                                                         const std::string& species, const std::string& component,
                                                         bool ed_pic);

// OpenPMDTimeSeries.get_particle(["x", "y", "z", "ux", "uy", "uz", "w"], species, iteration)
// plus the iteration time.
[[nodiscard]] ParticleDump read_particle_dump(const FileSeries& series, const ParticleSeriesInfo& info,
                                              const std::string& species, std::int64_t iteration,
                                              bool raw_reads = true);

}  // namespace guiding::io

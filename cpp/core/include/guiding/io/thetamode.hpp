#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <utility>
#include <vector>

#include "guiding/io/h5.hpp"

namespace guiding::io {

// One thetaMode mesh record component (e.g. E/t) of one iteration, read the way
// openpmd-viewer 1.11 reads it: all modes, values scaled by unitSI.
struct ThetaModeComponent {
  std::size_t n_modes = 0;
  std::size_t n_z = 0;
  std::size_t n_r = 0;
  bool float32 = false;       // storage precision (drives numpy dtype promotion)
  double time_s = 0.0;        // (time + timeOffset) * timeUnitSI
  std::vector<double> z_m;    // axis 0 of axisLabels [z, r]
  std::vector<double> r_m;    // axis 1, before mirroring below the axis
  std::vector<double> modes;  // C order [mode, z, r]
};

// Reads <meshesPath>/<record>/<component> of /data/<iteration>. The record must
// use axisLabels [z, r] (what cap_guiding.openpmd_io accepts).
[[nodiscard]] ThetaModeComponent read_thetamode_component(const h5::File& file, std::int64_t iteration,
                                                          std::string_view record, std::string_view component);

// Positive-r part of a theta slice oriented [row, z], as returned by
// cap_guiding.openpmd_io.read_field_rz(..., positive_r=True): rows keep the
// signed-r order of openpmd-viewer and include the mirrored r = -0.0 row when
// the radial grid starts on the axis.
struct PositiveRSlice {
  std::size_t n_rows = 0;
  std::size_t n_z = 0;
  std::vector<double> values;  // [row * n_z + z]
  std::vector<double> r_um;
  std::vector<double> z_um;
  double time_s = 0.0;
  // True when numpy would keep float32 values (theta-mode scalar or
  // cylindrical components stored as float32).
  bool float32 = false;
};

// Cartesian x and y at angle theta built from the r and t components
// (openpmd-viewer combine_cylindrical_components). Axes and time come from t.
[[nodiscard]] std::pair<PositiveRSlice, PositiveRSlice> cartesian_xy(const ThetaModeComponent& r,
                                                                     const ThetaModeComponent& t,
                                                                     double theta = 0.0);

// A scalar or cylindrical component (e.g. E/z) at angle theta.
[[nodiscard]] PositiveRSlice component_slice(const ThetaModeComponent& component, double theta = 0.0);

}  // namespace guiding::io

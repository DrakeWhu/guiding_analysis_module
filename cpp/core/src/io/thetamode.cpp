#include "guiding/io/thetamode.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <fmt/format.h>

#include "guiding/io/openpmd_series.hpp"
#include "guiding/numeric/npcompat.hpp"

namespace guiding::io {
namespace {

std::vector<double> axis_points(double spacing, double offset, double unit_si, double position, std::size_t n) {
  // openpmd-viewer FieldMetaInformation
  const double step = spacing * unit_si;
  const double start = offset * unit_si + position * step;
  const double end = start + static_cast<double>(static_cast<std::int64_t>(n) - 1) * step;
  return np::linspace(start, end, n);
}

struct RowRef {
  bool below_axis;
  std::size_t radial_index;
  double r_um;
};

// Rows of the signed-r axis concat(-r[::-1], r) whose r_um = r * 1e6 is >= 0,
// in signed order.
std::vector<RowRef> positive_rows(const std::vector<double>& r_m) {
  const std::size_t n_r = r_m.size();
  std::vector<RowRef> rows;
  for (std::size_t p = 0; p < n_r; ++p) {
    const std::size_t j = n_r - 1 - p;
    const double r_um = (-r_m[j]) * 1.0e6;
    if (r_um >= 0.0) {
      rows.push_back({true, j, r_um});
    }
  }
  for (std::size_t j = 0; j < n_r; ++j) {
    const double r_um = r_m[j] * 1.0e6;
    if (r_um >= 0.0) {
      rows.push_back({false, j, r_um});
    }
  }
  return rows;
}

// Mode multipliers of read_field_circ(m="all"): [1, cos(m th), sin(m th), ...]
// above the axis and the same times (-1)^m below it.
std::pair<std::vector<double>, std::vector<double>> mode_multipliers(std::size_t n_modes, double theta) {
  std::vector<double> above{1.0};
  std::vector<double> below{1.0};
  for (std::size_t mode = 1; mode <= n_modes / 2; ++mode) {
    const double angle = static_cast<double>(mode) * theta;
    const double c = std::cos(angle);
    const double s = std::sin(angle);
    const double sign = (mode % 2 == 1) ? -1.0 : 1.0;
    above.push_back(c);
    above.push_back(s);
    below.push_back(sign * c);
    below.push_back(sign * s);
  }
  if (above.size() != n_modes) {
    throw std::runtime_error(fmt::format("thetaMode dataset has an even number of modes ({})", n_modes));
  }
  return {std::move(above), std::move(below)};
}

// One cell of the openpmd-viewer mode sum (tensordot in float64, stored back
// in the dataset precision).
double mode_sum(const ThetaModeComponent& component, const std::vector<double>& multipliers, std::size_t z,
                std::size_t j) {
  double acc = 0.0;
  for (std::size_t k = 0; k < component.n_modes; ++k) {
    acc += multipliers[k] * component.modes[(k * component.n_z + z) * component.n_r + j];
  }
  return component.float32 ? static_cast<double>(static_cast<float>(acc)) : acc;
}

PositiveRSlice empty_slice(const ThetaModeComponent& axes, const std::vector<RowRef>& rows) {
  PositiveRSlice slice;
  slice.n_rows = rows.size();
  slice.n_z = axes.n_z;
  slice.values.resize(slice.n_rows * slice.n_z);
  slice.r_um.reserve(rows.size());
  for (const auto& row : rows) {
    slice.r_um.push_back(row.r_um);
  }
  slice.z_um.resize(axes.n_z);
  for (std::size_t z = 0; z < axes.n_z; ++z) {
    slice.z_um[z] = axes.z_m[z] * 1.0e6;
  }
  slice.time_s = axes.time_s;
  return slice;
}

}  // namespace

ThetaModeComponent read_thetamode_component(const h5::File& file, std::int64_t iteration, std::string_view record,
                                            std::string_view component) {
  const std::string base = fmt::format("/data/{}", iteration);
  const std::string meshes = h5::read_string_attribute(file.id(), "meshesPath");
  const std::string record_path = join_infile_path({base, meshes, record});
  const std::string component_path = join_infile_path({base, meshes, fmt::format("{}/{}", record, component)});

  const auto iteration_group = h5::open_object(file.id(), base);
  const auto record_group = h5::open_object(file.id(), record_path);
  const auto component_object = h5::open_object(file.id(), component_path);

  const auto labels = h5::read_string_array_attribute(record_group.id(), "axisLabels");
  if (labels.size() != 2 || labels[0] != "z" || labels[1] != "r") {
    std::string found;
    for (std::size_t i = 0; i < labels.size(); ++i) {
      found += fmt::format("{}{}: '{}'", i == 0 ? "" : ", ", i, labels[i]);
    }
    throw std::runtime_error(fmt::format("Unexpected axes={{{}}} in {}:{}. Expected {{0: 'z', 1: 'r'}}", found,
                                         file.path().string(), record_path));
  }

  ThetaModeComponent out;
  const double time = h5::read_double_attribute(iteration_group.id(), "time");
  const double time_unit = h5::read_double_attribute(iteration_group.id(), "timeUnitSI");
  const double time_offset = h5::read_double_attribute(record_group.id(), "timeOffset");
  out.time_s = (time + time_offset) * time_unit;

  std::vector<std::size_t> shape;
  const bool is_dataset = h5::is_dataset(component_object.id());
  h5::DatasetInfo info;
  if (is_dataset) {
    info = h5::dataset_info(component_object.id());
    shape = info.shape;
  } else {
    shape = h5::constant_record_shape(component_object.id());
  }
  if (shape.size() != 3) {
    throw std::runtime_error(fmt::format("{}:{} has rank {}, expected (modes, z, r)", file.path().string(),
                                         component_path, shape.size()));
  }
  out.n_modes = shape[0];
  out.n_z = shape[1];
  out.n_r = shape[2];
  const std::size_t count = out.n_modes * out.n_z * out.n_r;
  out.modes.resize(count);

  if (is_dataset) {
    if (info.kind == h5::ScalarKind::Float32) {
      out.float32 = true;
      std::vector<float> raw(count);
      file.read(component_object.id(), info, raw);
      std::copy(raw.begin(), raw.end(), out.modes.begin());
    } else if (info.kind == h5::ScalarKind::Float64) {
      file.read(component_object.id(), info, out.modes);
    } else {
      throw std::runtime_error(
          fmt::format("{}:{} is not a float32/float64 dataset", file.path().string(), component_path));
    }
  } else {
    std::fill(out.modes.begin(), out.modes.end(), h5::read_double_attribute(component_object.id(), "value"));
  }

  const double unit_si = h5::read_double_attribute(component_object.id(), "unitSI");
  if (unit_si != 1.0) {
    // data *= unitSI keeps the array dtype (float32 stays float32).
    for (double& value : out.modes) {
      value = out.float32 ? static_cast<double>(static_cast<float>(value * unit_si)) : value * unit_si;
    }
  }

  const auto spacing = h5::read_double_array_attribute(record_group.id(), "gridSpacing");
  const auto offset = h5::read_double_array_attribute(record_group.id(), "gridGlobalOffset");
  const double grid_unit = h5::read_double_attribute(record_group.id(), "gridUnitSI");
  const auto position = h5::read_double_array_attribute(component_object.id(), "position");
  if (spacing.size() < 2 || offset.size() < 2 || position.size() < 2) {
    throw std::runtime_error(fmt::format("{}:{} lacks 2-D grid attributes", file.path().string(), record_path));
  }
  out.z_m = axis_points(spacing[0], offset[0], grid_unit, position[0], out.n_z);
  out.r_m = axis_points(spacing[1], offset[1], grid_unit, position[1], out.n_r);
  return out;
}

std::pair<PositiveRSlice, PositiveRSlice> cartesian_xy(const ThetaModeComponent& r, const ThetaModeComponent& t,
                                                       double theta) {
  if (r.n_modes != t.n_modes || r.n_z != t.n_z || r.n_r != t.n_r) {
    throw std::runtime_error("r and t components have different shapes");
  }
  const auto rows = positive_rows(t.r_m);
  const auto [above, below] = mode_multipliers(r.n_modes, theta);
  const double c = std::cos(theta);
  const double s = std::sin(theta);

  PositiveRSlice x = empty_slice(t, rows);
  PositiveRSlice y = empty_slice(t, rows);
  for (std::size_t row = 0; row < rows.size(); ++row) {
    const RowRef& ref = rows[row];
    const auto& multipliers = ref.below_axis ? below : above;
    for (std::size_t z = 0; z < t.n_z; ++z) {
      const double fr = mode_sum(r, multipliers, z, ref.radial_index);
      const double ft = mode_sum(t, multipliers, z, ref.radial_index);
      double fx = c * fr - s * ft;
      double fy = s * fr + c * ft;
      if (ref.below_axis) {
        fx = fx * -1.0;
        fy = fy * -1.0;
      }
      x.values[row * t.n_z + z] = fx;
      y.values[row * t.n_z + z] = fy;
    }
  }
  return {std::move(x), std::move(y)};
}

PositiveRSlice component_slice(const ThetaModeComponent& component, double theta) {
  const auto rows = positive_rows(component.r_m);
  const auto [above, below] = mode_multipliers(component.n_modes, theta);
  PositiveRSlice slice = empty_slice(component, rows);
  slice.float32 = component.float32;
  for (std::size_t row = 0; row < rows.size(); ++row) {
    const RowRef& ref = rows[row];
    const auto& multipliers = ref.below_axis ? below : above;
    for (std::size_t z = 0; z < component.n_z; ++z) {
      slice.values[row * component.n_z + z] = mode_sum(component, multipliers, z, ref.radial_index);
    }
  }
  return slice;
}

}  // namespace guiding::io

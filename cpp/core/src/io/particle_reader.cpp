#include "guiding/io/particle_reader.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <string_view>

#include <fmt/format.h>

namespace guiding::io {
namespace {

constexpr double kSpeedOfLight = 299792458.0;  // scipy.constants.c

bool contains(const std::vector<std::string>& items, std::string_view value) {
  return std::find(items.begin(), items.end(), value) != items.end();
}

std::string strip_slashes(std::string text) {
  while (!text.empty() && text.front() == '/') {
    text.erase(text.begin());
  }
  while (!text.empty() && text.back() == '/') {
    text.pop_back();
  }
  return text;
}

std::string join_lines(const std::vector<std::string>& items) {
  std::string out;
  for (const auto& item : items) {
    out += (out.empty() ? "" : "\n - ") + item;
  }
  return out;
}

// h5py_reader.utilities.is_scalar_record
bool is_scalar_record(h5::Id object) { return h5::is_dataset(object) || h5::has_attribute(object, "value"); }

// params_reader.simplify_record
std::vector<std::string> simplify_record(std::vector<std::string> components) {
  auto remove_first = [&](std::string_view name) {
    components.erase(std::find(components.begin(), components.end(), name));
  };
  for (const char* axis : {"x", "y", "z", "r"}) {
    const std::string position = fmt::format("position/{}", axis);
    const std::string offset = fmt::format("positionOffset/{}", axis);
    if (contains(components, position) && contains(components, offset)) {
      remove_first(position);
      remove_first(offset);
      components.emplace_back(axis);
    }
  }
  for (const char* axis : {"x", "y", "z", "r"}) {
    const std::string momentum = fmt::format("momentum/{}", axis);
    if (contains(components, momentum)) {
      remove_first(momentum);
      components.push_back(fmt::format("u{}", axis));
    }
  }
  if (contains(components, "weighting")) {
    remove_first("weighting");
    components.emplace_back("w");
  }
  return components;
}

struct RecordData {
  std::vector<double> values;
  bool float32 = false;  // dtype kept by get_data without output_type
};

// h5py_reader.utilities.get_data for a dataset or a constant record, with
// unitSI applied. as_float64 mirrors output_type=np.float64.
RecordData get_data(const h5::File& file, h5::Id object, bool as_float64) {
  RecordData data;
  bool floating = true;
  if (h5::is_dataset(object)) {
    const auto info = h5::dataset_info(object);
    data.values.resize(info.element_count());
    file.read(object, info, data.values);
    data.float32 = !as_float64 && info.kind == h5::ScalarKind::Float32;
    floating = as_float64 || info.kind == h5::ScalarKind::Float32 || info.kind == h5::ScalarKind::Float64;
  } else {
    std::size_t count = 1;
    for (std::size_t extent : h5::constant_record_shape(object)) {
      count *= extent;
    }
    data.values.assign(count, h5::read_double_attribute(object, "value"));
  }
  if (floating) {
    const double unit_si = h5::read_double_attribute(object, "unitSI");
    if (unit_si != 1.0) {
      // data *= unitSI in place: computed in the promoted dtype, stored in the data dtype.
      const bool float32_product = data.float32 && h5::attribute_keeps_float32(object, "unitSI");
      for (double& value : data.values) {
        if (float32_product) {
          value = static_cast<double>(static_cast<float>(value) * static_cast<float>(unit_si));
        } else {
          value = data.float32 ? static_cast<double>(static_cast<float>(value * unit_si)) : value * unit_si;
        }
      }
    }
  }
  return data;
}

// w ** exponent for one element in the result dtype (float32 or float64),
// including numpy's fast_scalar_power paths (reciprocal, sqrt, square).
double scalar_power(double base, double exponent, bool float32) {
  if (exponent == 1.0) {
    return base;
  }
  if (exponent == -1.0) {
    return float32 ? static_cast<double>(1.0F / static_cast<float>(base)) : 1.0 / base;
  }
  if (exponent == 0.0) {
    return 1.0;
  }
  if (exponent == 0.5) {
    return float32 ? static_cast<double>(std::sqrt(static_cast<float>(base))) : std::sqrt(base);
  }
  if (exponent == 2.0) {
    return float32 ? static_cast<double>(static_cast<float>(base) * static_cast<float>(base)) : base * base;
  }
  if (float32) {
    return static_cast<double>(std::pow(static_cast<float>(base), static_cast<float>(exponent)));
  }
  return std::pow(base, exponent);
}

}  // namespace

ParticleSeriesInfo read_particle_series_info(const FileSeries& series) {
  ParticleSeriesInfo info;
  if (series.iterations().empty()) {
    return info;
  }
  const std::int64_t iteration = series.iterations().front();
  const auto file = h5::File::open_read_only(series.file(iteration));
  const std::string base = fmt::format("/data/{}", iteration);
  const auto base_group = h5::open_object(file.id(), base);
  const auto children = h5::child_names(base_group.id());

  info.ed_pic = (h5::read_uint64_attribute(file.id(), "openPMDextension") & 1U) == 1U;

  if (h5::has_attribute(file.id(), "meshesPath")) {
    const std::string meshes = strip_slashes(h5::read_string_attribute(file.id(), "meshesPath"));
    if (contains(children, meshes)) {
      const auto group = h5::open_object(base_group.id(), meshes);
      info.fields = h5::child_names(group.id());
    }
  }

  if (h5::has_attribute(file.id(), "particlesPath")) {
    const std::string particles = strip_slashes(h5::read_string_attribute(file.id(), "particlesPath"));
    if (contains(children, particles)) {
      const auto group = h5::open_object(base_group.id(), particles);
      auto species_names = h5::child_names(group.id());
      if (!species_names.empty()) {
        for (const auto& species : species_names) {
          const auto species_group = h5::open_object(group.id(), species);
          std::vector<std::string> components;
          for (const auto& record : h5::child_names(species_group.id())) {
            if (record == "particlePatches") {
              continue;
            }
            const auto record_object = h5::open_object(species_group.id(), record);
            if (is_scalar_record(record_object.id())) {
              components.push_back(record);
            } else {
              for (const auto& coordinate : h5::child_names(record_object.id())) {
                components.push_back(record + "/" + coordinate);
              }
            }
          }
          info.record_components[species] = simplify_record(std::move(components));
        }
        info.species = std::move(species_names);
      }
    }
  }
  return info;
}

double iteration_time_s(const h5::File& file, std::int64_t iteration) {
  const auto group = h5::open_object(file.id(), fmt::format("/data/{}", iteration));
  return h5::read_double_attribute(group.id(), "time") * h5::read_double_attribute(group.id(), "timeUnitSI");
}

std::vector<double> read_species_component(const h5::File& file, std::int64_t iteration, const std::string& species,
                                           const std::string& component, bool ed_pic) {
  static const std::map<std::string, std::string> kComponentPaths{
      {"x", "position/x"}, {"y", "position/y"},   {"z", "position/z"},   {"r", "position/r"},  {"ux", "momentum/x"},
      {"uy", "momentum/y"}, {"uz", "momentum/z"}, {"ur", "momentum/r"}, {"w", "weighting"}};
  const auto mapped = kComponentPaths.find(component);
  const std::string record_component = mapped != kComponentPaths.end() ? mapped->second : component;

  const std::string particles = h5::read_string_attribute(file.id(), "particlesPath");
  const std::string species_path = join_infile_path({fmt::format("/data/{}", iteration), particles, species});
  const auto species_group = h5::open_object(file.id(), species_path);
  const auto component_object = h5::open_object(species_group.id(), record_component);
  std::vector<double> values = get_data(file, component_object.id(), true).values;

  if (ed_pic && record_component != "weighting") {
    const std::string record = record_component.substr(0, record_component.find('/'));
    const auto record_object = h5::open_object(species_group.id(), record);
    const double macro_weighted = h5::read_double_attribute(record_object.id(), "macroWeighted");
    const double weighting_power = h5::read_double_attribute(record_object.id(), "weightingPower");
    if (macro_weighted == 1.0 && weighting_power != 0.0) {
      const auto weighting = h5::open_object(species_group.id(), "weighting");
      const auto weights = get_data(file, weighting.id(), false);
      // w ** (-weightingPower): the numpy scalar exponent takes part in the
      // result dtype, so float32 weights only stay float32 with a float32 attribute.
      const bool float32 = weights.float32 && h5::attribute_keeps_float32(record_object.id(), "weightingPower");
      for (std::size_t i = 0; i < values.size(); ++i) {
        values[i] *= scalar_power(weights.values.at(i), -weighting_power, float32);
      }
    }
  }

  if (component == "x" || component == "y" || component == "z" || component == "r") {
    const auto offset_object = h5::open_object(species_group.id(), "positionOffset/" + component);
    const auto offset = get_data(file, offset_object.id(), false);
    for (std::size_t i = 0; i < values.size(); ++i) {
      values[i] += offset.values.at(i);
    }
  } else if (component == "ux" || component == "uy" || component == "uz" || component == "ur") {
    const auto mass_object = h5::open_object(species_group.id(), "mass");
    const auto mass = get_data(file, mass_object.id(), false);
    const bool all_nonzero = std::all_of(mass.values.begin(), mass.values.end(), [](double m) { return m != 0.0; });
    if (all_nonzero) {
      for (std::size_t i = 0; i < values.size(); ++i) {
        const double m = mass.values.at(i);
        // norm_factor = 1. / (m * c); a Python float stays weak against float32.
        const double norm_factor =
            mass.float32
                ? static_cast<double>(1.0F / (static_cast<float>(m) * static_cast<float>(kSpeedOfLight)))
                : 1.0 / (m * kSpeedOfLight);
        values[i] *= norm_factor;
      }
    }
  }
  return values;
}

ParticleDump read_particle_dump(const FileSeries& series, const ParticleSeriesInfo& info, const std::string& species,
                                std::int64_t iteration, bool raw_reads) {
  if (!info.species) {
    throw std::runtime_error("No particle data in this time series");
  }
  if (!contains(*info.species, species)) {
    throw std::runtime_error(fmt::format(
        "The argument `species` is missing or erroneous.\nThe available species are: \n - {}\nPlease set the "
        "argument `species` accordingly.",
        join_lines(*info.species)));
  }
  static const std::vector<std::string> kVariables{"x", "y", "z", "ux", "uy", "uz", "w"};
  const auto& components = info.record_components.at(species);
  for (const auto& variable : kVariables) {
    if (!contains(components, variable)) {
      throw std::runtime_error(fmt::format(
          "The argument `var_list` is missing or erroneous.\nIt should be a list of strings representing species "
          "record components.\n The available quantities for species '{}' are:\n - {}\nPlease set the argument "
          "`var_list` accordingly.",
          species, join_lines(components)));
    }
  }
  const auto& iterations = series.iterations();
  if (!std::binary_search(iterations.begin(), iterations.end(), iteration)) {
    throw std::runtime_error(fmt::format("The requested iteration '{}' is not available in {}", iteration,
                                         series.directory().string()));
  }

  auto file = h5::File::open_read_only(series.file(iteration));
  file.set_raw_reads_enabled(raw_reads);

  ParticleDump dump;
  dump.iteration = iteration;
  const std::vector<std::vector<double>*> targets{&dump.x_m, &dump.y_m, &dump.z_m, &dump.ux,
                                                  &dump.uy,  &dump.uz,  &dump.w};
  for (std::size_t i = 0; i < kVariables.size(); ++i) {
    *targets[i] = read_species_component(file, iteration, species, kVariables[i], info.ed_pic);
  }
  for (std::size_t i = 0; i < kVariables.size(); ++i) {
    if (targets[i]->size() != dump.w.size()) {
      throw std::invalid_argument(fmt::format("Particle array length mismatch for {}: len={}, len(w)={}",
                                              kVariables[i], targets[i]->size(), dump.w.size()));
    }
  }
  try {
    dump.time_fs = iteration_time_s(file, iteration) * 1.0e15;
  } catch (const std::exception&) {
    dump.time_fs = std::numeric_limits<double>::quiet_NaN();
  }
  return dump;
}

}  // namespace guiding::io

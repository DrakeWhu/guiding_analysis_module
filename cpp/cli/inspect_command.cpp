#include <memory>
#include <string>

#include <fmt/format.h>

#include "commands.hpp"
#include "guiding/io/h5.hpp"
#include "guiding/io/openpmd_series.hpp"
#include "guiding/table/py_format.hpp"

namespace guiding::cli {
namespace {

struct InspectArgs {
  std::string diag;
};

std::string json_string(std::string_view text) {
  std::string out = "\"";
  for (char c : text) {
    switch (c) {
      case '"': out += "\\\""; break;
      case '\\': out += "\\\\"; break;
      case '\n': out += "\\n"; break;
      case '\t': out += "\\t"; break;
      default:
        if (static_cast<unsigned char>(c) < 0x20) {
          out += fmt::format("\\u{:04x}", static_cast<unsigned char>(c));
        } else {
          out.push_back(c);
        }
    }
  }
  return out + "\"";
}

const char* kind_name(h5::ScalarKind kind) {
  switch (kind) {
    case h5::ScalarKind::Float32: return "float32";
    case h5::ScalarKind::Float64: return "float64";
    case h5::ScalarKind::Int32: return "int32";
    case h5::ScalarKind::Int64: return "int64";
    case h5::ScalarKind::UInt32: return "uint32";
    case h5::ScalarKind::UInt64: return "uint64";
    default: return "other";
  }
}

std::string dataset_json(h5::Id object) {
  const auto info = h5::dataset_info(object);
  std::string shape;
  for (std::size_t i = 0; i < info.shape.size(); ++i) {
    shape += fmt::format("{}{}", i == 0 ? "" : ", ", info.shape[i]);
  }
  return fmt::format(R"({{"shape": [{}], "dtype": "{}", "raw_read": {}}})", shape, kind_name(info.kind),
                     info.contiguous_offset.has_value() ? "true" : "false");
}

// Iteration range plus the meshes and species of the first iteration.
int run_inspect(const InspectArgs& args) {
  const auto series = io::FileSeries::scan(args.diag);
  const auto& iterations = series.iterations();
  std::string out = fmt::format("{{\n  \"directory\": {},\n  \"n_iterations\": {}",
                                json_string(table::python_path_string(args.diag)), iterations.size());
  if (iterations.empty()) {
    fmt::print("{}\n}}\n", out);
    return 0;
  }
  out += fmt::format(",\n  \"first_iteration\": {},\n  \"last_iteration\": {}", iterations.front(), iterations.back());

  const auto file = h5::File::open_read_only(series.file(iterations.front()));
  std::string meshes;
  if (h5::has_attribute(file.id(), "meshesPath")) {
    const std::string path = io::meshes_path(file.id(), iterations.front());
    if (h5::link_exists(file.id(), path)) {
      const auto group = h5::open_object(file.id(), path);
      for (const auto& record : h5::child_names(group.id())) {
        const auto object = h5::open_object(group.id(), record);
        meshes += fmt::format("{}\n    {}: ", meshes.empty() ? "" : ",", json_string(record));
        if (h5::is_dataset(object.id())) {
          meshes += dataset_json(object.id());
          continue;
        }
        std::string components;
        for (const auto& component : h5::child_names(object.id())) {
          const auto child = h5::open_object(object.id(), component);
          if (h5::is_dataset(child.id())) {
            components += fmt::format("{}{}: {}", components.empty() ? "" : ", ", json_string(component),
                                      dataset_json(child.id()));
          }
        }
        meshes += "{" + components + "}";
      }
    }
  }
  std::string species;
  if (h5::has_attribute(file.id(), "particlesPath")) {
    const std::string path = io::particles_path(file.id(), iterations.front());
    if (h5::link_exists(file.id(), path)) {
      const auto group = h5::open_object(file.id(), path);
      for (const auto& name : h5::child_names(group.id())) {
        species += fmt::format("{}{}", species.empty() ? "" : ", ", json_string(name));
      }
    }
  }
  fmt::print("{},\n  \"meshes\": {{{}\n  }},\n  \"species\": [{}]\n}}\n", out, meshes, species);
  return 0;
}

}  // namespace

Command add_inspect_command(CLI::App& app) {
  auto args = std::make_shared<InspectArgs>();
  auto* command = app.add_subcommand("inspect", "Summarise an openPMD diagnostic directory as JSON");
  command->add_option("--diag", args->diag, "Path to an openPMD diagnostic directory")->required();
  return {command, [args] { return run_inspect(*args); }};
}

}  // namespace guiding::cli

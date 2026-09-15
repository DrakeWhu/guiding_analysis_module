#include <cstdio>
#include <exception>
#include <filesystem>
#include <mutex>
#include <string>

#include <CLI/CLI.hpp>
#include <fmt/format.h>

#include "guiding/exec/parallel.hpp"
#include "guiding/io/h5.hpp"
#include "guiding/io/openpmd_series.hpp"
#include "guiding/products/case_reduction.hpp"
#include "guiding/products/singlecase_score.hpp"
#include "guiding/table/py_format.hpp"

namespace {

namespace fs = std::filesystem;
using guiding::products::python_path_string;
using guiding::table::py_float_repr;

struct CaseArgs {
  std::string diag;
  std::string outdir;
  int stride = 1;
  double smooth_um = 2.0;
  double wake_behind_um = 120.0;
  double wake_gap_um = 5.0;
  double lambda0_m = 0.8e-6;
  bool skip_existing = false;
  bool overwrite = false;
  bool no_plots = false;
  bool no_singlecase_score = false;
  unsigned threads = 0;
  bool no_raw_reads = false;
};

struct InspectArgs {
  std::string diag;
};

void ensure_singlecase_sidecar(const fs::path& csv_path, const std::string& case_id, bool overwrite) {
  fs::path score_path;
  const bool written = guiding::products::ensure_singlecase_guiding_score_csv(csv_path, case_id, overwrite, &score_path);
  if (written) {
    fmt::print("[OK] wrote {}\n", python_path_string(score_path));
  } else {
    fmt::print("[USE] existing single-case guiding score: {}\n", python_path_string(score_path));
  }
}

// scripts/analyze_case.py
int run_case(const CaseArgs& args) {
  const fs::path diag(args.diag);
  const fs::path outdir(args.outdir);
  const fs::path csv_path = outdir / "guiding_metrics.csv";
  const std::string case_id = fs::path(python_path_string(outdir)).filename().string();

  if (fs::exists(csv_path) && args.skip_existing && !args.overwrite) {
    fmt::print("[SKIP] existing {}\n", python_path_string(csv_path));
    if (!args.no_singlecase_score) {
      ensure_singlecase_sidecar(csv_path, case_id, false);
    }
    return 0;
  }
  if (fs::exists(csv_path) && !args.overwrite) {
    throw std::runtime_error(fmt::format("Output already exists: {}. Use --overwrite or --skip-existing.",
                                         python_path_string(csv_path)));
  }
  fs::create_directories(outdir);

  guiding::products::CaseReductionOptions options;
  options.params = {args.stride, args.smooth_um, args.wake_behind_um, args.wake_gap_um, args.lambda0_m};
  options.threads = args.threads == 0 ? guiding::exec::default_thread_count() : args.threads;
  options.raw_reads = !args.no_raw_reads;

  fmt::print("=== Capillary guiding case analysis ===\n");
  fmt::print("diag           = {}\n", python_path_string(diag));
  fmt::print("outdir         = {}\n", python_path_string(outdir));
  fmt::print("stride         = {}\n", args.stride);
  fmt::print("smooth_um      = {}\n", py_float_repr(args.smooth_um));
  fmt::print("wake_behind_um = {}\n", py_float_repr(args.wake_behind_um));
  fmt::print("wake_gap_um    = {}\n", py_float_repr(args.wake_gap_um));
  fmt::print("lambda0_m      = {}\n", py_float_repr(args.lambda0_m));
  fmt::print("threads        = {}\n", options.threads);
  fmt::print("=======================================\n");
  std::fflush(stdout);

  std::mutex print_mutex;
  options.on_iteration = [&](std::int64_t iteration) {
    std::lock_guard lock(print_mutex);
    fmt::print("[READ] iteration {}\n", iteration);
    std::fflush(stdout);
  };

  const auto rows = guiding::products::compute_case_rows(diag, options);
  guiding::products::write_guiding_metrics_csv(rows, csv_path);
  fmt::print("[OK] wrote {}\n", python_path_string(csv_path));

  if (!args.no_singlecase_score) {
    ensure_singlecase_sidecar(csv_path, case_id, true);
  }
  if (!args.no_plots) {
    fmt::print("[INFO] guiding_cli does not render PNG plots; use guiding_gui or the Python plotting scripts\n");
  }
  return 0;
}

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

const char* kind_name(guiding::h5::ScalarKind kind) {
  using guiding::h5::ScalarKind;
  switch (kind) {
    case ScalarKind::Float32: return "float32";
    case ScalarKind::Float64: return "float64";
    case ScalarKind::Int32: return "int32";
    case ScalarKind::Int64: return "int64";
    case ScalarKind::UInt32: return "uint32";
    case ScalarKind::UInt64: return "uint64";
    default: return "other";
  }
}

std::string dataset_json(guiding::h5::Id object) {
  const auto info = guiding::h5::dataset_info(object);
  std::string shape;
  for (std::size_t i = 0; i < info.shape.size(); ++i) {
    shape += fmt::format("{}{}", i == 0 ? "" : ", ", info.shape[i]);
  }
  return fmt::format(R"({{"shape": [{}], "dtype": "{}", "raw_read": {}}})", shape, kind_name(info.kind),
                     info.contiguous_offset.has_value() ? "true" : "false");
}

// Summary of the first iteration: meshes, components and storage layout.
int run_inspect(const InspectArgs& args) {
  namespace h5 = guiding::h5;
  const auto series = guiding::io::FileSeries::scan(args.diag);
  const auto& iterations = series.iterations();
  fmt::print("{{\n  \"directory\": {},\n", json_string(python_path_string(args.diag)));
  fmt::print("  \"n_iterations\": {},\n", iterations.size());
  if (iterations.empty()) {
    fmt::print("  \"iterations\": []\n}}\n");
    return 0;
  }
  fmt::print("  \"first_iteration\": {},\n  \"last_iteration\": {},\n", iterations.front(), iterations.back());

  const auto file = h5::File::open_read_only(series.file(iterations.front()));
  std::string meshes_json;
  if (h5::has_attribute(file.id(), "meshesPath")) {
    const std::string meshes = guiding::io::meshes_path(file.id(), iterations.front());
    if (h5::link_exists(file.id(), meshes)) {
      const auto group = h5::open_object(file.id(), meshes);
      bool first_record = true;
      for (const auto& record : h5::child_names(group.id())) {
        const auto object = h5::open_object(group.id(), record);
        meshes_json += fmt::format("{}\n    {}: ", first_record ? "" : ",", json_string(record));
        first_record = false;
        if (h5::is_dataset(object.id())) {
          meshes_json += dataset_json(object.id());
          continue;
        }
        std::string components;
        bool first_component = true;
        for (const auto& component : h5::child_names(object.id())) {
          const auto child = h5::open_object(object.id(), component);
          if (!h5::is_dataset(child.id())) {
            continue;
          }
          components += fmt::format("{}{}: {}", first_component ? "" : ", ", json_string(component),
                                    dataset_json(child.id()));
          first_component = false;
        }
        meshes_json += "{" + components + "}";
      }
    }
  }
  std::string species_json;
  if (h5::has_attribute(file.id(), "particlesPath")) {
    const std::string particles = guiding::io::particles_path(file.id(), iterations.front());
    if (h5::link_exists(file.id(), particles)) {
      const auto group = h5::open_object(file.id(), particles);
      bool first = true;
      for (const auto& species : h5::child_names(group.id())) {
        species_json += fmt::format("{}{}", first ? "" : ", ", json_string(species));
        first = false;
      }
    }
  }
  fmt::print("  \"meshes\": {{{}\n  }},\n", meshes_json);
  fmt::print("  \"species\": [{}]\n}}\n", species_json);
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  CLI::App app{"guiding_cli: reduction pipeline for WarpX RZ capillary guiding simulations"};
  app.set_version_flag("--version", "guiding_cli 0.1.0");
  app.require_subcommand(1);

  CaseArgs case_args;
  auto* case_cmd = app.add_subcommand("case", "Analyze one WarpX RZ openPMD field diagnostic (scripts/analyze_case.py)");
  case_cmd->add_option("--diag", case_args.diag, "Path to openPMD diagnostic directory, e.g. CASE/diags/diag1")
      ->required();
  case_cmd->add_option("--outdir", case_args.outdir, "Output directory, e.g. analysis_outputs/case_metrics/CASE_ID")
      ->required();
  case_cmd->add_option("--stride", case_args.stride)->capture_default_str();
  case_cmd->add_option("--smooth-um", case_args.smooth_um)->capture_default_str();
  case_cmd->add_option("--wake-behind-um", case_args.wake_behind_um)->capture_default_str();
  case_cmd->add_option("--wake-gap-um", case_args.wake_gap_um)->capture_default_str();
  case_cmd->add_option("--lambda0-m", case_args.lambda0_m,
                       "Laser wavelength [m] used to convert peak transverse E field to a0.")
      ->capture_default_str();
  case_cmd->add_flag("--skip-existing", case_args.skip_existing, "Skip if guiding_metrics.csv already exists.");
  case_cmd->add_flag("--overwrite", case_args.overwrite, "Overwrite existing guiding_metrics.csv.");
  case_cmd->add_flag("--no-plots", case_args.no_plots, "Accepted for compatibility; the CLI never renders plots.");
  case_cmd->add_flag("--no-singlecase-score", case_args.no_singlecase_score,
                     "Do not write guiding_singlecase_score.csv sidecar.");
  case_cmd->add_option("--threads", case_args.threads,
                       "Worker threads (default: GUIDING_THREADS, SLURM_CPUS_PER_TASK or all available cores)");
  case_cmd->add_flag("--no-raw-reads", case_args.no_raw_reads,
                     "Read every dataset through HDF5 (disables the pread fast path)");

  InspectArgs inspect_args;
  auto* inspect_cmd = app.add_subcommand("inspect", "Summarise an openPMD diagnostic directory as JSON");
  inspect_cmd->add_option("--diag", inspect_args.diag, "Path to an openPMD diagnostic directory")->required();

  CLI11_PARSE(app, argc, argv);

  try {
    if (*case_cmd) {
      return run_case(case_args);
    }
    if (*inspect_cmd) {
      return run_inspect(inspect_args);
    }
  } catch (const std::exception& error) {
    std::fflush(stdout);
    fmt::print(stderr, "[FAIL] {}\n", error.what());
    return 1;
  }
  return 0;
}

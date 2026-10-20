#include <filesystem>
#include <memory>
#include <stdexcept>

#include <fmt/format.h>

#include "commands.hpp"
#include "guiding/products/case_reduction.hpp"
#include "guiding/table/py_format.hpp"

namespace guiding::cli {
namespace {

namespace fs = std::filesystem;
using table::py_float_repr;
using table::python_path_string;

struct CaseArgs {
  std::string diag;
  std::string outdir;
  bool skip_existing = false;
  bool overwrite = false;
  bool no_plots = false;
  bool no_singlecase_score = false;
  FieldOptions field;
};

// scripts/analyze_case.py
int run_case(const CaseArgs& args) {
  const fs::path diag(args.diag);
  const fs::path outdir(args.outdir);
  const fs::path csv_path = outdir / "guiding_metrics.csv";
  const std::string case_id = fs::path(python_path_string(outdir)).filename().string();

  if (fs::exists(csv_path) && args.skip_existing && !args.overwrite) {
    print_line(fmt::format("[SKIP] existing {}", python_path_string(csv_path)));
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

  const auto options = reduction_options(args.field);
  print_line("=== Capillary guiding case analysis ===");
  print_line(fmt::format("diag           = {}", python_path_string(diag)));
  print_line(fmt::format("outdir         = {}", python_path_string(outdir)));
  print_line(fmt::format("stride         = {}", args.field.stride));
  print_line(fmt::format("smooth_um      = {}", py_float_repr(args.field.smooth_um)));
  print_line(fmt::format("wake_behind_um = {}", py_float_repr(args.field.wake_behind_um)));
  print_line(fmt::format("wake_gap_um    = {}", py_float_repr(args.field.wake_gap_um)));
  print_line(fmt::format("lambda0_m      = {}", py_float_repr(args.field.lambda0_m)));
  print_line(fmt::format("threads        = {}", options.threads));
  print_line("=======================================");

  const auto rows = products::compute_case_rows(diag, options);
  products::write_guiding_metrics_csv(rows, csv_path);
  print_line(fmt::format("[OK] wrote {}", python_path_string(csv_path)));

  if (!args.no_singlecase_score) {
    ensure_singlecase_sidecar(csv_path, case_id, true);
  }
  if (!args.no_plots) {
    print_line("[INFO] guiding_cli does not render PNG plots; use guiding_gui or the Python plotting scripts");
  }
  return 0;
}

}  // namespace

Command add_case_command(CLI::App& app) {
  auto args = std::make_shared<CaseArgs>();
  auto* command = app.add_subcommand("case", "Analyze one WarpX RZ openPMD field diagnostic (scripts/analyze_case.py)");
  command->add_option("--diag", args->diag, "Path to openPMD diagnostic directory, e.g. CASE/diags/diag1")->required();
  command->add_option("--outdir", args->outdir, "Output directory, e.g. analysis_outputs/case_metrics/CASE_ID")
      ->required();
  add_field_options(*command, args->field);
  command->add_flag("--skip-existing", args->skip_existing, "Skip if guiding_metrics.csv already exists.");
  command->add_flag("--overwrite", args->overwrite, "Overwrite existing guiding_metrics.csv.");
  command->add_flag("--no-plots", args->no_plots, "Accepted for compatibility; the CLI never renders plots.");
  command->add_flag("--no-singlecase-score", args->no_singlecase_score,
                    "Do not write guiding_singlecase_score.csv sidecar.");
  return {command, [args] { return run_case(*args); }};
}

}  // namespace guiding::cli

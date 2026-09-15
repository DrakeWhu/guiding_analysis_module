#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "commands.hpp"
#include "guiding/products/triplet_tables.hpp"
#include "guiding/table/py_format.hpp"

namespace guiding::cli {
namespace {

namespace fs = std::filesystem;
using table::python_path_string;

struct TripletArgs {
  std::string channel;
  std::string uniform;
  std::string vacuum;
  std::string outdir;
  std::string label = "triplet";
  double late_fraction = 1.0 / 3.0;
  bool skip_existing = false;
  bool overwrite = false;
  bool no_plots = false;
};

// scripts/compare_triplet.py (late tables are echoed as CSV rather than
// pandas.DataFrame.to_string).
int run_triplet(const TripletArgs& args) {
  const fs::path outdir = args.outdir.empty() ? fs::path(args.channel).parent_path() : fs::path(args.outdir);
  const fs::path wide_path = outdir / "guiding_triplet_wide.csv";

  if (fs::exists(wide_path) && args.skip_existing && !args.overwrite) {
    print_line(fmt::format("[SKIP] existing {}", python_path_string(wide_path)));
    return 0;
  }
  if (fs::exists(wide_path) && !args.overwrite) {
    throw std::runtime_error(fmt::format("Output already exists: {}. Use --overwrite or --skip-existing.",
                                         python_path_string(wide_path)));
  }

  print_line("=== Capillary guiding triplet comparison ===");
  print_line(fmt::format("channel       = {}", args.channel));
  print_line(fmt::format("uniform       = {}", args.uniform));
  print_line(fmt::format("vacuum        = {}", args.vacuum));
  print_line(fmt::format("outdir        = {}", python_path_string(outdir)));
  print_line(fmt::format("label         = {}", args.label));
  print_line(fmt::format("late_fraction = {}", table::py_float_repr(args.late_fraction)));
  print_line("============================================");

  std::vector<fs::path> missing;
  for (const auto& path : {fs::path(args.channel), fs::path(args.uniform), fs::path(args.vacuum)}) {
    if (!fs::is_regular_file(path)) {
      missing.push_back(path);
    }
  }
  if (!missing.empty()) {
    print_line("");
    print_line("[ERROR] Missing guiding_metrics.csv file(s):");
    for (const auto& path : missing) {
      print_line(fmt::format("  - {}", python_path_string(path)));
    }
    print_line("");
    print_line("compare_triplet only compares existing CSVs. To generate missing case metrics from WarpX "
               "diagnostics, use `guiding_cli campaign --run-cases` or `guiding_cli case`.");
    return 2;
  }

  const auto tables = products::build_triplet_tables(args.channel, args.uniform, args.vacuum, args.label,
                                                     args.late_fraction);
  const auto paths = products::write_triplet_tables(tables, outdir);
  for (const auto& path : {paths.long_table, paths.wide, paths.late_summary, paths.late_ratios}) {
    print_line(fmt::format("[OK] wrote {}", python_path_string(path)));
  }

  const auto iterations = tables.wide.doubles("iteration");
  const auto propagation = tables.wide.doubles("propagation_mm");
  print_line("");
  print_line("=== Common iterations ===");
  print_line(fmt::format("n_common = {}", tables.wide.row_count()));
  print_line(fmt::format("first,last iteration = {}, {}", static_cast<long long>(iterations.front()),
                         static_cast<long long>(iterations.back())));
  print_line(fmt::format("first,last propagation = {:.3f}, {:.3f} mm", propagation.front(), propagation.back()));
  print_line("");
  print_line("=== Late summary ===");
  fmt::print("{}", tables.late_summary.to_pandas_csv());
  print_line("");
  print_line("=== Late ratios ===");
  fmt::print("{}", tables.late_ratios.to_pandas_csv());
  if (!args.no_plots) {
    print_line("[INFO] guiding_cli does not render PNG plots; use guiding_gui or the Python plotting scripts");
  }
  return 0;
}

}  // namespace

Command add_triplet_command(CLI::App& app) {
  auto args = std::make_shared<TripletArgs>();
  auto* command = app.add_subcommand(
      "triplet", "Compare one channel/uniform/vacuum triplet from guiding_metrics.csv files (scripts/compare_triplet.py)");
  command->add_option("--channel", args->channel, "channel guiding_metrics.csv")->required();
  command->add_option("--uniform", args->uniform, "uniform guiding_metrics.csv")->required();
  command->add_option("--vacuum", args->vacuum, "vacuum guiding_metrics.csv")->required();
  command->add_option("--outdir", args->outdir, "Triplet output directory. Defaults to the channel CSV parent directory.");
  command->add_option("--label", args->label)->capture_default_str();
  command->add_option("--late-fraction", args->late_fraction)->capture_default_str();
  command->add_flag("--skip-existing", args->skip_existing, "Skip if guiding_triplet_wide.csv already exists.");
  command->add_flag("--overwrite", args->overwrite, "Overwrite existing triplet CSVs.");
  command->add_flag("--no-plots", args->no_plots, "Accepted for compatibility; the CLI never renders plots.");
  return {command, [args] { return run_triplet(*args); }};
}

}  // namespace guiding::cli

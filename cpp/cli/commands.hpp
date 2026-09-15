#pragma once

#include <filesystem>
#include <functional>
#include <string>
#include <utility>

#include <CLI/CLI.hpp>

#include "guiding/products/case_reduction.hpp"

namespace guiding::cli {

using Runner = std::function<int()>;
using Command = std::pair<CLI::App*, Runner>;

// Each function registers one subcommand and returns the code that runs it.
Command add_case_command(CLI::App& app);
Command add_campaign_command(CLI::App& app);
Command add_triplet_command(CLI::App& app);
Command add_inspect_command(CLI::App& app);

// Field-reduction options shared by `case` and `campaign`.
struct FieldOptions {
  int stride = 1;
  double smooth_um = 2.0;
  double wake_behind_um = 120.0;
  double wake_gap_um = 5.0;
  double lambda0_m = 0.8e-6;
  unsigned threads = 0;
  bool no_raw_reads = false;
};

void add_field_options(CLI::App& command, FieldOptions& options);
// Reduction options with "[READ] iteration N" progress lines.
[[nodiscard]] products::CaseReductionOptions reduction_options(const FieldOptions& options);

// Writes or reuses guiding_singlecase_score.csv next to guiding_metrics.csv
// and prints "[OK] wrote" or "[USE] existing ..." like the Python workflow.
void ensure_singlecase_sidecar(const std::filesystem::path& csv_path, const std::string& case_id, bool overwrite);

// Serialises console output from worker threads.
void print_line(const std::string& line);

}  // namespace guiding::cli

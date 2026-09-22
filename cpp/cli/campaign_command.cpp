#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "commands.hpp"
#include "guiding/campaign/discovery.hpp"
#include "guiding/products/case_reduction.hpp"
#include "guiding/products/triplet_tables.hpp"
#include "guiding/table/csv.hpp"
#include "guiding/table/py_format.hpp"

namespace guiding::cli {
namespace {

namespace fs = std::filesystem;
using campaign::CaseInfo;
using campaign::TripletInfo;
using table::py_float_repr;
using table::python_path_string;

struct CampaignArgs {
  std::string campaign_root;
  std::string outdir = "analysis_outputs/campaign";
  std::string case_metrics_root;
  std::string triplets_root;
  bool run_cases = false;
  bool run_triplets = false;
  bool skip_existing = false;
  bool overwrite_cases = false;
  bool overwrite_triplets = false;
  std::int64_t min_h5 = 2;
  double min_last_h5_age_min = 0.0;
  double late_fraction = 1.0 / 3.0;
  bool no_case_plots = false;
  bool plot_existing_cases = false;
  bool overwrite_case_plots = false;
  bool no_triplet_plots = false;
  FieldOptions field;
};

fs::path case_metrics_path(const fs::path& root, const std::string& case_id) {
  return root / case_id / "guiding_metrics.csv";
}

// analyze_campaign.py:_case_metrics_is_valid
bool case_metrics_is_valid(const fs::path& path) {
  std::error_code error;
  if (!fs::is_regular_file(path, error)) {
    return false;
  }
  try {
    const auto table = table::read_csv_file(path);
    if (table.rows.empty()) {
      return false;
    }
    for (const auto& column : products::triplet_required_columns()) {
      if (!table.has_column(column)) {
        return false;
      }
    }
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

std::string age_text(const CaseInfo& info, double now) {
  const auto age = campaign::newest_h5_age_min(info, now);
  return age ? fmt::format("{:.2f} min", *age) : "none";
}

std::vector<const CaseInfo*> members(const TripletInfo& triplet) {
  return {&*triplet.channel, &*triplet.uniform, &*triplet.vacuum};
}

// workflows.py:ensure_case_metrics (plots are never rendered here).
void ensure_case_metrics(const CaseInfo& info, const fs::path& case_metrics_root, const FieldOptions& field,
                         bool overwrite) {
  const fs::path outdir = case_metrics_root / info.case_id;
  const fs::path csv_path = outdir / "guiding_metrics.csv";
  if (fs::exists(csv_path) && !overwrite) {
    print_line(fmt::format("[USE] existing case metrics: {}", python_path_string(csv_path)));
    ensure_singlecase_sidecar(csv_path, info.case_id, false);
    return;
  }
  print_line(fmt::format("[MAKE] case metrics for {}", info.case_id));
  print_line(fmt::format("       diag   = {}", python_path_string(info.diag_dir)));
  print_line(fmt::format("       outdir = {}", python_path_string(outdir)));
  const auto rows = products::compute_case_rows(info.diag_dir, reduction_options(field));
  products::write_guiding_metrics_csv(rows, csv_path);
  print_line(fmt::format("[OK] wrote {}", python_path_string(csv_path)));
  ensure_singlecase_sidecar(csv_path, info.case_id, true);
}

// scripts/analyze_campaign.py
int run_campaign(const CampaignArgs& args) {
  const fs::path campaign_root(args.campaign_root);
  const fs::path outdir(args.outdir);
  const fs::path case_metrics_root =
      args.case_metrics_root.empty() ? outdir / "case_metrics" : fs::path(args.case_metrics_root);
  const std::optional<fs::path> triplets_root =
      args.triplets_root.empty() ? std::nullopt : std::optional<fs::path>(args.triplets_root);
  const double now = campaign::unix_time_now();

  const auto cases = campaign::discover_cases(campaign_root);
  const auto triplets = campaign::build_triplets(cases);

  std::vector<const TripletInfo*> complete;
  std::size_t incomplete = 0;
  for (const auto& triplet : triplets) {
    if (triplet.complete()) {
      complete.push_back(&triplet);
    } else {
      ++incomplete;
    }
  }
  auto has_valid_metrics = [&](const CaseInfo& info) {
    return case_metrics_is_valid(case_metrics_path(case_metrics_root, info.case_id));
  };
  auto usable = [&](const CaseInfo& info) {
    return has_valid_metrics(info) || campaign::case_is_ready(info, args.min_h5, args.min_last_h5_age_min, now);
  };

  std::size_t ready_min_h5 = 0;
  std::size_t ready_for_analysis = 0;
  std::size_t triplets_csv_complete = 0;
  std::size_t triplets_usable = 0;
  for (const auto* triplet : complete) {
    ready_min_h5 += campaign::triplet_ready_min_h5(*triplet, args.min_h5) ? 1 : 0;
    ready_for_analysis += campaign::triplet_is_ready(*triplet, args.min_h5, args.min_last_h5_age_min, now) ? 1 : 0;
    const auto triplet_members = members(*triplet);
    triplets_csv_complete +=
        std::all_of(triplet_members.begin(), triplet_members.end(), [&](auto* c) { return has_valid_metrics(*c); });
    triplets_usable += std::all_of(triplet_members.begin(), triplet_members.end(), [&](auto* c) { return usable(*c); });
  }
  std::size_t cases_with_valid_metrics = 0;
  std::size_t cases_usable = 0;
  std::size_t insufficient = 0;
  std::size_t unstable = 0;
  for (const auto& info : cases) {
    cases_with_valid_metrics += has_valid_metrics(info) ? 1 : 0;
    cases_usable += usable(info) ? 1 : 0;
    insufficient += info.h5_count < args.min_h5 ? 1 : 0;
    unstable += (info.h5_count >= args.min_h5 &&
                 !campaign::case_is_ready(info, args.min_h5, args.min_last_h5_age_min, now))
                    ? 1
                    : 0;
  }

  const auto reports =
      campaign::write_campaign_report(cases, triplets, outdir, args.min_h5, args.min_last_h5_age_min, now);

  print_line("=== Campaign dry-run summary ===");
  print_line(fmt::format("campaign_root        = {}", python_path_string(campaign_root)));
  print_line(fmt::format("outdir               = {}", python_path_string(outdir)));
  print_line(fmt::format("case_metrics_root    = {}", python_path_string(case_metrics_root)));
  print_line(fmt::format("triplets_root        = {}",
                         triplets_root ? python_path_string(*triplets_root) : "<channel metrics dir>"));
  print_line(fmt::format("cases detected       = {}", cases.size()));
  print_line(fmt::format("triplets structural complete = {}", complete.size()));
  print_line(fmt::format("triplets ready min_h5        = {}", ready_min_h5));
  print_line(fmt::format("triplets ready for analysis  = {}", ready_for_analysis));
  print_line(fmt::format("cases with valid CSV metrics = {}", cases_with_valid_metrics));
  print_line(fmt::format("cases usable CSV or HDF5     = {}", cases_usable));
  print_line(fmt::format("triplets CSV complete        = {}", triplets_csv_complete));
  print_line(fmt::format("triplets usable CSV or HDF5  = {}", triplets_usable));
  print_line(fmt::format("triplets incomplete          = {}", incomplete));
  print_line(fmt::format("cases insufficient h5        = {}", insufficient));
  print_line(fmt::format("cases unstable h5 age        = {}", unstable));
  print_line(fmt::format("min_h5                       = {}", args.min_h5));
  print_line(fmt::format("min_last_h5_age_min          = {}", py_float_repr(args.min_last_h5_age_min)));
  print_line("reports:");
  for (const auto& path : {reports.cases, reports.triplets, reports.insufficient_h5, reports.unstable_h5}) {
    print_line(fmt::format("  - {}", python_path_string(path)));
  }
  print_line("===============================");

  if (!args.run_cases && !args.run_triplets && !args.plot_existing_cases) {
    return 0;
  }

  int status = 0;
  if (args.run_cases) {
    for (const auto& info : cases) {
      const fs::path csv_path = case_metrics_path(case_metrics_root, info.case_id);
      const bool valid = case_metrics_is_valid(csv_path);
      if (valid && !args.overwrite_cases) {
        print_line(fmt::format("[USE] valid existing case metrics: {}", python_path_string(csv_path)));
        ensure_singlecase_sidecar(csv_path, info.case_id, false);
        continue;
      }
      if (fs::exists(csv_path) && !valid && !args.overwrite_cases) {
        print_line(fmt::format("[SKIP] existing INVALID case metrics, use --overwrite-cases to regenerate: {}",
                               python_path_string(csv_path)));
        continue;
      }
      if (!campaign::case_is_ready(info, args.min_h5, args.min_last_h5_age_min, now)) {
        print_line(fmt::format("[SKIP] case not ready (h5={}, min_h5={}, newest_h5_age={}, min_last_h5_age_min={}): {}",
                               info.h5_count, args.min_h5, age_text(info, now),
                               py_float_repr(args.min_last_h5_age_min), info.case_id));
        continue;
      }
      try {
        ensure_case_metrics(info, case_metrics_root, args.field, args.overwrite_cases);
      } catch (const std::exception& error) {
        // The Python script aborts the whole campaign here; keep going instead.
        print_line(fmt::format("[FAIL] case {}: {}", info.case_id, error.what()));
        status = 1;
      }
    }
    if (!args.no_case_plots) {
      print_line("[INFO] case plots are not rendered by guiding_cli; use guiding_gui or --plot-existing-cases in Python");
    }
  }

  if (args.plot_existing_cases) {
    print_line("[INFO] --plot-existing-cases renders PNGs, which guiding_cli does not do; "
               "run scripts/analyze_campaign.py --plot-existing-cases or open the campaign in guiding_gui");
  }

  if (args.run_triplets) {
    for (const auto* triplet : complete) {
      const std::string label = triplet->label();
      const fs::path channel_csv = case_metrics_path(case_metrics_root, triplet->channel->case_id);
      const fs::path uniform_csv = case_metrics_path(case_metrics_root, triplet->uniform->case_id);
      const fs::path vacuum_csv = case_metrics_path(case_metrics_root, triplet->vacuum->case_id);

      std::vector<fs::path> invalid;
      for (const auto& path : {channel_csv, uniform_csv, vacuum_csv}) {
        if (!case_metrics_is_valid(path)) {
          invalid.push_back(path);
        }
      }
      if (!invalid.empty()) {
        print_line(fmt::format("[SKIP] missing/invalid case metrics for triplet {}:", label));
        for (const auto& path : invalid) {
          print_line(fmt::format("       - {}", python_path_string(path)));
        }
        print_line("       HDF5 status:");
        for (const auto* member : members(*triplet)) {
          print_line(fmt::format("       - {}: h5={}, newest_h5_age={}", member->case_id, member->h5_count,
                                 age_text(*member, now)));
        }
        continue;
      }

      const fs::path triplet_outdir = triplets_root ? *triplets_root / label : channel_csv.parent_path();
      const fs::path wide_path = triplet_outdir / "guiding_triplet_wide.csv";
      if (fs::exists(wide_path) && args.skip_existing && !args.overwrite_triplets) {
        print_line(fmt::format("[SKIP] existing triplet: {}", python_path_string(wide_path)));
        continue;
      }
      if (fs::exists(wide_path) && !args.overwrite_triplets) {
        print_line(fmt::format("[SKIP] existing triplet, use --overwrite-triplets: {}", python_path_string(wide_path)));
        continue;
      }

      products::TripletTables tables;
      try {
        tables = products::build_triplet_tables(channel_csv, uniform_csv, vacuum_csv, label, args.late_fraction);
      } catch (const std::exception& error) {
        print_line(fmt::format("[FAIL] triplet {}: {}", label, error.what()));
        continue;
      }
      const auto paths = products::write_triplet_tables(tables, triplet_outdir);
      for (const auto& path : {paths.long_table, paths.wide, paths.late_summary, paths.late_ratios}) {
        print_line(fmt::format("[OK] wrote {}", python_path_string(path)));
      }
    }
  }
  return status;
}

}  // namespace

Command add_campaign_command(CLI::App& app) {
  auto args = std::make_shared<CampaignArgs>();
  auto* command =
      app.add_subcommand("campaign", "Dry-run or analyze a full CLPU capillary guiding campaign (scripts/analyze_campaign.py)");
  command->add_option("--campaign-root", args->campaign_root, "Root containing case folders or cases_full.tsv")
      ->required();
  command->add_option("--outdir", args->outdir, "Campaign analysis output root")->capture_default_str();
  command->add_option("--case-metrics-root", args->case_metrics_root, "Defaults to OUTDIR/case_metrics");
  command->add_option("--triplets-root", args->triplets_root,
                      "Optional legacy central root for triplet outputs. If omitted, each triplet is written next to "
                      "the channel guiding_metrics.csv.");
  command->add_flag("--run-cases", args->run_cases, "Generate missing per-case guiding_metrics.csv");
  command->add_flag("--run-triplets", args->run_triplets, "Generate triplet CSVs for complete ready triplets");
  command->add_flag("--skip-existing", args->skip_existing, "Skip existing case/triplet outputs");
  command->add_flag("--overwrite-cases", args->overwrite_cases, "Regenerate existing case metrics");
  command->add_flag("--overwrite-triplets", args->overwrite_triplets, "Regenerate existing triplet outputs");
  command->add_option("--min-h5", args->min_h5)->capture_default_str();
  command->add_option("--min-last-h5-age-min", args->min_last_h5_age_min,
                      "Require the newest HDF5 file of each case to be at least this many minutes old before "
                      "analyzing it. Default 0 disables the stability gate.")
      ->capture_default_str();
  add_field_options(*command, args->field);
  command->add_option("--late-fraction", args->late_fraction)->capture_default_str();
  command->add_flag("--no-case-plots", args->no_case_plots, "Accepted for compatibility");
  command->add_flag("--plot-existing-cases", args->plot_existing_cases,
                    "Accepted for compatibility; PNG rendering stays in the Python pipeline");
  command->add_flag("--overwrite-case-plots", args->overwrite_case_plots, "Accepted for compatibility");
  command->add_flag("--no-triplet-plots", args->no_triplet_plots, "Accepted for compatibility");
  return {command, [args] { return run_campaign(*args); }};
}

}  // namespace guiding::cli

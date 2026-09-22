#include <cstdint>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "commands.hpp"
#include "guiding/products/score_runs.hpp"
#include "guiding/table/frame.hpp"
#include "guiding/table/py_format.hpp"

namespace guiding::cli {
namespace {

namespace fs = std::filesystem;
using table::python_path_string;

struct CommonArgs {
  std::string campaign_root;
  std::string case_metrics_root;
  std::string outdir;
  std::size_t top = 50;
};

// The Python scripts default the metrics root to the campaign root.
fs::path metrics_root_of(const CommonArgs& args) {
  return args.case_metrics_root.empty() ? fs::path(args.campaign_root) : fs::path(args.case_metrics_root);
}

fs::path outdir_of(const CommonArgs& args, const std::string& prefix) {
  if (!args.outdir.empty()) {
    return args.outdir;
  }
  return fs::path(args.campaign_root) / "analysis_outputs" / products::timestamped(prefix);
}

void add_common(CLI::App& command, CommonArgs& args, std::size_t default_top) {
  args.top = default_top;
  command.add_option("--campaign-root", args.campaign_root)->required();
  command.add_option("--case-metrics-root", args.case_metrics_root, "Defaults to CAMPAIGN_ROOT");
  command.add_option("--outdir", args.outdir, "Defaults to CAMPAIGN_ROOT/analysis_outputs/<name>_<timestamp>");
  command.add_option("--top", args.top)->capture_default_str();
}

void add_score_config(CLI::App& command, products::ScoreConfig& config) {
  command.add_option("--entry-window-mm", config.entry_window_mm)->capture_default_str();
  command.add_option("--exit-before-mm", config.exit_before_mm)->capture_default_str();
  command.add_option("--exit-after-mm", config.exit_after_mm)->capture_default_str();
  command.add_option("--a0-target", config.a0_target)->capture_default_str();
  command.add_option("--a0-component-cap", config.a0_component_cap)->capture_default_str();
  command.add_option("--waist-growth-sigma", config.waist_growth_sigma)->capture_default_str();
  command.add_option("--waist-jitter-sigma", config.waist_jitter_sigma)->capture_default_str();
}

// A compact stand-in for pandas' to_string of the top table.
void print_top_table(const table::Frame& frame, const std::vector<std::string>& columns) {
  std::vector<std::string> shown;
  for (const auto& column : columns) {
    if (frame.has(column)) {
      shown.push_back(column);
    }
  }
  if (frame.row_count() == 0 || shown.empty()) {
    return;
  }
  std::string header;
  for (const auto& column : shown) {
    header += (header.empty() ? "" : "  ") + column;
  }
  print_line(header);
  for (std::size_t row = 0; row < frame.row_count(); ++row) {
    std::string line;
    for (const auto& column : shown) {
      const auto& values = frame.column(column);
      std::string text;
      if (const auto* texts = std::get_if<table::Frame::Strings>(&values)) {
        text = (*texts)[row];
      } else if (const auto* integers = std::get_if<table::Frame::Integers>(&values)) {
        text = std::to_string((*integers)[row]);
      } else {
        text = fmt::format("{:.6g}", std::get<table::Frame::Doubles>(values)[row]);
      }
      line += (line.empty() ? "" : "  ") + text;
    }
    print_line(line);
  }
}

struct CampaignArgs {
  CommonArgs common;
  std::string case_type = "channel";
  products::ScoreConfig config;
};

int run_score_campaign(const CampaignArgs& args) {
  const fs::path outdir = outdir_of(args.common, "scored");
  const auto run = products::run_score_campaign(args.common.campaign_root, metrics_root_of(args.common), outdir,
                                                args.case_type, args.config, args.common.top);
  print_line("=== Guiding score summary ===");
  print_line(fmt::format("campaign_root     = {}", python_path_string(args.common.campaign_root)));
  print_line(fmt::format("case_metrics_root = {}", python_path_string(metrics_root_of(args.common))));
  print_line(fmt::format("outdir            = {}", python_path_string(outdir)));
  print_line(fmt::format("case_type         = {}", args.case_type));
  print_line(fmt::format("cases scored      = {}", run.cases_scored));
  print_line(fmt::format("ok scores         = {}", run.tables.ok));
  print_line(fmt::format("failed scores     = {}", run.cases_scored - run.tables.ok));
  for (const auto& [name, path] : run.files) {
    print_line(fmt::format("{:18}= {}", name, python_path_string(path)));
  }
  print_line("");
  print_line(fmt::format("Top {} cases:", run.tables.top.row_count()));
  print_top_table(run.tables.top, {"rank", "score", "case_id", "a0_exit", "a0_exit_over_analysis_max", "waist_growth",
                                   "waist_jitter_log", "valid_fraction", "plateau", "diameter", "focus"});
  return 0;
}

struct TripletArgs {
  CommonArgs common;
  products::ScoreConfig case_config;
  products::TripletScoreConfig triplet_config;
};

int run_score_triplets(const TripletArgs& args) {
  const fs::path outdir = outdir_of(args.common, "triplet_scores");
  const auto run = products::run_score_triplets(args.common.campaign_root, metrics_root_of(args.common), outdir,
                                                args.case_config, args.triplet_config, args.common.top);
  print_line("=== Triplet score summary ===");
  print_line(fmt::format("campaign_root      = {}", python_path_string(args.common.campaign_root)));
  print_line(fmt::format("case_metrics_root  = {}", python_path_string(metrics_root_of(args.common))));
  print_line(fmt::format("outdir             = {}", python_path_string(outdir)));
  print_line(fmt::format("triplets total     = {}", run.triplets_total));
  print_line(fmt::format("triplets scored    = {}", run.triplets_scored));
  print_line(fmt::format("triplets ok        = {}", run.result.tables.ok));
  print_line(fmt::format("triplets failed    = {}", run.triplets_scored - run.result.tables.ok));
  print_line(fmt::format("incomplete skipped = {}", run.result.skipped_incomplete));
  for (const auto& [name, path] : run.files) {
    print_line(fmt::format("{:19}= {}", name, python_path_string(path)));
  }
  print_line("");
  print_line(fmt::format("Top {} triplets:", run.result.tables.top.row_count()));
  print_top_table(run.result.tables.top,
                  {"rank", "final_score", "reference_factor", "score_channel", "score_uniform", "score_vacuum",
                   "reference_kind", "channel_case_id", "plateau", "diameter", "focus"});
  return 0;
}

struct BeamlikeArgs {
  CommonArgs common;
  std::string particle_outdir_name = "particle_analysis";
  std::string row_selection = "single";
  products::BeamlikePairConfig config;
};

void print_pair_rows(const std::vector<table::Record>& rows) {
  static const std::vector<std::string> columns{
      "rank", "beamlike_gain_score", "beamlike_reference_factor", "beamlike_score_channel", "beamlike_score_uniform",
      "beamlike_score_delta", "charge_hot_pC_channel", "charge_hot_pC_uniform", "E95_hot_MeV_channel",
      "E95_hot_MeV_uniform", "mono_proxy_E95_over_Emax_channel", "mono_proxy_E95_over_Emax_uniform",
      "channel_case_id", "uniform_case_id", "plateau", "diameter", "focus"};
  for (const auto& row : rows) {
    std::string line;
    for (const auto& column : columns) {
      if (const auto cell = row.get(column)) {
        line += (line.empty() ? "" : "  ") + fmt::format("{}={}", column, products::py_str(*cell));
      }
    }
    print_line("  " + line);
  }
}

int run_score_beamlike_pairs(const BeamlikeArgs& args) {
  const fs::path outdir = outdir_of(args.common, "beamlike_pairs");
  const auto selection = products::parse_row_selection(args.row_selection);
  const auto run = products::run_score_beamlike_pairs(args.common.campaign_root, metrics_root_of(args.common),
                                                      args.particle_outdir_name, outdir, selection, args.config,
                                                      args.common.top);
  const std::size_t ok = run.tables.positive.size() + run.tables.neutral.size() + run.tables.negative.size();
  print_line("=== Beamlike channel-vs-uniform summary ===");
  print_line(fmt::format("campaign_root           = {}", python_path_string(args.common.campaign_root)));
  print_line(fmt::format("case_metrics_root       = {}", python_path_string(metrics_root_of(args.common))));
  print_line(fmt::format("particle_outdir_name    = {}", args.particle_outdir_name));
  print_line(fmt::format("row_selection           = {}", args.row_selection));
  print_line(fmt::format("outdir                  = {}", python_path_string(outdir)));
  print_line(fmt::format("cases discovered        = {}", run.cases_discovered));
  print_line(fmt::format("pairs attempted         = {}", run.tables.attempted));
  print_line(fmt::format("pairs ok                = {}", ok));
  print_line(fmt::format("pairs positive          = {}", run.tables.positive.size()));
  print_line(fmt::format("pairs neutral           = {}", run.tables.neutral.size()));
  print_line(fmt::format("pairs negative          = {}", run.tables.negative.size()));
  print_line(fmt::format("pairs failed            = {}", run.tables.failed.size()));
  print_line(fmt::format("skipped no channel      = {}", run.tables.skipped_without_channel));
  print_line(fmt::format("skipped no uniform      = {}", run.tables.skipped_without_uniform));
  for (const auto& [name, path] : run.files) {
    print_line(fmt::format("{:24}= {}", name, python_path_string(path)));
  }
  print_line("");
  if (!run.top_rows.empty()) {
    print_line(fmt::format("Top {} positive beamlike channel-vs-uniform pairs:", run.top_rows.size()));
    print_pair_rows(run.top_rows);
  } else {
    print_line("No positive beamlike channel-vs-uniform pairs.");
  }
  print_line("");
  if (!run.worst_rows.empty()) {
    print_line(fmt::format("Worst {} negative beamlike channel-vs-uniform pairs:", run.worst_rows.size()));
    print_pair_rows(run.worst_rows);
  } else {
    print_line("No negative beamlike channel-vs-uniform pairs.");
  }
  if (!run.tables.failed.empty()) {
    print_line("");
    print_line("First failures:");
    for (std::size_t i = 0; i < std::min<std::size_t>(10, run.tables.failed.size()); ++i) {
      const auto& row = run.tables.failed[i];
      const auto text = [&](const char* key) {
        const auto cell = row.get(key);
        return cell ? products::py_str(*cell) : std::string();
      };
      print_line(fmt::format("  {} vs {}: {}", text("channel_case_id"), text("uniform_case_id"),
                             text("failure_reason")));
    }
  }
  return 0;
}

struct JointArgs {
  std::string campaign_root;
  std::string triplet_scores_csv;
  std::string beamlike_pair_scores_csv;
  std::string outdir;
  std::string join_how = "inner";
  std::size_t top = 50;
};

int run_joint(const JointArgs& args) {
  if (args.campaign_root.empty() && (args.triplet_scores_csv.empty() || args.beamlike_pair_scores_csv.empty())) {
    throw std::runtime_error(
        "Provide --campaign-root or both --triplet-scores-csv and --beamlike-pair-scores-csv.");
  }
  const fs::path triplet_csv =
      args.triplet_scores_csv.empty()
          ? products::latest_analysis_file(args.campaign_root, "triplet_scores_*", "triplet_scores.csv")
          : fs::path(args.triplet_scores_csv);
  const fs::path pair_csv =
      args.beamlike_pair_scores_csv.empty()
          ? products::latest_analysis_file(args.campaign_root, "beamlike_pairs_*", "beamlike_pair_scores.csv")
          : fs::path(args.beamlike_pair_scores_csv);
  const fs::path outdir = !args.outdir.empty() ? fs::path(args.outdir)
                          : !args.campaign_root.empty()
                              ? fs::path(args.campaign_root) / "analysis_outputs" /
                                    products::timestamped("guiding_beamlike_joint")
                              : fs::current_path() / products::timestamped("guiding_beamlike_joint");

  const auto run = products::run_joint_scores(triplet_csv, pair_csv, outdir, products::parse_join_how(args.join_how),
                                              args.top);
  print_line("=== Guiding + beamlike joint summary ===");
  print_line(fmt::format("triplet_scores_csv        = {}", python_path_string(triplet_csv)));
  print_line(fmt::format("beamlike_pair_scores_csv  = {}", python_path_string(pair_csv)));
  print_line(fmt::format("join_how                  = {}", args.join_how));
  print_line(fmt::format("outdir                    = {}", python_path_string(outdir)));
  print_line(fmt::format("joined pairs              = {}", run.joined.row_count()));
  print_line("");
  for (const char* column : {"joint_bucket", "triple_bucket"}) {
    const auto counts = products::bucket_counts(run.joined, column);
    print_line(fmt::format("{} counts:", column));
    print_top_table(counts, {std::string(column), "count"});
    print_line("");
  }
  print_line("Outputs:");
  for (const auto& output : run.outputs) {
    print_line(fmt::format("{:36} = {}", output.key, python_path_string(output.path)));
  }
  print_line("");
  print_line("Correlations:");
  print_top_table(products::compute_joint_correlations(run.joined), {"description", "n", "pearson", "spearman"});
  return 0;
}

}  // namespace

Command add_score_command(CLI::App& app) {
  auto* score = app.add_subcommand("score", "Scoring layers over reduced CSVs (scripts/score_*.py)");
  score->require_subcommand(1);

  auto campaign = std::make_shared<CampaignArgs>();
  auto* campaign_command = score->add_subcommand("campaign", "Guiding score per case (scripts/score_campaign.py)");
  add_common(*campaign_command, campaign->common, 30);
  campaign_command->add_option("--case-type", campaign->case_type, "channel, uniform, vacuum or all")
      ->check(CLI::IsMember({"channel", "uniform", "vacuum", "all"}))
      ->capture_default_str();
  add_score_config(*campaign_command, campaign->config);

  auto triplets = std::make_shared<TripletArgs>();
  auto* triplets_command = score->add_subcommand("triplets", "Reference-aware triplet score (scripts/score_triplets.py)");
  add_common(*triplets_command, triplets->common, 50);
  add_score_config(*triplets_command, triplets->case_config);
  triplets_command->add_option("--reference-deadband", triplets->triplet_config.reference_deadband)
      ->capture_default_str();
  triplets_command->add_option("--reference-scale", triplets->triplet_config.reference_scale)->capture_default_str();

  auto beamlike = std::make_shared<BeamlikeArgs>();
  auto* beamlike_command =
      score->add_subcommand("beamlike-pairs", "Channel-vs-uniform particle comparison (scripts/score_beamlike_pairs.py)");
  add_common(*beamlike_command, beamlike->common, 50);
  beamlike_command->add_option("--particle-outdir-name", beamlike->particle_outdir_name)->capture_default_str();
  beamlike_command->add_option("--row-selection", beamlike->row_selection)
      ->check(CLI::IsMember({"single", "last", "max-beamlike"}))
      ->capture_default_str();
  beamlike_command->add_option("--reference-deadband-log", beamlike->config.reference_deadband_log)
      ->capture_default_str();
  beamlike_command->add_option("--reference-scale-log", beamlike->config.reference_scale_log)->capture_default_str();
  beamlike_command->add_option("--score-floor", beamlike->config.score_floor)->capture_default_str();

  auto joint = std::make_shared<JointArgs>();
  auto* joint_command =
      score->add_subcommand("joint", "Join guiding and beamlike scores (scripts/compare_guiding_beamlike_scores.py)");
  joint_command->add_option("--campaign-root", joint->campaign_root,
                            "Locates the newest analysis_outputs/triplet_scores_* and beamlike_pairs_*");
  joint_command->add_option("--triplet-scores-csv", joint->triplet_scores_csv);
  joint_command->add_option("--beamlike-pair-scores-csv", joint->beamlike_pair_scores_csv);
  joint_command->add_option("--outdir", joint->outdir);
  joint_command->add_option("--top", joint->top)->capture_default_str();
  joint_command->add_option("--join-how", joint->join_how)
      ->check(CLI::IsMember({"inner", "left", "right", "outer"}))
      ->capture_default_str();

  const Runner run = [campaign, campaign_command, triplets, triplets_command, beamlike, beamlike_command, joint,
                      joint_command] {
    if (campaign_command->parsed()) {
      return run_score_campaign(*campaign);
    }
    if (triplets_command->parsed()) {
      return run_score_triplets(*triplets);
    }
    if (beamlike_command->parsed()) {
      return run_score_beamlike_pairs(*beamlike);
    }
    if (joint_command->parsed()) {
      return run_joint(*joint);
    }
    return 1;
  };
  return {score, run};
}

}  // namespace guiding::cli

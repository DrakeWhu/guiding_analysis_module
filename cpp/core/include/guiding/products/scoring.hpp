#pragma once

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "guiding/campaign/discovery.hpp"
#include "guiding/table/frame.hpp"
#include "guiding/table/record.hpp"

// Port of cap_guiding/scoring.py and scripts/score_campaign.py / score_triplets.py.
namespace guiding::products {

struct ScoreConfig {
  double entry_window_mm = 1.0;
  double exit_before_mm = 1.0;
  double exit_after_mm = 2.0;
  double a0_target = 1.5;
  double a0_component_cap = 1.25;
  double waist_growth_sigma = 0.75;
  double waist_jitter_sigma = 0.25;
  double weight_a0_exit = 0.50;
  double weight_a0_retention = 0.20;
  double weight_waist_growth = 0.20;
  double weight_waist_stability = 0.10;
};

struct TripletScoreConfig {
  double reference_deadband = 5.0;
  double reference_scale = 15.0;
  double reference_deadband_log = std::log(1.05);
  double reference_scale_log = std::log(1.5);
  double reference_a0_weight = 0.80;
  double reference_waist_weight = 0.20;
};

// score_case_csv: guiding score of one guiding_metrics.csv (failures are rows
// with status "failed" and a failure_reason, never exceptions).
[[nodiscard]] table::Record score_case_csv(const std::filesystem::path& csv_path,
                                           const std::optional<std::string>& case_id = std::nullopt,
                                           const ScoreConfig& config = {});

// tanh of the deadbanded, combined log advantage; (factor, combined).
[[nodiscard]] std::pair<double, double> reference_factor_from_local_advantage(double a0_log_advantage,
                                                                              double waist_log_advantage,
                                                                              double deadband_log, double scale_log,
                                                                              double a0_weight, double waist_weight);

[[nodiscard]] table::Record score_triplet_csvs(const std::filesystem::path& channel_csv,
                                               const std::filesystem::path& uniform_csv,
                                               const std::filesystem::path& vacuum_csv,
                                               const std::optional<std::string>& channel_case_id,
                                               const std::optional<std::string>& uniform_case_id,
                                               const std::optional<std::string>& vacuum_case_id,
                                               const ScoreConfig& case_config = {},
                                               const TripletScoreConfig& triplet_config = {});

// A scored table and its top rows (status == "ok", sorted by a score column
// descending, with a leading 1-based "rank").
struct ScoreTables {
  table::Frame scores;
  table::Frame top;
  std::size_t ok = 0;
};

// Rows with status "ok" sorted by `column` (descending, NaN last), first `top`, with rank.
[[nodiscard]] table::Frame top_rows(const table::Frame& frame, const std::string& column, std::size_t top);

// score_campaign.py: case_type is "channel", "uniform", "vacuum" or "all".
[[nodiscard]] ScoreTables score_campaign_cases(const std::vector<campaign::CaseInfo>& cases,
                                               const std::filesystem::path& case_metrics_root,
                                               const std::string& case_type, const ScoreConfig& config,
                                               std::size_t top);

struct TripletScoreTables {
  ScoreTables tables;
  std::size_t skipped_incomplete = 0;
};

// score_triplets.py over the complete triplets.
[[nodiscard]] TripletScoreTables score_campaign_triplets(const std::vector<campaign::TripletInfo>& triplets,
                                                         const std::filesystem::path& case_metrics_root,
                                                         const ScoreConfig& case_config,
                                                         const TripletScoreConfig& triplet_config, std::size_t top);

// Python float(value) of a record cell; nullopt where Python raises.
[[nodiscard]] std::optional<double> py_float(const table::Cell& cell);

}  // namespace guiding::products

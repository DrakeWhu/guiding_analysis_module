#pragma once

#include <cmath>
#include <filesystem>
#include <optional>
#include <string>

#include "guiding/table/csv.hpp"
#include "guiding/table/record.hpp"

// Port of cap_guiding/singlecase_guiding.py (guiding_singlecase_score_v1).
namespace guiding::products {

inline constexpr const char* kSingleCaseScoreFilename = "guiding_singlecase_score.csv";

struct SingleCaseGuidingConfig {
  std::string schema_version = "guiding_singlecase_score_v1";
  std::string config_id = "guiding_singlecase_score_v1_waist_retention_20260701";
  double plateau_start_default_mm = 5.0;
  double entry_window_mm = 1.0;
  double exit_window_mm = 1.0;
  int min_valid_plateau_rows = 3;
  double sigma_a0_drop_log = std::log(1.30);
  double waist_growth_deadband = 1.10;
  double sigma_waist_growth_log = std::log(1.50);
  double sigma_waist_jitter_log = std::log(1.35);
  double weight_a0_retention = 0.35;
  double weight_a0_stability = 0.15;
  double weight_waist_growth = 0.45;
  double weight_waist_stability = 0.05;
};

struct PlateauOverride {
  std::optional<double> start_mm;
  std::optional<double> end_mm;
  std::optional<double> length_mm;
};

// score_singlecase_guiding_dataframe(df, case_id=..., csv_path=...)
[[nodiscard]] table::Record score_singlecase_guiding_table(const table::CsvTable& table, const std::string& case_id,
                                                           const std::optional<std::string>& csv_path,
                                                           const PlateauOverride& plateau = {},
                                                           const SingleCaseGuidingConfig& config = {});

// score_singlecase_guiding_csv(path, case_id=...): reads the CSV; case_id
// defaults to the parent directory name.
[[nodiscard]] table::Record score_singlecase_guiding_csv(const std::filesystem::path& csv_path,
                                                         const std::optional<std::string>& case_id,
                                                         const PlateauOverride& plateau = {},
                                                         const SingleCaseGuidingConfig& config = {});

// ensure_singlecase_guiding_score_csv: writes <csv dir>/guiding_singlecase_score.csv
// unless it exists and overwrite is false. Returns true when a file was written.
bool ensure_singlecase_guiding_score_csv(const std::filesystem::path& guiding_metrics_csv,
                                         const std::optional<std::string>& case_id, bool overwrite,
                                         std::filesystem::path* score_path_out = nullptr);

// str(pathlib.Path(text)) for POSIX paths: collapses repeated separators, drops
// "." components and a trailing separator.
[[nodiscard]] std::string python_path_string(const std::filesystem::path& path);

}  // namespace guiding::products

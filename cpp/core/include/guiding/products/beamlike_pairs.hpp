#pragma once

#include <cmath>
#include <cstddef>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include "guiding/campaign/discovery.hpp"
#include "guiding/table/record.hpp"

// Port of cap_guiding/beamlike_pairs.py and scripts/score_beamlike_pairs.py:
// channel-vs-uniform comparison of particle_summary.csv rows.
namespace guiding::products {

struct BeamlikePairConfig {
  double reference_deadband_log = std::log(1.05);
  double reference_scale_log = std::log(1.5);
  double score_floor = 1.0;
  double transverse_reference_deadband_log = std::log(1.05);
  double transverse_reference_scale_log = std::log(1.5);
  double transverse_score_floor = 1.0;
};

enum class RowSelection { Single, Last, MaxBeamlike };
[[nodiscard]] RowSelection parse_row_selection(std::string_view name);
[[nodiscard]] const char* row_selection_name(RowSelection selection) noexcept;

// csv.DictReader rows (all values str) of a particle_summary.csv.
[[nodiscard]] std::vector<table::Record> read_dict_rows(const std::filesystem::path& path);

// read_particle_summary_row; throws with the reference failure messages.
[[nodiscard]] table::Record read_particle_summary_row(const std::filesystem::path& csv_path, RowSelection selection);

[[nodiscard]] table::Record compare_beamlike_pair_rows(const table::Record& channel, const table::Record& uniform,
                                                       const std::string& channel_case_id,
                                                       const std::string& uniform_case_id,
                                                       const std::string& channel_csv, const std::string& uniform_csv,
                                                       RowSelection selection, const BeamlikePairConfig& config);

[[nodiscard]] table::Record compare_beamlike_pair_csvs(const std::filesystem::path& channel_csv,
                                                       const std::filesystem::path& uniform_csv,
                                                       const std::string& channel_case_id,
                                                       const std::string& uniform_case_id, RowSelection selection,
                                                       const BeamlikePairConfig& config);

// "positive" | "neutral" | "negative" | "failed"
[[nodiscard]] std::string classify_pair_row(const table::Record& row);

[[nodiscard]] const std::vector<std::string>& pair_output_columns();

// write_rows_csv: preferred columns first, then the remaining keys; LF rows.
[[nodiscard]] std::string format_pair_rows_csv(const std::vector<table::Record>& rows);

struct BeamlikePairTables {
  std::vector<table::Record> positive;  // by beamlike_gain_score, descending
  std::vector<table::Record> neutral;   // by plateau, diameter, focus, channel id
  std::vector<table::Record> negative;  // by beamlike_gain_score, ascending
  std::vector<table::Record> failed;    // by failure reason, channel id, uniform id
  std::size_t attempted = 0;
  std::size_t skipped_without_channel = 0;
  std::size_t skipped_without_uniform = 0;

  [[nodiscard]] std::vector<table::Record> all() const;
};

// score_beamlike_pairs.py for every triplet with a channel and a uniform case.
[[nodiscard]] BeamlikePairTables score_beamlike_pairs(const std::vector<campaign::TripletInfo>& triplets,
                                                      const std::filesystem::path& case_metrics_root,
                                                      const std::string& particle_outdir_name, RowSelection selection,
                                                      const BeamlikePairConfig& config);

// Copies of the first `top` rows with a trailing "rank" key.
[[nodiscard]] std::vector<table::Record> ranked(const std::vector<table::Record>& rows, std::size_t top);

// str(value) as the reference prints row values.
[[nodiscard]] std::string py_str(const table::Cell& cell);

}  // namespace guiding::products

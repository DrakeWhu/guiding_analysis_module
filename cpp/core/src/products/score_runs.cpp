#include "guiding/products/score_runs.hpp"

#include <ctime>
#include <stdexcept>

#include <fmt/format.h>

#include "guiding/campaign/discovery.hpp"
#include "guiding/products/particle_campaign.hpp"
#include "guiding/table/csv.hpp"
#include "guiding/table/py_format.hpp"

namespace guiding::products {
namespace {

namespace fs = std::filesystem;

std::string frame_csv(const table::Frame& frame) {
  return frame.names().empty() ? std::string("\n") : frame.to_pandas_csv();
}

}  // namespace

CampaignScoreRun run_score_campaign(const fs::path& campaign_root, const fs::path& case_metrics_root,
                                    const fs::path& outdir, const std::string& case_type, const ScoreConfig& config,
                                    std::size_t top) {
  fs::create_directories(outdir);
  const auto cases = campaign::discover_cases(campaign_root);
  CampaignScoreRun run;
  run.tables = score_campaign_cases(cases, case_metrics_root, case_type, config, top);
  run.cases_scored = run.tables.scores.row_count();
  run.files = {{"case_scores", outdir / "case_scores.csv"}, {"top_cases", outdir / "top_cases.csv"}};
  table::write_file_atomically(run.files[0].second, frame_csv(run.tables.scores));
  table::write_file_atomically(run.files[1].second, frame_csv(run.tables.top));
  return run;
}

TripletScoreRun run_score_triplets(const fs::path& campaign_root, const fs::path& case_metrics_root,
                                   const fs::path& outdir, const ScoreConfig& case_config,
                                   const TripletScoreConfig& triplet_config, std::size_t top) {
  fs::create_directories(outdir);
  const auto cases = campaign::discover_cases(campaign_root);
  const auto triplets = campaign::build_triplets(cases);
  TripletScoreRun run;
  run.result = score_campaign_triplets(triplets, case_metrics_root, case_config, triplet_config, top);
  run.triplets_total = triplets.size();
  run.triplets_scored = run.result.tables.scores.row_count();
  run.files = {{"triplet_scores", outdir / "triplet_scores.csv"}, {"top_triplets", outdir / "top_triplets.csv"}};
  table::write_file_atomically(run.files[0].second, frame_csv(run.result.tables.scores));
  table::write_file_atomically(run.files[1].second, frame_csv(run.result.tables.top));
  return run;
}

BeamlikePairRun run_score_beamlike_pairs(const fs::path& campaign_root, const fs::path& case_metrics_root,
                                         const std::string& particle_outdir_name, const fs::path& outdir,
                                         RowSelection selection, const BeamlikePairConfig& config, std::size_t top) {
  fs::create_directories(outdir);
  const auto cases = campaign::discover_cases(campaign_root);
  const auto triplets = campaign::build_triplets(cases);
  BeamlikePairRun run;
  run.cases_discovered = cases.size();
  run.tables = score_beamlike_pairs(triplets, case_metrics_root, particle_outdir_name, selection, config);
  run.top_rows = ranked(run.tables.positive, top);
  run.worst_rows = ranked(run.tables.negative, top);
  run.files = {{"pair_scores", outdir / "beamlike_pair_scores.csv"},
               {"positive_pairs", outdir / "positive_beamlike_pairs.csv"},
               {"neutral_pairs", outdir / "neutral_beamlike_pairs.csv"},
               {"negative_pairs", outdir / "negative_beamlike_pairs.csv"},
               {"failed_pairs", outdir / "failed_beamlike_pairs.csv"},
               {"top_positive_pairs", outdir / "top_beamlike_pairs.csv"},
               {"worst_negative_pairs", outdir / "worst_negative_beamlike_pairs.csv"}};
  const auto all = run.tables.all();
  table::write_file_atomically(run.files[0].second, format_pair_rows_csv(all));
  table::write_file_atomically(run.files[1].second, format_pair_rows_csv(run.tables.positive));
  table::write_file_atomically(run.files[2].second, format_pair_rows_csv(run.tables.neutral));
  table::write_file_atomically(run.files[3].second, format_pair_rows_csv(run.tables.negative));
  table::write_file_atomically(run.files[4].second, format_pair_rows_csv(run.tables.failed));
  table::write_file_atomically(run.files[5].second, format_pair_rows_csv(run.top_rows));
  table::write_file_atomically(run.files[6].second, format_pair_rows_csv(run.worst_rows));
  return run;
}

JointScoreRun run_joint_scores(const fs::path& triplet_scores_csv, const fs::path& beamlike_pair_scores_csv,
                               const fs::path& outdir, JoinHow how, std::size_t top) {
  JointScoreRun run;
  run.joined = load_and_join_guiding_beamlike(triplet_scores_csv, beamlike_pair_scores_csv, how);
  run.outputs = write_joint_outputs(outdir, run.joined, top);
  return run;
}

fs::path latest_analysis_file(const fs::path& campaign_root, const std::string& dirname_glob,
                              const std::string& filename) {
  const fs::path analysis_root = campaign_root / "analysis_outputs";
  fs::path best;
  fs::file_time_type best_time{};
  std::error_code error;
  if (fs::is_directory(analysis_root, error)) {
    for (const auto& directory : glob_directories(analysis_root, dirname_glob)) {
      const fs::path candidate = directory / filename;
      if (fs::is_regular_file(candidate, error)) {
        const auto time = fs::last_write_time(candidate, error);
        if (best.empty() || time > best_time) {
          best = candidate;
          best_time = time;
        }
      }
    }
  }
  if (best.empty()) {
    throw std::runtime_error(fmt::format("No {} found under {}/{}", filename,
                                         table::python_path_string(analysis_root), dirname_glob));
  }
  return best;
}

std::string timestamped(const std::string& prefix) {
  const auto now = std::time(nullptr);
  std::tm local{};
  localtime_r(&now, &local);
  return fmt::format("{}_{:04}{:02}{:02}_{:02}{:02}{:02}", prefix, local.tm_year + 1900, local.tm_mon + 1,
                     local.tm_mday, local.tm_hour, local.tm_min, local.tm_sec);
}

}  // namespace guiding::products

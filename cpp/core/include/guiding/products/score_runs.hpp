#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#include "guiding/products/beamlike_pairs.hpp"
#include "guiding/products/joint_scores.hpp"
#include "guiding/products/scoring.hpp"

// The scoring scripts as functions that write their output files; shared by
// guiding_cli and the parity tests.
namespace guiding::products {

using OutputFiles = std::vector<std::pair<std::string, std::filesystem::path>>;

struct CampaignScoreRun {
  ScoreTables tables;
  std::size_t cases_scored = 0;
  OutputFiles files;  // case_scores, top_cases
};

// scripts/score_campaign.py
CampaignScoreRun run_score_campaign(const std::filesystem::path& campaign_root,
                                    const std::filesystem::path& case_metrics_root, const std::filesystem::path& outdir,
                                    const std::string& case_type, const ScoreConfig& config, std::size_t top);

struct TripletScoreRun {
  TripletScoreTables result;
  std::size_t triplets_total = 0;
  std::size_t triplets_scored = 0;
  OutputFiles files;  // triplet_scores, top_triplets
};

// scripts/score_triplets.py
TripletScoreRun run_score_triplets(const std::filesystem::path& campaign_root,
                                   const std::filesystem::path& case_metrics_root, const std::filesystem::path& outdir,
                                   const ScoreConfig& case_config, const TripletScoreConfig& triplet_config,
                                   std::size_t top);

struct BeamlikePairRun {
  BeamlikePairTables tables;
  std::vector<table::Record> top_rows;
  std::vector<table::Record> worst_rows;
  std::size_t cases_discovered = 0;
  OutputFiles files;  // pair_scores ... worst_negative_pairs
};

// scripts/score_beamlike_pairs.py
BeamlikePairRun run_score_beamlike_pairs(const std::filesystem::path& campaign_root,
                                         const std::filesystem::path& case_metrics_root,
                                         const std::string& particle_outdir_name, const std::filesystem::path& outdir,
                                         RowSelection selection, const BeamlikePairConfig& config, std::size_t top);

struct JointScoreRun {
  table::Frame joined;
  std::vector<JointOutput> outputs;
};

// scripts/compare_guiding_beamlike_scores.py (explicit CSV paths)
JointScoreRun run_joint_scores(const std::filesystem::path& triplet_scores_csv,
                               const std::filesystem::path& beamlike_pair_scores_csv,
                               const std::filesystem::path& outdir, JoinHow how, std::size_t top);

// Newest <campaign_root>/analysis_outputs/<dirname_glob>/<filename> by mtime.
[[nodiscard]] std::filesystem::path latest_analysis_file(const std::filesystem::path& campaign_root,
                                                         const std::string& dirname_glob, const std::string& filename);

// "<prefix>_YYYYmmdd_HHMMSS" with the local time.
[[nodiscard]] std::string timestamped(const std::string& prefix);

}  // namespace guiding::products

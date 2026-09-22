#pragma once

#include <cstddef>
#include <filesystem>
#include <string>
#include <vector>

#include "guiding/products/particle_reduction.hpp"

// Port of scripts/analyze_particle_campaign.py, running the cases in-process.
namespace guiding::products {

struct ParticleCampaignOptions {
  std::filesystem::path campaign_root;
  std::string particle_diag_name = "auto";
  // --species exactly as given; the reference resolves the diagnostic
  // directory with this raw text.
  std::string species_text = "electrons";
  std::string case_glob = "0*_from_*";
  std::string outdir_name = "particle_analysis";
  // Exact exit selection from CASE/resolved_parameters.json (C++ extension).
  bool use_resolved_parameters = false;
  // Per-case settings; diag, outdir and resolved_parameters are set per case.
  ParticleCaseOptions case_options;
  unsigned threads = 1;  // cases analysed concurrently
};

struct ParticleCampaignResult {
  std::size_t ok = 0;
  std::size_t failed = 0;
};

// pathlib.Path(root).glob(pattern) restricted to directories, sorted.
[[nodiscard]] std::vector<std::filesystem::path> glob_directories(const std::filesystem::path& root,
                                                                  const std::string& pattern);

// Logs are buffered per case and emitted in case order.
[[nodiscard]] ParticleCampaignResult run_particle_campaign(const ParticleCampaignOptions& options,
                                                           const LineSink& log);

}  // namespace guiding::products

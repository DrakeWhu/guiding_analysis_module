#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <string>

#include <nlohmann/json.hpp>

#include "guiding/products/score_runs.hpp"
#include "guiding/table/csv.hpp"
#include "support/table_compare.hpp"

namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;
using namespace guiding;

fs::path data_dir() { return fs::path(GUIDING_TEST_DATA_DIR); }

std::string read_bytes(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary);
  REQUIRE(stream.good());
  return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

// The reference ran from cpp/tests/data, so the paths it recorded are relative to it.
struct DataDirectory {
  fs::path previous = fs::current_path();
  DataDirectory() { fs::current_path(data_dir()); }
  ~DataDirectory() { fs::current_path(previous); }
};

double number(const json& options, const char* flag, double fallback) {
  return options.contains(flag) ? options.at(flag).get<double>() : fallback;
}

products::ScoreConfig score_config(const json& options) {
  products::ScoreConfig config;
  config.entry_window_mm = number(options, "entry-window-mm", config.entry_window_mm);
  config.exit_before_mm = number(options, "exit-before-mm", config.exit_before_mm);
  config.exit_after_mm = number(options, "exit-after-mm", config.exit_after_mm);
  config.a0_target = number(options, "a0-target", config.a0_target);
  config.a0_component_cap = number(options, "a0-component-cap", config.a0_component_cap);
  config.waist_growth_sigma = number(options, "waist-growth-sigma", config.waist_growth_sigma);
  config.waist_jitter_sigma = number(options, "waist-jitter-sigma", config.waist_jitter_sigma);
  return config;
}

void check_outputs(const fs::path& actual_dir, const fs::path& expected_dir) {
  // numpy's vectorized tanh differs from libm by up to 1 ULP, and the
  // correlations go through BLAS dot products; those columns use 1e-12.
  const std::set<std::string> tolerant_columns{"pearson", "spearman", "reference_factor", "final_score"};
  for (const auto& entry : fs::directory_iterator(expected_dir)) {
    const std::string name = entry.path().filename().string();
    if (name == "stdout.txt") {
      continue;
    }
    DYNAMIC_SECTION(name) {
      const fs::path actual = actual_dir / name;
      REQUIRE(fs::exists(actual));
      const std::string actual_bytes = read_bytes(actual);
      const std::string expected_bytes = read_bytes(entry.path());
      if (actual_bytes == expected_bytes) {
        SUCCEED();
        continue;
      }
      const auto actual_table = table::read_csv_file(actual);
      const auto expected_table = table::read_csv_file(entry.path());
      const auto exact = testing::compare_tables(actual_table, expected_table, 0.0, tolerant_columns);
      INFO(exact.describe());
      CHECK(exact.ok());
      const auto close = testing::compare_tables(actual_table, expected_table, 1e-12);
      INFO(close.describe());
      CHECK(close.ok());
      const bool tolerant_table = std::any_of(tolerant_columns.begin(), tolerant_columns.end(),
                                              [&](const std::string& c) { return expected_table.has_column(c); });
      CHECK(tolerant_table);  // every other table must be byte-identical
    }
  }
}

}  // namespace

TEST_CASE("scoring layers match the Python reference", "[parity][scoring]") {
  const json runs = json::parse(read_bytes(data_dir() / "golden" / "scoring" / "runs.json"));
  REQUIRE(runs.size() >= 4);
  const fs::path scratch = fs::temp_directory_path() / "guiding_tests_scoring";
  fs::remove_all(scratch);
  DataDirectory in_data_dir;

  const fs::path campaign_root = "synthetic/scoring/campaign";
  const fs::path metrics_root = "synthetic/scoring/case_metrics";
  for (const auto& run : runs) {
    const std::string name = run.at("name").get<std::string>();
    const std::string script = run.at("script").get<std::string>();
    const json& options = run.at("options");
    const fs::path outdir = scratch / name;
    const auto top = static_cast<std::size_t>(options.value("top", script == "score_campaign" ? 30 : 50));
    DYNAMIC_SECTION(name) {
      if (script == "score_campaign") {
        (void)products::run_score_campaign(campaign_root, metrics_root, outdir, options.value("case-type", "channel"),
                                           score_config(options), top);
      } else if (script == "score_triplets") {
        products::TripletScoreConfig triplet_config;
        triplet_config.reference_deadband = number(options, "reference-deadband", triplet_config.reference_deadband);
        triplet_config.reference_scale = number(options, "reference-scale", triplet_config.reference_scale);
        (void)products::run_score_triplets(campaign_root, metrics_root, outdir, score_config(options), triplet_config,
                                           top);
      } else if (script == "score_beamlike_pairs") {
        products::BeamlikePairConfig config;
        config.reference_deadband_log = number(options, "reference-deadband-log", config.reference_deadband_log);
        config.reference_scale_log = number(options, "reference-scale-log", config.reference_scale_log);
        config.score_floor = number(options, "score-floor", config.score_floor);
        (void)products::run_score_beamlike_pairs(
            campaign_root, metrics_root, options.value("particle-outdir-name", "particle_analysis"), outdir,
            products::parse_row_selection(options.value("row-selection", "single")), config, top);
      } else if (script == "compare_guiding_beamlike_scores") {
        (void)products::run_joint_scores(options.at("triplet-scores-csv").get<std::string>(),
                                         options.at("beamlike-pair-scores-csv").get<std::string>(), outdir,
                                         products::parse_join_how(options.value("join-how", "inner")), top);
      } else {
        FAIL("unknown script in runs.json: " << script);
      }
      check_outputs(outdir, fs::path("golden") / "scoring" / name);
    }
  }
  fs::remove_all(scratch);
}

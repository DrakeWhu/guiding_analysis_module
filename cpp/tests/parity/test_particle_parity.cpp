#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "guiding/products/particle_campaign.hpp"
#include "guiding/products/particle_reduction.hpp"
#include "guiding/table/csv.hpp"
#include "guiding/table/py_format.hpp"
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

std::vector<std::string> read_lines(const fs::path& path) {
  std::vector<std::string> lines;
  std::ifstream stream(path);
  for (std::string line; std::getline(stream, line);) {
    lines.push_back(line);
  }
  return lines;
}

const std::vector<std::string> kProducts{"particle_summary.csv", "particle_acceptance_curves.csv",
                                         "particle_soft50_curves.csv"};
// Paths are spelled differently by the two runs.
const std::set<std::string> kPathColumns{"guiding_metrics_csv", "resolved_parameters_path"};

// The same flags generate_golden.py passes to analyze_particle_case.py.
products::ParticleCaseOptions options_from_run(const json& run) {
  products::ParticleCaseOptions options;
  options.diag = data_dir() / run.at("diag").get<std::string>();
  for (const auto& [flag, value] : run.at("options").items()) {
    const auto text = [&] { return value.get<std::string>(); };
    if (flag == "species") {
      options.species = products::parse_species_list(text());
    } else if (flag == "which") {
      options.which = products::parse_particle_which(text());
    } else if (flag == "stride") {
      options.stride = value.get<int>();
    } else if (flag == "hot-energy-mev") {
      options.hot_energy_mev = value.get<double>();
    } else if (flag == "longitudinal") {
      options.longitudinal = physics::parse_longitudinal(text());
    } else if (flag == "no-forward-cut") {
      options.forward_only = false;
    } else if (flag == "exit-window-mm") {
      options.exit_window_mm = value.get<double>();
    } else if (flag == "exit-kind") {
      options.exit_kind = products::parse_exit_kind(text());
    } else if (flag == "target-propagation-mm") {
      options.target_propagation_mm = value.get<double>();
    } else if (flag == "guiding-metrics") {
      options.guiding_metrics = data_dir() / text();
    } else if (flag == "resolved-parameters") {
      options.resolved_parameters = data_dir() / text();
    } else if (flag == "maximum-target-iteration-delta") {
      options.maximum_target_iteration_delta = value.get<std::int64_t>();
    } else if (flag == "downramp-mm") {
      options.downramp_mm = value.get<double>();
    } else if (flag == "acceptance-theta-cuts-mrad") {
      options.acceptance_theta_cuts_mrad = products::parse_float_list(text());
    } else if (flag == "acceptance-energy-cuts-mev") {
      options.acceptance_energy_cuts_mev = products::parse_float_list(text());
    } else if (flag == "soft50-energy-low-mev") {
      options.soft50.energy_low_mev = value.get<double>();
    } else if (flag == "soft50-energy-target-mev") {
      options.soft50.energy_target_mev = value.get<double>();
    } else if (flag == "soft50-reliability-floor") {
      options.soft50.reliability_floor = value.get<double>();
    } else if (flag == "soft50-effective-count-reference") {
      options.soft50.effective_count_reference = value.get<double>();
    } else if (flag == "soft50-curve-energy-low-mev") {
      options.soft50_curve_energy_low_mev = products::parse_float_list(text());
    } else {
      FAIL("runs.json uses an option the parity test does not map: " << flag);
    }
  }
  return options;
}

// "[SERIES] ..." and the "[SELECTION]" block without path-valued entries.
std::vector<std::string> selection_block(const std::vector<std::string>& lines) {
  std::vector<std::string> block;
  bool inside = false;
  for (const auto& line : lines) {
    if (line.starts_with("[SERIES]")) {
      block.push_back(line);
    } else if (line == "[SELECTION]") {
      inside = true;
      block.push_back(line);
    } else if (inside && line.starts_with("  ")) {
      const bool path_valued = line.starts_with("  guiding_metrics_csv = ") ||
                               line.starts_with("  resolved_parameters_path = ");
      if (!path_valued) {
        block.push_back(line);
      }
    } else if (inside) {
      break;
    }
  }
  return block;
}

void check_products(const fs::path& actual_dir, const fs::path& expected_dir) {
  for (const auto& name : kProducts) {
    DYNAMIC_SECTION(name) {
      const auto actual = table::read_csv_file(actual_dir / name);
      const auto expected = table::read_csv_file(expected_dir / name);
      const auto comparison = testing::compare_tables(actual, expected, 0.0, kPathColumns);
      INFO(comparison.describe());
      CHECK(comparison.ok());
      const bool has_paths = std::any_of(expected.columns.begin(), expected.columns.end(),
                                         [](const std::string& column) { return kPathColumns.contains(column); });
      if (!has_paths) {
        // Same CRLF records, quoting and float spelling as csv.DictWriter.
        CHECK(read_bytes(actual_dir / name) == read_bytes(expected_dir / name));
      }
    }
  }
}

}  // namespace

TEST_CASE("particle case products match the Python reference", "[parity][particles]") {
  const fs::path golden = data_dir() / "golden" / "particles";
  const json runs = json::parse(read_bytes(golden / "runs.json"));
  REQUIRE(runs.size() >= 5);

  for (const auto& run : runs) {
    const std::string name = run.at("name").get<std::string>();
    DYNAMIC_SECTION(name) {
      auto options = options_from_run(run);
      options.outdir = fs::temp_directory_path() / "guiding_tests_particles" / name;
      fs::remove_all(options.outdir);
      options.overwrite = true;
      options.threads = 3;

      std::mutex mutex;
      std::vector<std::string> lines;
      const auto log = [&](const std::string& line) {
        std::lock_guard lock(mutex);
        lines.push_back(line);
      };
      REQUIRE(products::run_particle_case(options, log) == products::ParticleCaseOutcome::Written);

      CHECK(selection_block(lines) == selection_block(read_lines(golden / name / "stdout.txt")));
      check_products(options.outdir, golden / name);
      fs::remove_all(options.outdir);
    }
  }
}

TEST_CASE("particle campaign matches the Python reference", "[parity][particles][campaign]") {
  const fs::path scratch = fs::temp_directory_path() / "guiding_tests_particle_campaign";
  fs::remove_all(scratch);
  const fs::path root = scratch / "campaign";
  fs::create_directories(root);
  fs::copy(data_dir() / "synthetic" / "campaign", root, fs::copy_options::recursive);

  products::ParticleCampaignOptions options;
  options.campaign_root = root;
  options.case_glob = "0*";
  options.threads = 3;
  std::vector<std::string> lines;
  std::mutex mutex;
  const auto log = [&](const std::string& line) {
    std::lock_guard lock(mutex);
    lines.push_back(line);
  };
  const auto result = products::run_particle_campaign(options, log);
  CHECK(result.ok == 2);
  CHECK(result.failed == 1);

  const std::string shown_root = table::python_path_string(root);
  std::vector<std::string> normalized;
  for (auto line : lines) {
    for (auto at = line.find(shown_root); at != std::string::npos; at = line.find(shown_root, at)) {
      line.replace(at, shown_root.size(), "<CAMPAIGN_ROOT>");
    }
    normalized.push_back(line);
  }
  const fs::path golden = data_dir() / "golden" / "particles" / "campaign";
  CHECK(normalized == read_lines(golden / "stdout.txt"));

  for (const auto& entry : fs::directory_iterator(golden)) {
    if (entry.is_directory()) {
      DYNAMIC_SECTION(entry.path().filename().string()) {
        check_products(root / entry.path().filename() / "particle_analysis", entry.path());
      }
    }
  }
  fs::remove_all(scratch);
}

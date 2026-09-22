#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "guiding/campaign/discovery.hpp"
#include "guiding/products/case_reduction.hpp"
#include "guiding/table/csv.hpp"
#include "support/table_compare.hpp"

// Opt-in check against real WarpX output: point GUIDING_REAL_DATA_DIR at a
// campaign (or a single case) that already has guiding_metrics.csv files
// written by the Python reference, and this recomputes them from the HDF5
// dumps and compares.
//
//   GUIDING_REAL_DATA_DIR=~/guiding_real_subset ctest --preset release
//   GUIDING_REAL_DATA_DIR=~/guiding_real_subset build/release/cpp/tests/guiding_tests "[real]"
namespace {

namespace fs = std::filesystem;
using namespace guiding;

std::string read_bytes(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

// Case directories that hold both a diagnostic and a reference CSV.
std::vector<fs::path> real_cases(const fs::path& root) {
  std::vector<fs::path> cases;
  const auto usable = [](const fs::path& directory) {
    std::error_code error;
    return fs::is_regular_file(directory / "guiding_metrics.csv", error) &&
           fs::is_directory(campaign::resolve_field_diag_dir(directory, false), error);
  };
  if (usable(root)) {
    cases.push_back(root);
    return cases;
  }
  std::error_code error;
  for (auto it = fs::directory_iterator(root, error); !error && it != fs::directory_iterator(); it.increment(error)) {
    if (it->is_directory(error) && usable(it->path())) {
      cases.push_back(it->path());
    }
  }
  std::sort(cases.begin(), cases.end());
  return cases;
}

}  // namespace

TEST_CASE("real WarpX dumps reproduce their guiding_metrics.csv", "[real]") {
  const char* root = std::getenv("GUIDING_REAL_DATA_DIR");
  if (root == nullptr || *root == '\0') {
    SKIP("set GUIDING_REAL_DATA_DIR to a campaign or case directory with reference CSVs");
  }
  const auto cases = real_cases(root);
  INFO("looked under " << root);
  REQUIRE_FALSE(cases.empty());

  for (const auto& case_dir : cases) {
    DYNAMIC_SECTION(case_dir.filename().string()) {
      const fs::path reference_csv = case_dir / "guiding_metrics.csv";
      const fs::path diag = campaign::resolve_field_diag_dir(case_dir);
      products::CaseReductionOptions options;
      const auto rows = products::compute_case_rows(diag, options);
      const std::string actual = products::format_guiding_metrics_csv(rows);
      const std::string expected = read_bytes(reference_csv);
      if (actual == expected) {
        SUCCEED();
        continue;
      }
      // Not byte-identical: report which columns and rows differ.
      const auto comparison = testing::compare_tables(table::parse_csv(actual), table::parse_csv(expected), 0.0);
      INFO(comparison.describe());
      CHECK(comparison.ok());
    }
  }
}

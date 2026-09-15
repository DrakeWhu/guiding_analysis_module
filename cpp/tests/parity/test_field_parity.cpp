#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "guiding/products/case_reduction.hpp"
#include "guiding/products/singlecase_score.hpp"
#include "guiding/table/csv.hpp"
#include "guiding/table/record.hpp"
#include "support/table_compare.hpp"

namespace {

namespace fs = std::filesystem;
namespace products = guiding::products;

fs::path data_dir() { return fs::path(GUIDING_TEST_DATA_DIR); }

std::vector<fs::path> synthetic_cases() {
  std::vector<fs::path> cases;
  for (const auto& entry : fs::directory_iterator(data_dir() / "synthetic" / "campaign")) {
    if (fs::is_directory(entry.path() / "diags" / "diag1")) {
      cases.push_back(entry.path());
    }
  }
  std::sort(cases.begin(), cases.end());
  return cases;
}

std::string read_bytes(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary);
  REQUIRE(stream.good());
  return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::vector<products::GuidingMetricsRow> reduce(const fs::path& case_dir, unsigned threads, bool raw_reads) {
  products::CaseReductionOptions options;
  options.threads = threads;
  options.raw_reads = raw_reads;
  return products::compute_case_rows(case_dir / "diags" / "diag1", options);
}

}  // namespace

TEST_CASE("guiding_metrics.csv is byte-identical to the Python reference", "[parity][field]") {
  const auto cases = synthetic_cases();
  REQUIRE(cases.size() >= 3);
  for (const auto& case_dir : cases) {
    DYNAMIC_SECTION(case_dir.filename().string()) {
      const auto actual_text = products::format_guiding_metrics_csv(reduce(case_dir, 3, true));
      const auto golden = data_dir() / "golden" / "cases" / case_dir.filename() / "guiding_metrics.csv";
      const auto comparison = guiding::testing::compare_tables(guiding::table::parse_csv(actual_text),
                                                               guiding::table::read_csv_file(golden), 1e-9);
      INFO(comparison.describe());
      CHECK(comparison.ok());
      CHECK(actual_text == read_bytes(golden));
    }
  }
}

TEST_CASE("pread fast path, HDF5 reads and thread count give identical output", "[field][io]") {
  for (const auto& case_dir : synthetic_cases()) {
    DYNAMIC_SECTION(case_dir.filename().string()) {
      const auto reference = products::format_guiding_metrics_csv(reduce(case_dir, 1, false));
      CHECK(products::format_guiding_metrics_csv(reduce(case_dir, 1, true)) == reference);
      CHECK(products::format_guiding_metrics_csv(reduce(case_dir, 4, true)) == reference);
    }
  }
}

TEST_CASE("guiding_singlecase_score.csv matches the Python reference", "[parity][singlecase]") {
  const fs::path scratch = fs::temp_directory_path() / "guiding_tests_singlecase";
  fs::remove_all(scratch);
  for (const auto& case_dir : synthetic_cases()) {
    DYNAMIC_SECTION(case_dir.filename().string()) {
      // Same directory name as the reference so the plateau window is inferred identically.
      const fs::path outdir = scratch / case_dir.filename();
      const fs::path csv = outdir / "guiding_metrics.csv";
      products::write_guiding_metrics_csv(reduce(case_dir, 2, true), csv);
      REQUIRE(products::ensure_singlecase_guiding_score_csv(csv, case_dir.filename().string(), true));

      const auto golden_dir = data_dir() / "golden" / "cases" / case_dir.filename();
      const auto comparison = guiding::testing::compare_tables(
          guiding::table::read_csv_file(outdir / products::kSingleCaseScoreFilename),
          guiding::table::read_csv_file(golden_dir / products::kSingleCaseScoreFilename), 1e-12, {"csv_path"});
      INFO(comparison.describe());
      CHECK(comparison.ok());
    }
  }
  fs::remove_all(scratch);
}

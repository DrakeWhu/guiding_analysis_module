#include <catch2/catch_message.hpp>
#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <iterator>
#include <set>
#include <string>

#include "guiding/campaign/discovery.hpp"
#include "guiding/products/triplet_tables.hpp"
#include "guiding/table/csv.hpp"
#include "support/table_compare.hpp"

namespace {

namespace fs = std::filesystem;

fs::path data_dir() { return fs::path(GUIDING_TEST_DATA_DIR); }

std::string read_bytes(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary);
  REQUIRE(stream.good());
  return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

const std::string kChannel = "000_f20_chan_n4e18cm3_L2mm_d150um_foc0um_rz";
const std::string kUniform = "001_f20_uni_n4e18cm3_L2mm_refd150um_foc0um_rz";
const std::string kVacuum = "002_f20_vac_L2mm_refd150um_foc0um_rz";

}  // namespace

TEST_CASE("campaign report CSVs match the Python reference", "[parity][campaign]") {
  const fs::path root = data_dir() / "synthetic" / "campaign";
  const auto cases = guiding::campaign::discover_cases(root);
  REQUIRE(cases.size() == 3);
  const auto triplets = guiding::campaign::build_triplets(cases);
  REQUIRE(triplets.size() == 1);
  CHECK(triplets.front().complete());
  CHECK(triplets.front().label() == "f20_n4e18cm3_L2mm_d150um_foc0um");

  const fs::path out = fs::temp_directory_path() / "guiding_tests_campaign_report";
  fs::remove_all(out);
  guiding::campaign::write_campaign_report(cases, triplets, out, 2, 0.0, guiding::campaign::unix_time_now());

  const fs::path golden = data_dir() / "golden" / "campaign";
  // Paths depend on where the campaign root was given; ages on the clock.
  const std::set<std::string> ignored{"case_dir",
                                      "diag_dir",
                                      "newest_h5_age_min",
                                      "channel_newest_h5_age_min",
                                      "uniform_newest_h5_age_min",
                                      "vacuum_newest_h5_age_min"};
  for (const char* name : {"campaign_cases.csv", "campaign_triplets.csv", "campaign_insufficient_h5.csv",
                           "campaign_unstable_h5.csv"}) {
    DYNAMIC_SECTION(name) {
      const auto actual = guiding::table::read_csv_file(out / name);
      const auto expected = guiding::table::read_csv_file(golden / name);
      const auto comparison = guiding::testing::compare_tables(actual, expected, 0.0, ignored);
      INFO(comparison.describe());
      CHECK(comparison.ok());
      for (const char* column : {"case_dir", "diag_dir"}) {
        if (!expected.has_column(column)) {
          continue;
        }
        const auto actual_paths = actual.string_column(column);
        const auto expected_paths = expected.string_column(column);
        REQUIRE(actual_paths.size() == expected_paths.size());
        for (std::size_t i = 0; i < actual_paths.size(); ++i) {
          INFO(actual_paths[i] << " should end with " << expected_paths[i]);
          CHECK(actual_paths[i].ends_with(expected_paths[i]));
        }
      }
    }
  }
  fs::remove_all(out);
}

TEST_CASE("triplet tables match the Python reference", "[parity][triplet]") {
  const fs::path metrics = data_dir() / "golden" / "campaign" / "case_metrics";
  const auto tables = guiding::products::build_triplet_tables(
      metrics / kChannel / "guiding_metrics.csv", metrics / kUniform / "guiding_metrics.csv",
      metrics / kVacuum / "guiding_metrics.csv", "f20_n4e18cm3_L2mm_d150um_foc0um");

  const fs::path golden = metrics / kChannel;
  CHECK(tables.wide.to_pandas_csv() == read_bytes(golden / "guiding_triplet_wide.csv"));
  CHECK(tables.late_summary.to_pandas_csv() == read_bytes(golden / "guiding_triplet_late_summary.csv"));
  CHECK(tables.late_ratios.to_pandas_csv() == read_bytes(golden / "guiding_triplet_late_ratios.csv"));

  // The long table carries the input paths in source_csv.
  const auto comparison = guiding::testing::compare_tables(
      guiding::table::parse_csv(tables.long_table.to_pandas_csv()),
      guiding::table::read_csv_file(golden / "guiding_triplet_long.csv"), 0.0, {"source_csv"});
  INFO(comparison.describe());
  CHECK(comparison.ok());
}

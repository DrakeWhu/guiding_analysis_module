#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>

#include "data_store.hpp"

namespace {

namespace fs = std::filesystem;
using namespace guiding;

fs::path data_dir() { return fs::path(GUIDING_TEST_DATA_DIR); }

const std::string kChannel = "000_f20_chan_n4e18cm3_L2mm_d150um_foc0um_rz";
const std::string kVacuum = "002_f20_vac_L2mm_refd150um_foc0um_rz";

// Polls until the store is idle (background jobs publish asynchronously).
void wait_idle(const gui::DataStore& store) {
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  while (store.busy() && std::chrono::steady_clock::now() < deadline) {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  REQUIRE_FALSE(store.busy());
}

std::string read_bytes(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary);
  return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

}  // namespace

TEST_CASE("GUI data store scans, loads and caches campaign products", "[gui]") {
  gui::DataStore store(2);
  gui::CampaignSettings settings;
  settings.root = data_dir() / "synthetic" / "campaign";
  settings.case_metrics_root = data_dir() / "golden" / "campaign" / "case_metrics";
  store.request_scan(settings);
  wait_idle(store);

  const auto snapshot = store.campaign();
  REQUIRE(snapshot != nullptr);
  CHECK(snapshot->error.empty());
  REQUIRE(snapshot->cases.size() == 3);
  REQUIRE(snapshot->triplets.size() == 1);
  for (const auto& record : snapshot->cases) {
    CHECK(record.reduced_ready);
    CHECK(record.raw_ready());
    CHECK(record.singlecase_score.has_value());
  }

  const gui::CaseRecord* channel = snapshot->find_case(kChannel);
  REQUIRE(channel != nullptr);
  CHECK(store.case_metrics(*channel) == nullptr);  // scheduled, not loaded yet
  wait_idle(store);
  const auto metrics = store.case_metrics(*channel);
  REQUIRE(metrics != nullptr);
  CHECK(metrics->error.empty());
  CHECK(metrics->rows == 4);
  CHECK(metrics->ref_iteration == 0);
  REQUIRE(metrics->plateau_mm.has_value());
  CHECK(metrics->plateau_mm->first == 5.0);
  CHECK(metrics->plateau_mm->second == 7.0);
  CHECK(metrics->a0_norm.front() == 1.0);
  CHECK_FALSE(metrics->singlecase.rows.empty());
  CHECK(store.case_metrics(*channel) == metrics);  // cached while the CSV is unchanged

  const auto& triplet_info = snapshot->triplets.front();
  CHECK(store.triplet(*snapshot, triplet_info) == nullptr);
  wait_idle(store);
  const auto triplet = store.triplet(*snapshot, triplet_info);
  REQUIRE(triplet != nullptr);
  CHECK(triplet->error.empty());
  CHECK(triplet->column("propagation_mm").size() == 4);
  CHECK(triplet->plateau_mm.has_value());
  CHECK(triplet->late_window_mm.has_value());
}

TEST_CASE("GUI data store reduces a case and picks it up on rescan", "[gui]") {
  const fs::path metrics_root = fs::temp_directory_path() / "guiding_tests_gui_reduce";
  fs::remove_all(metrics_root);

  gui::DataStore store(2);
  gui::CampaignSettings settings;
  settings.root = data_dir() / "synthetic" / "campaign";
  settings.case_metrics_root = metrics_root;
  store.request_scan(settings);
  wait_idle(store);
  auto snapshot = store.campaign();
  REQUIRE(snapshot != nullptr);
  const gui::CaseRecord* vacuum = snapshot->find_case(kVacuum);
  REQUIRE(vacuum != nullptr);
  CHECK_FALSE(vacuum->reduced_ready);

  store.reduce_cases({vacuum->info}, settings, false);
  wait_idle(store);
  snapshot = store.campaign();
  vacuum = snapshot->find_case(kVacuum);
  REQUIRE(vacuum != nullptr);
  CHECK(vacuum->reduced_ready);
  CHECK(vacuum->singlecase_score.has_value());
  CHECK(read_bytes(metrics_root / kVacuum / "guiding_metrics.csv") ==
        read_bytes(data_dir() / "golden" / "cases" / kVacuum / "guiding_metrics.csv"));
  fs::remove_all(metrics_root);
}

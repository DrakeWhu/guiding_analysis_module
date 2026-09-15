#include <catch2/catch_test_macros.hpp>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "guiding/campaign/discovery.hpp"

namespace {

namespace fs = std::filesystem;
using namespace guiding::campaign;

struct TempDir {
  fs::path path;
  explicit TempDir(const std::string& name) : path(fs::temp_directory_path() / name) {
    fs::remove_all(path);
    fs::create_directories(path);
  }
  ~TempDir() {
    std::error_code error;
    fs::remove_all(path, error);
  }
};

void touch(const fs::path& path) {
  fs::create_directories(path.parent_path());
  std::ofstream(path) << "x";
}

CaseInfo make_case(const std::string& case_id) {
  CaseInfo info;
  info.case_id = case_id;
  info.case_type = infer_case_type(case_id).value();
  info.tokens = parse_case_tokens(case_id);
  return info;
}

}  // namespace

TEST_CASE("case names are tokenised like campaign.py", "[campaign]") {
  const auto channel = parse_case_tokens("000_f20_chan_n7e17cm3_L5mm_d150um_focm5mm_rz");
  CHECK(channel.laser_case == "20");
  CHECK(channel.density == "7e17cm3");
  CHECK(channel.plateau == "5");
  CHECK(channel.focus == "m5mm");
  CHECK(channel.diameter == "150");
  CHECK_FALSE(channel.ref_density);

  const auto uniform = parse_case_tokens("243_f20_uni_n7e17cm3_L5mm_refd500um_focm5mm_rz");
  CHECK(uniform.density == "7e17cm3");
  CHECK_FALSE(uniform.diameter);
  CHECK(uniform.focus == "m5mm");

  const auto vacuum = parse_case_tokens("324_f20_vac_L5mm_refd500um_focm5mm_rz");
  CHECK_FALSE(vacuum.density);
  CHECK_FALSE(vacuum.ref_density);

  const auto upper = parse_case_tokens("010_F20_CHAN_N4E18CM3_L2P5MM_D300UM_FOCP500UM_RZ");
  CHECK(upper.laser_case == "20");
  CHECK(upper.density == "4e18cm3");
  CHECK(upper.plateau == "2p5");
  CHECK(upper.diameter == "300");
  CHECK(upper.focus == "p500um");

  const auto dotted = parse_case_tokens("x_f32_chan_n4.5e18_L2.5mm_foc1.5mm_refn3e18");
  CHECK(dotted.density == "4p5e18");
  CHECK(dotted.plateau == "2p5");
  CHECK(dotted.focus == "1p5mm");
  CHECK(dotted.ref_density == "3e18");

  CHECK(infer_case_type("000_f20_chan_n7e17cm3") == CaseType::Channel);
  CHECK(infer_case_type("a-uniform-b") == CaseType::Uniform);
  CHECK(infer_case_type("run/vacuum") == CaseType::Vacuum);
  CHECK_FALSE(infer_case_type("channelized_run"));
}

TEST_CASE("triplets share uniform and vacuum baselines across channel diameters", "[campaign]") {
  const std::vector<CaseInfo> cases{
      make_case("000_f20_chan_n4e18cm3_L5mm_d150um_focm5mm_rz"),
      make_case("001_f20_chan_n4e18cm3_L5mm_d300um_focm5mm_rz"),
      make_case("243_f20_uni_n4e18cm3_L5mm_refd500um_focm5mm_rz"),
      make_case("324_f20_vac_L5mm_refd500um_focm5mm_rz"),
      make_case("400_f20_uni_n7e17cm3_L5mm_focm5mm_rz"),
  };
  const auto triplets = build_triplets(cases);
  REQUIRE(triplets.size() == 3);
  CHECK(triplets[0].label() == "f20_n4e18cm3_L5mm_d150um_focm5mm");
  CHECK(triplets[0].complete());
  CHECK(triplets[1].label() == "f20_n4e18cm3_L5mm_d300um_focm5mm");
  CHECK(triplets[1].uniform->case_id == "243_f20_uni_n4e18cm3_L5mm_refd500um_focm5mm_rz");
  CHECK(triplets[1].vacuum->case_id == "324_f20_vac_L5mm_refd500um_focm5mm_rz");
  CHECK(triplets[2].label() == "f20_n7e17cm3_L5mm_focm5mm");
  CHECK_FALSE(triplets[2].channel);
  CHECK(triplets[2].vacuum);
}

TEST_CASE("readiness follows min_h5 and the newest-file age gate", "[campaign]") {
  CaseInfo info = make_case("000_f20_chan_n4e18cm3_L5mm_d150um_focm5mm_rz");
  info.h5_count = 3;
  info.newest_h5_mtime_s = 1000.0;
  CHECK(case_has_min_h5(info, 2));
  CHECK_FALSE(case_has_min_h5(info, 4));
  CHECK(case_has_stable_h5(info, 0.0, 1000.0));
  CHECK_FALSE(case_has_stable_h5(info, 10.0, 1000.0 + 9.0 * 60.0));
  CHECK(case_has_stable_h5(info, 10.0, 1000.0 + 10.0 * 60.0));
  CHECK(case_is_ready(info, 2, 0.0, 0.0));
  info.newest_h5_mtime_s.reset();
  CHECK_FALSE(case_has_stable_h5(info, 1.0, 5000.0));
}

TEST_CASE("infer_case_info_from_dir prefers diags/fields over legacy diag1", "[campaign]") {
  TempDir tmp("guiding_tests_diag_resolution");
  const fs::path case_dir = tmp.path / "000_f20_chan_n4e18cm3_L5mm_d150um_foc0um_rz";
  touch(case_dir / "diags" / "diag1" / "openpmd_000000.h5");
  const auto legacy = infer_case_info_from_dir(case_dir);
  REQUIRE(legacy);
  CHECK(legacy->diag_dir == case_dir / "diags" / "diag1");
  CHECK(legacy->h5_count == 1);

  touch(case_dir / "diags" / "fields" / "openpmd_000000.h5");
  touch(case_dir / "diags" / "fields" / "nested" / "openpmd_000100.h5");
  touch(case_dir / "diags" / "fields" / "notes.txt");
  const auto modern = infer_case_info_from_dir(case_dir);
  REQUIRE(modern);
  CHECK(modern->diag_dir == case_dir / "diags" / "fields");
  CHECK(modern->h5_count == 2);
  CHECK(modern->newest_h5_mtime_s.has_value());

  CHECK_FALSE(infer_case_info_from_dir(tmp.path / "not_a_case"));
}

TEST_CASE("cases_full.tsv parsing handles a BOM, comments, NA cells and type overrides", "[campaign]") {
  TempDir tmp("guiding_tests_cases_full");
  fs::create_directories(tmp.path / "runs" / "000_f20_chan_n4e18cm3_L5mm_foc0um_rz");
  std::ofstream(tmp.path / "cases_full.tsv") << "\xEF\xBB\xBF"
                                                "CASE_ID\tcase_dir\tprofile\n"
                                                "# generated by the campaign launcher\n"
                                                "000\truns/000_f20_chan_n4e18cm3_L5mm_foc0um_rz\tNA\n"
                                                "001\truns/001_f20_chan_n4e18cm3_L5mm_foc0um_rz\tuniform\n"
                                                "\t\t\n";
  const auto cases = load_cases_from_cases_full(tmp.path);
  REQUIRE(cases.size() == 2);
  CHECK(cases[0].case_id == "000_f20_chan_n4e18cm3_L5mm_foc0um_rz");
  CHECK(cases[0].case_type == CaseType::Channel);
  CHECK(cases[0].source == "cases_full.tsv");
  CHECK(cases[0].case_dir == tmp.path / "runs" / "000_f20_chan_n4e18cm3_L5mm_foc0um_rz");
  CHECK(cases[1].case_type == CaseType::Uniform);
}

TEST_CASE("directory discovery is sorted and skips non-case entries", "[campaign]") {
  TempDir tmp("guiding_tests_discover_by_name");
  fs::create_directories(tmp.path / "002_f20_vac_L5mm_foc0um_rz");
  fs::create_directories(tmp.path / "000_f20_chan_n4e18cm3_L5mm_foc0um_rz");
  fs::create_directories(tmp.path / "analysis_outputs");
  touch(tmp.path / "001_f20_uni_n4e18cm3_L5mm_foc0um_rz.log");
  const auto cases = discover_cases(tmp.path);
  REQUIRE(cases.size() == 2);
  CHECK(cases[0].case_id == "000_f20_chan_n4e18cm3_L5mm_foc0um_rz");
  CHECK(cases[1].case_id == "002_f20_vac_L5mm_foc0um_rz");
  CHECK(cases[0].source == "name");
  CHECK(cases[0].diag_dir == tmp.path / "000_f20_chan_n4e18cm3_L5mm_foc0um_rz" / "diags" / "fields");
}

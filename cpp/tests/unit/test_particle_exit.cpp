#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <filesystem>
#include <fstream>
#include <string>

#include "guiding/products/particle_campaign.hpp"
#include "guiding/products/particle_exit.hpp"
#include "guiding/products/particle_reduction.hpp"
#include "guiding/table/py_format.hpp"

namespace {

namespace fs = std::filesystem;
using namespace guiding;

fs::path scratch_dir(const std::string& name) {
  const fs::path path = fs::temp_directory_path() / ("guiding_tests_" + name);
  fs::remove_all(path);
  fs::create_directories(path);
  return path;
}

void write_text(const fs::path& path, const std::string& text) {
  fs::create_directories(path.parent_path());
  std::ofstream(path, std::ios::binary) << text;
}

}  // namespace

// Expected values were produced with scripts/analyze_particle_case.py:parse_case_env.
TEST_CASE("case.env parsing follows shlex and the quote-stripping fallback", "[particles][exit]") {
  const fs::path dir = scratch_dir("case_env");
  write_text(dir / "case.env", "# comment\n"
                               "export A=\"2.5\"\n"
                               "B = 'x y' z\n"
                               "C=a\\ b\n"
                               "D=\"unterminated\n"
                               "E='it'\"'\"'s'\n"
                               "F=\n"
                               "G=\"a\\\"b\"\n"
                               "H='a\\b'\n"
                               " export I=1_000\n"
                               "J=abc # tail\n"
                               "K=\"\\$HOME\"\n"
                               "L=a=b\n"
                               "M=\"\"\n");
  const auto env = products::parse_case_env(dir / "case.env");
  CHECK(env.at("A") == "2.5");
  CHECK(env.at("B") == "x y");
  CHECK(env.at("C") == "a b");
  CHECK(env.at("D") == "unterminated");
  CHECK(env.at("E") == "it's");
  CHECK(env.at("F").empty());
  CHECK(env.at("G") == "a\"b");
  CHECK(env.at("H") == "a\\b");
  CHECK(env.at("I") == "1_000");
  CHECK(env.at("J") == "abc");
  CHECK(env.at("K") == "\\$HOME");
  CHECK(env.at("L") == "a=b");
  CHECK(env.at("M").empty());
  CHECK(env.size() == 13);

  CHECK(products::get_float_env(env, {"F", "M", "I"}) == 1000.0);
  CHECK(products::get_float_env(env, {"B", "A"}) == 2.5);
  CHECK_FALSE(products::get_float_env(env, {"missing", "B"}).has_value());
  CHECK(products::parse_case_env(dir / "absent.env").empty());
  fs::remove_all(dir);
}

TEST_CASE("Python float and int text parsing", "[particles][exit]") {
  using table::parse_py_float;
  using table::parse_py_int;
  CHECK(parse_py_float("1_000") == 1000.0);
  CHECK(parse_py_float(" 2.5 ") == 2.5);
  CHECK(parse_py_float("1e3") == 1000.0);
  CHECK(parse_py_float(".5") == 0.5);
  CHECK(parse_py_float("5.") == 5.0);
  CHECK(parse_py_float("-inf") == -INFINITY);
  CHECK(parse_py_float("Infinity") == INFINITY);
  CHECK(std::isnan(*parse_py_float("nan")));
  CHECK(parse_py_float("1e1_0") == 1.0e10);
  CHECK(parse_py_float("+.5e-3") == 0.0005);
  for (const char* invalid : {"1__0", "_1", "0x10", "1.5e", "", "e5", "nan(1)", "1.5 x"}) {
    CAPTURE(invalid);
    CHECK_FALSE(parse_py_float(invalid).has_value());
  }
  CHECK(parse_py_int("6150") == 6150);
  CHECK(parse_py_int(" 6_150 ") == 6150);
  CHECK(parse_py_int("+7") == 7);
  CHECK(parse_py_int("-3") == -3);
  for (const char* invalid : {"6150.0", "0x1", "", "1_", "99999999999999999999"}) {
    CAPTURE(invalid);
    CHECK_FALSE(parse_py_int(invalid).has_value());
  }
}

TEST_CASE("plateau length from the case name", "[particles][exit]") {
  using products::infer_plateau_length_from_case_name;
  CHECK(infer_plateau_length_from_case_name("000_f20_uni_L2mm_x") == 2.0);
  CHECK(infer_plateau_length_from_case_name("a_L2p5mm_b") == 2.5);
  CHECK(infer_plateau_length_from_case_name("a_L2.5mm_") == 2.5);
  CHECK_FALSE(infer_plateau_length_from_case_name("a_L2p5mm").has_value());
  CHECK(infer_plateau_length_from_case_name("a_Lp5mm_b_L3mm_c") == 3.0);
  CHECK(infer_plateau_length_from_case_name("a_L2p5pmm_b_L4mm_") == 4.0);
  CHECK(infer_plateau_length_from_case_name("_L12mm_") == 12.0);
}

TEST_CASE("nearest and exact particle iteration selection", "[particles][exit]") {
  const std::vector<std::int64_t> iterations{0, 5400, 6800};
  const auto tie = products::nearest_particle_iteration(iterations, 6100, "diag");
  CHECK(std::get<std::int64_t>(*tie.get("selected_particle_iteration")) == 5400);
  CHECK(std::get<std::int64_t>(*tie.get("target_iteration_delta")) == -700);

  CHECK_THROWS_WITH(products::require_exact_particle_iteration(iterations, 6100, "diag"),
                    "Exact particle exit dump is missing: target iteration=6100, available min=0, available "
                    "max=6800, count=3. Refusing nearest/last fallback.");

  table::Record info = tie;
  CHECK_THROWS_WITH(products::validate_exit_iteration_alignment(info, 100),
                    "target-iteration alignment can only be required for --which exit");
  info.set("selection_mode", std::string("exit"));
  info.set("target_guiding_iteration", std::int64_t{6100});
  CHECK_NOTHROW(products::validate_exit_iteration_alignment(info, 700));
  CHECK_THROWS_WITH(products::validate_exit_iteration_alignment(info, 699),
                    Catch::Matchers::ContainsSubstring("delta=-700, allowed=699"));
  info.set("target_iteration_delta", std::int64_t{5});
  CHECK_THROWS_WITH(products::validate_exit_iteration_alignment(info, 700),
                    "inconsistent exit selection metadata: selected=5400, target=6100, delta=5");
}

TEST_CASE("resolved particle exit targets are validated", "[particles][exit]") {
  const fs::path dir = scratch_dir("resolved");
  const fs::path path = dir / "resolved_parameters.json";
  const auto message = [&](const std::string& json) {
    write_text(path, json);
    try {
      (void)products::read_resolved_particle_exit_target(path, products::ExitKind::Plateau);
    } catch (const std::exception& error) {
      return std::string(error.what());
    }
    return std::string("no error");
  };
  const std::string shown = table::python_path_string(path);
  CHECK(message("[1]") == "Resolved simulation parameters must be a JSON object: " + shown);
  CHECK(message("{") == "Could not read resolved simulation parameters: " + shown);
  CHECK(message("{}") == "Resolved simulation parameters lack particle_diagnostic_targets in " + shown);
  CHECK(message(R"({"particle_diagnostic_targets": {"capillary_exit": {}}})") ==
        "Resolved simulation parameters lack particle target 'plateau_exit' in " + shown);
  CHECK(message(R"({"particle_diagnostic_targets": {"plateau_exit": {}}})") ==
        "Particle target 'plateau_exit' lacks iteration in " + shown);
  CHECK(message(R"({"particle_diagnostic_targets": {"plateau_exit": {"iteration": true}}})") ==
        "Particle target 'plateau_exit' iteration must be an integer");
  CHECK(message(R"({"particle_diagnostic_targets": {"plateau_exit": {"iteration": "6150.0"}}})") ==
        "Particle target 'plateau_exit' iteration is not numeric in " + shown);
  CHECK(message(R"({"particle_diagnostic_targets": {"plateau_exit": {"iteration": 6150.5}}})") ==
        "Particle target 'plateau_exit' iteration must be a non-negative integer");
  CHECK(message(R"({"particle_diagnostic_targets": {"plateau_exit": {"iteration": -1}}})") ==
        "Particle target 'plateau_exit' iteration must be a non-negative integer");
  CHECK(message(R"({"particle_diagnostic_targets": {"plateau_exit": {"iteration": 1, "dump_distance_m": null}}})") ==
        "Particle target 'plateau_exit' field 'dump_distance_m' is not numeric in " + shown);
  CHECK(message(R"({"particle_diagnostic_targets": {"plateau_exit": {"iteration": 1, "dump_distance_m": "nan"}}})") ==
        "Particle target 'plateau_exit' field 'dump_distance_m' is not finite in " + shown);

  write_text(path, R"({"particle_diagnostic_targets": {"plateau_exit": {"iteration": " 6_150 ", "target_distance_m": "6.2e-3"}}})");
  const auto target = products::read_resolved_particle_exit_target(path, products::ExitKind::Plateau);
  CHECK(std::get<std::int64_t>(*target.get("target_particle_iteration")) == 6150);
  CHECK(std::get<double>(*target.get("target_propagation_mm")) == 6.2e-3 * 1.0e3);
  fs::remove_all(dir);
}

TEST_CASE("particle case directory and campaign globbing", "[particles][campaign]") {
  CHECK(products::particle_case_dir("runs/CASE/diags/plasma_electrons/", "elsewhere/out") == fs::path("runs/CASE"));
  CHECK(products::particle_case_dir("diags/x", "out") == fs::path("."));
  CHECK(products::particle_case_dir("runs/CASE/diags/electrons/openpmd", "runs/CASE/particle_analysis") ==
        fs::path("runs/CASE"));
  CHECK(products::particle_case_dir("x", "out") == fs::path("."));
  CHECK(products::case_id_from_case_dir("runs/017_from_base_f20") == "017");

  const fs::path root = scratch_dir("glob");
  for (const char* name : {"001_from_a", "002_from_b", ".003_from_c", "010_other", "sub/004_from_d"}) {
    fs::create_directories(root / name);
  }
  write_text(root / "005_from_file", "not a directory");
  const auto names = [&](const std::string& pattern) {
    std::vector<std::string> out;
    for (const auto& path : products::glob_directories(root, pattern)) {
      out.push_back(fs::relative(path, root).generic_string());
    }
    return out;
  };
  CHECK(names("0*_from_*") == std::vector<std::string>{"001_from_a", "002_from_b"});
  CHECK(names("*_from_*") == std::vector<std::string>{".003_from_c", "001_from_a", "002_from_b"});
  CHECK(names("*/0*") == std::vector<std::string>{"sub/004_from_d"});
  CHECK(names("0[!0]*") == std::vector<std::string>{"010_other"});
  CHECK(names("**/*_from_*") ==
        std::vector<std::string>{".003_from_c", "001_from_a", "002_from_b", "sub/004_from_d"});
  CHECK_THROWS(products::glob_directories(root, ""));
  fs::remove_all(root);
}

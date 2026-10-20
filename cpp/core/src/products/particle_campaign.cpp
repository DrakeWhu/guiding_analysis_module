#include "guiding/products/particle_campaign.hpp"

#include <fnmatch.h>

#include <algorithm>
#include <mutex>
#include <stdexcept>

#include <fmt/format.h>

#include "guiding/campaign/discovery.hpp"
#include "guiding/exec/parallel.hpp"
#include "guiding/table/py_format.hpp"

namespace guiding::products {
namespace {

namespace fs = std::filesystem;
using table::py_float_repr;
using table::python_path_string;

}  // namespace

std::vector<fs::path> glob_directories(const fs::path& root, const std::string& pattern) {
  std::vector<fs::path> current{root};
  std::size_t start = 0;
  bool matched_any_segment = false;
  while (start <= pattern.size()) {
    const std::size_t end = std::min(pattern.find('/', start), pattern.size());
    const std::string segment = pattern.substr(start, end - start);
    start = end + 1;
    if (!segment.empty() && segment != ".") {
      matched_any_segment = true;
      std::vector<fs::path> next;
      std::error_code error;
      if (segment == "**") {
        for (const auto& base : current) {
          next.push_back(base);
          for (auto it = fs::recursive_directory_iterator(base, fs::directory_options::skip_permission_denied, error);
               !error && it != fs::recursive_directory_iterator(); it.increment(error)) {
            if (it->is_directory(error)) {
              next.push_back(it->path());
            }
          }
        }
      } else if (segment.find_first_of("*?[") == std::string::npos) {
        for (const auto& base : current) {
          if (fs::exists(base / segment, error)) {
            next.push_back(base / segment);
          }
        }
      } else {
        for (const auto& base : current) {
          if (!fs::is_directory(base, error)) {
            continue;
          }
          for (auto it = fs::directory_iterator(base, error); !error && it != fs::directory_iterator();
               it.increment(error)) {
            const std::string name = it->path().filename().string();
            // fnmatch without FNM_PERIOD: pathlib's "*" also matches dot files.
            if (::fnmatch(segment.c_str(), name.c_str(), FNM_NOESCAPE) == 0) {
              next.push_back(it->path());
            }
          }
        }
      }
      current = std::move(next);
    }
    if (end == pattern.size()) {
      break;
    }
  }
  if (!matched_any_segment) {
    throw std::invalid_argument(fmt::format("Unacceptable pattern: {}", table::py_str_repr(pattern)));
  }
  std::vector<fs::path> directories;
  for (const auto& path : current) {
    std::error_code error;
    if (fs::is_directory(path, error)) {
      directories.push_back(path);
    }
  }
  std::sort(directories.begin(), directories.end());
  directories.erase(std::unique(directories.begin(), directories.end()), directories.end());
  return directories;
}

ParticleCampaignResult run_particle_campaign(const ParticleCampaignOptions& options, const LineSink& log) {
  const auto emit = [&](const std::string& line) {
    if (log) {
      log(line);
    }
  };
  const auto cases = glob_directories(options.campaign_root, options.case_glob);
  if (cases.empty()) {
    throw std::runtime_error(
        fmt::format("No case directories found with glob {}", table::py_str_repr(options.case_glob)));
  }

  const auto& common = options.case_options;
  emit("=== Particle campaign analysis ===");
  emit(fmt::format("campaign_root     = {}", python_path_string(options.campaign_root)));
  emit(fmt::format("cases discovered  = {}", cases.size()));
  emit(fmt::format("diag name         = {}", options.particle_diag_name));
  emit(fmt::format("species           = {}", options.species_text));
  emit(fmt::format("which             = {}", particle_which_name(common.which)));
  emit(fmt::format("exit_kind         = {}", exit_kind_name(common.exit_kind)));
  emit(fmt::format("target_prop_mm    = {}", common.target_propagation_mm
                                                 ? py_float_repr(*common.target_propagation_mm)
                                                 : std::string("None")));
  emit(fmt::format("hot_energy_mev    = {}", py_float_repr(common.hot_energy_mev)));
  emit("==================================");

  struct CaseResult {
    std::vector<std::string> lines;
    bool ok = false;
    bool done = false;
  };
  std::vector<CaseResult> results(cases.size());
  std::mutex mutex;
  std::size_t next_to_emit = 0;

  exec::parallel_for(cases.size(), std::max(1U, options.threads), [&](std::size_t index) {
    const fs::path& case_dir = cases[index];
    CaseResult result;
    const auto case_log = [&result](const std::string& line) { result.lines.push_back(line); };
    fs::path diag;
    bool have_diag = false;
    try {
      diag = campaign::resolve_particle_diag_dir(case_dir, options.species_text, options.particle_diag_name);
      have_diag = true;
    } catch (const std::exception& error) {
      case_log(fmt::format("[MISSING] {}", error.what()));
    }
    if (have_diag) {
      case_log(fmt::format("[CASE] {}", case_dir.filename().string()));
      try {
        ParticleCaseOptions case_options = common;
        case_options.diag = diag;
        case_options.outdir = case_dir / options.outdir_name;
        case_options.threads = 1;
        if (options.use_resolved_parameters) {
          case_options.resolved_parameters = case_dir / "resolved_parameters.json";
        }
        run_particle_case(case_options, case_log);
        result.ok = true;
      } catch (const std::exception& error) {
        case_log(fmt::format("[FAIL] {}", error.what()));
        case_log(fmt::format("[ERROR] case failed with return code 1: {}", python_path_string(case_dir)));
      }
    }

    std::lock_guard lock(mutex);
    result.done = true;
    results[index] = std::move(result);
    while (next_to_emit < results.size() && results[next_to_emit].done) {
      for (const auto& line : results[next_to_emit].lines) {
        emit(line);
      }
      ++next_to_emit;
    }
  });

  ParticleCampaignResult summary;
  for (const auto& result : results) {
    (result.ok ? summary.ok : summary.failed) += 1;
  }
  emit("=== Particle campaign summary ===");
  emit(fmt::format("ok     = {}", summary.ok));
  emit(fmt::format("failed = {}", summary.failed));
  return summary;
}

}  // namespace guiding::products

#pragma once

#include <cstdint>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <vector>

#include "guiding/physics/particles.hpp"
#include "guiding/products/particle_exit.hpp"
#include "guiding/table/record.hpp"

// Port of scripts/analyze_particle_case.py (CSV products only; no plots).
namespace guiding::products {

struct ParticleCaseOptions {
  std::filesystem::path diag;
  std::filesystem::path outdir;
  std::vector<std::string> species{"electrons"};
  ParticleWhich which = ParticleWhich::Last;
  int stride = 1;
  double hot_energy_mev = 10.0;
  physics::Longitudinal longitudinal = physics::Longitudinal::Z;
  bool forward_only = true;
  std::optional<double> exit_window_mm;
  ExitKind exit_kind = ExitKind::Plateau;
  std::optional<double> target_propagation_mm;
  std::optional<std::filesystem::path> guiding_metrics;
  std::optional<std::filesystem::path> resolved_parameters;
  std::optional<std::int64_t> maximum_target_iteration_delta;
  std::optional<double> downramp_mm;
  std::vector<double> acceptance_theta_cuts_mrad{physics::kDefaultAcceptanceThetaCutsMrad.begin(),
                                                 physics::kDefaultAcceptanceThetaCutsMrad.end()};
  std::vector<double> acceptance_energy_cuts_mev{physics::kDefaultAcceptanceEnergyMinMeV.begin(),
                                                 physics::kDefaultAcceptanceEnergyMinMeV.end()};
  physics::Soft50Config soft50;
  std::vector<double> soft50_curve_energy_low_mev{5.0, 10.0};
  // Plot options, only echoed in the run header.
  double spectrum_emin_mev = 0.0;
  bool spectrum_log_y = false;

  bool skip_existing = false;
  bool overwrite = false;
  bool plots_only = false;
  unsigned threads = 1;   // iterations analysed concurrently (--which all)
  bool raw_reads = true;  // pread fast path for contiguous datasets
};

struct ParticleCaseTables {
  table::Record selection_info;
  std::vector<table::Record> summary_rows;
  std::vector<table::Record> acceptance_rows;
  std::vector<table::Record> soft50_curve_rows;
};

// Case directory as the script derives it: DIAG/../.. when DIAG sits in a
// "diags" directory, else the parent of OUTDIR (lexical, like pathlib).
[[nodiscard]] std::filesystem::path particle_case_dir(const std::filesystem::path& diag,
                                                      const std::filesystem::path& outdir);
// Text before the first "_" of the case directory name.
[[nodiscard]] std::string case_id_from_case_dir(const std::filesystem::path& case_dir);

// "[a, b]" / "a,b" / "a b" -> species list; throws when empty or repeated.
[[nodiscard]] std::vector<std::string> parse_species_list(std::string_view text);
[[nodiscard]] std::vector<double> parse_float_list(std::string_view text);

// Reads the series, selects iterations and computes all rows. Logs the
// [SERIES], [SELECTION] and [READ] lines; `log` must be thread-safe when
// options.threads > 1.
[[nodiscard]] ParticleCaseTables compute_particle_case_tables(const ParticleCaseOptions& options,
                                                              const LineSink& log);

[[nodiscard]] std::string format_particle_summary_csv(std::span<const table::Record> rows);
[[nodiscard]] std::string format_particle_acceptance_csv(std::span<const table::Record> rows);
[[nodiscard]] std::string format_particle_soft50_curves_csv(std::span<const table::Record> rows);

enum class ParticleCaseOutcome { Written, Skipped, PlotsOnly };

// analyze_particle_case.py main(): argument checks, skip/overwrite policy,
// header, computation and the three CSV files.
ParticleCaseOutcome run_particle_case(const ParticleCaseOptions& options, const LineSink& log);

}  // namespace guiding::products

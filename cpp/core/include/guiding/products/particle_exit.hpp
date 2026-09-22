#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "guiding/table/record.hpp"

// Particle iteration selection: cap_guiding/particle_exit.py plus the
// selection helpers of scripts/analyze_particle_case.py.
namespace guiding::products {

enum class ExitKind { Plateau, Capillary };
enum class ParticleWhich { Last, All, Exit };

[[nodiscard]] ExitKind parse_exit_kind(std::string_view name);
[[nodiscard]] const char* exit_kind_name(ExitKind kind) noexcept;
[[nodiscard]] ParticleWhich parse_particle_which(std::string_view name);
[[nodiscard]] const char* particle_which_name(ParticleWhich which) noexcept;

using LineSink = std::function<void(const std::string&)>;

// ---- case.env -------------------------------------------------------------

// KEY=VALUE lines ("export " prefix allowed, first shlex token of the value).
using CaseEnv = std::map<std::string, std::string>;
[[nodiscard]] CaseEnv parse_case_env(const std::filesystem::path& path);

// First listed, non-empty entry that parses as a Python float.
[[nodiscard]] std::optional<double> get_float_env(const CaseEnv& env, std::initializer_list<const char*> names);

// "_L<length>mm_" in the case name ("p" as decimal point).
[[nodiscard]] std::optional<double> infer_plateau_length_from_case_name(std::string_view case_name);

// ---- targets ----------------------------------------------------------------

// read_resolved_particle_exit_target: particle_diagnostic_targets.<kind>_exit.
[[nodiscard]] table::Record read_resolved_particle_exit_target(const std::filesystem::path& resolved_parameters,
                                                               ExitKind kind);

// (target_end_z - plasma_start_z) in mm from CASE/resolved_parameters.json, or
// nullopt when that file does not exist.
[[nodiscard]] std::optional<double> target_propagation_from_resolved_parameters(
    const std::filesystem::path& case_dir, ExitKind kind);

// Legacy physical exit target: explicit value, resolved parameters, case.env
// lengths, then the case name. Warnings go to `log`.
[[nodiscard]] double target_propagation_from_case(const std::filesystem::path& case_dir, ExitKind kind,
                                                  std::optional<double> target_propagation_mm,
                                                  std::optional<double> downramp_mm, const LineSink& log);

// Guiding iteration whose propagation_mm is nearest the target.
[[nodiscard]] table::Record read_guiding_iteration_at_propagation(const std::filesystem::path& metrics_csv,
                                                                  double target_propagation_mm);

// ---- particle iterations ----------------------------------------------------

[[nodiscard]] table::Record require_exact_particle_iteration(std::span<const std::int64_t> iterations,
                                                             std::int64_t target_iteration,
                                                             const std::filesystem::path& diag);

[[nodiscard]] table::Record nearest_particle_iteration(std::span<const std::int64_t> iterations,
                                                       std::int64_t target_iteration,
                                                       const std::filesystem::path& diag);

// Throws unless |selected - target guiding iteration| <= maximum_delta.
void validate_exit_iteration_alignment(const table::Record& selection_info, std::int64_t maximum_delta);

struct SelectionRequest {
  ParticleWhich which = ParticleWhich::Last;
  int stride = 1;
  std::filesystem::path diag;
  std::filesystem::path case_dir;
  std::optional<std::filesystem::path> guiding_metrics;
  ExitKind exit_kind = ExitKind::Plateau;
  std::optional<double> target_propagation_mm;
  std::optional<double> downramp_mm;
  std::optional<std::filesystem::path> resolved_parameters;
};

struct IterationSelection {
  std::vector<std::int64_t> iterations;
  table::Record info;
};

// resolve_iterations over the (sorted) iterations of the particle series.
[[nodiscard]] IterationSelection resolve_iterations(std::span<const std::int64_t> series_iterations,
                                                    const SelectionRequest& request, const LineSink& log);

}  // namespace guiding::products

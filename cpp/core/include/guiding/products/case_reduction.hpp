#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <span>
#include <string>
#include <vector>

#include "guiding/physics/field_metrics.hpp"

// Port of cap_guiding/metrics.py:compute_case_rows / write_case_csv.
namespace guiding::products {

struct GuidingMetricsRow {
  std::int64_t iteration = 0;
  double time_fs = 0.0;
  double z_min_um = 0.0;
  double z_max_um = 0.0;
  physics::LaserMetrics laser{};
  physics::WakeMetrics wake{};
  double propagation_mm = 0.0;
  double z_peak_relative_um = 0.0;
};

struct CaseReductionOptions {
  physics::FieldParams params;
  unsigned threads = 0;    // 0 selects exec::default_thread_count()
  bool raw_reads = true;   // pread fast path for contiguous datasets
  // Called from worker threads when an iteration starts; must be thread-safe.
  std::function<void(std::int64_t iteration)> on_iteration;
};

// Reduces one field-diagnostic iteration (E/r, E/t, E/z at theta = 0).
[[nodiscard]] GuidingMetricsRow reduce_field_iteration(const std::filesystem::path& file, std::int64_t iteration,
                                                       const physics::FieldParams& params, bool raw_reads = true);

// Reads every (stride-selected) iteration of a diagnostic directory in
// parallel; rows are in iteration order. Throws like the Python reference when
// there are no iterations or no valid laser dump.
[[nodiscard]] std::vector<GuidingMetricsRow> compute_case_rows(const std::filesystem::path& diag,
                                                               const CaseReductionOptions& options);

// Row validity used for normalisations (metrics.py:valid_laser_mask).
[[nodiscard]] bool is_valid_laser_row(const GuidingMetricsRow& row);

[[nodiscard]] const std::vector<std::string>& guiding_metrics_columns();

// guiding_metrics.csv exactly as csv.DictWriter writes it.
[[nodiscard]] std::string format_guiding_metrics_csv(std::span<const GuidingMetricsRow> rows);
void write_guiding_metrics_csv(std::span<const GuidingMetricsRow> rows, const std::filesystem::path& path);

}  // namespace guiding::products

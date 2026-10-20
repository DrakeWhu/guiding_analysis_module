#include "guiding/products/case_reduction.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <tuple>

#include <fmt/format.h>

#include "guiding/exec/parallel.hpp"
#include "guiding/io/h5.hpp"
#include "guiding/io/openpmd_series.hpp"
#include "guiding/io/thetamode.hpp"
#include "guiding/numeric/npcompat.hpp"
#include "guiding/table/csv.hpp"

namespace guiding::products {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

}  // namespace

GuidingMetricsRow reduce_field_iteration(const std::filesystem::path& file_path, std::int64_t iteration,
                                         const physics::FieldParams& params, bool raw_reads) {
  auto file = h5::File::open_read_only(file_path);
  file.set_raw_reads_enabled(raw_reads);

  GuidingMetricsRow row;
  row.iteration = iteration;

  io::PositiveRSlice ex;
  io::PositiveRSlice ey;
  {
    const auto e_r = io::read_thetamode_component(file, iteration, "E", "r");
    const auto e_t = io::read_thetamode_component(file, iteration, "E", "t");
    std::tie(ex, ey) = io::cartesian_xy(e_r, e_t);
  }
  const auto ez = io::component_slice(io::read_thetamode_component(file, iteration, "E", "z"));

  row.laser = physics::weighted_laser_metrics(ex, ey, params.smooth_um, params.lambda0_m);
  row.wake = physics::wake_metrics(ez, row.laser.z_peak_um, params.wake_behind_um, params.wake_gap_um);
  row.time_fs = std::isfinite(ex.time_s) ? ex.time_s * 1.0e15 : kNaN;
  row.z_min_um = np::min_value<double>(ex.z_um);
  row.z_max_um = np::max_value<double>(ex.z_um);
  return row;
}

bool is_valid_laser_row(const GuidingMetricsRow& row) {
  return std::isfinite(row.laser.z_peak_um) && std::isfinite(row.laser.waist_um) &&
         std::isfinite(row.laser.peak_I_proxy) && std::isfinite(row.laser.energy_proxy) &&
         row.laser.peak_I_proxy > 0.0 && row.laser.energy_proxy > 0.0;
}

std::vector<GuidingMetricsRow> compute_case_rows(const std::filesystem::path& diag,
                                                 const CaseReductionOptions& options) {
  const auto series = io::FileSeries::scan(diag);
  const auto iterations = io::apply_stride(series.iterations(), options.params.stride);
  if (iterations.empty()) {
    throw std::runtime_error(fmt::format("No iterations found in diagnostic: {}", diag.string()));
  }

  std::vector<GuidingMetricsRow> rows(iterations.size());
  const unsigned threads = options.threads == 0 ? exec::default_thread_count() : options.threads;
  exec::parallel_for(iterations.size(), threads, [&](std::size_t i) {
    if (options.on_iteration) {
      options.on_iteration(iterations[i]);
    }
    rows[i] = reduce_field_iteration(series.file(iterations[i]), iterations[i], options.params, options.raw_reads);
  });

  const bool any_valid = std::any_of(rows.begin(), rows.end(), is_valid_laser_row);
  if (!any_valid) {
    throw std::runtime_error("No valid laser dumps found: peak_I/energy are zero or NaN everywhere.");
  }

  // Monotonic moving-window coordinate (metrics.py:add_propagation_columns).
  const double z_max_0 = rows.front().z_max_um;
  for (auto& row : rows) {
    row.propagation_mm = (row.z_max_um - z_max_0) * 1.0e-3;
    row.z_peak_relative_um = row.laser.z_peak_um - row.z_min_um;
  }
  return rows;
}

const std::vector<std::string>& guiding_metrics_columns() {
  static const std::vector<std::string> columns{
      "iteration",      "time_fs",        "z_min_um",       "z_max_um",       "z_peak_um",
      "front_margin_um", "back_margin_um", "waist_um",       "peak_I_proxy",   "Eperp_peak_Vm",
      "a0_peak",        "energy_proxy",   "Ez_wake_max",    "Ez_wake_min",    "Ez_wake_absmax",
      "Ez_wake_rms",    "z_Ez_absmax_um", "z_Ez_absmax_rel_um", "propagation_mm", "z_peak_relative_um"};
  return columns;
}

std::string format_guiding_metrics_csv(std::span<const GuidingMetricsRow> rows) {
  if (rows.empty()) {
    throw std::invalid_argument("No rows to write");
  }
  const auto dialect = table::CsvDialect::python_csv();
  std::string out;
  table::append_csv_header(out, guiding_metrics_columns(), dialect);
  std::vector<table::Cell> cells;
  for (const auto& row : rows) {
    cells = {row.iteration,
             row.time_fs,
             row.z_min_um,
             row.z_max_um,
             row.laser.z_peak_um,
             row.laser.front_margin_um,
             row.laser.back_margin_um,
             row.laser.waist_um,
             row.laser.peak_I_proxy,
             row.laser.Eperp_peak_Vm,
             row.laser.a0_peak,
             row.laser.energy_proxy,
             row.wake.Ez_wake_max,
             row.wake.Ez_wake_min,
             row.wake.Ez_wake_absmax,
             row.wake.Ez_wake_rms,
             row.wake.z_Ez_absmax_um,
             row.wake.z_Ez_absmax_rel_um,
             row.propagation_mm,
             row.z_peak_relative_um};
    table::append_csv_record(out, cells, dialect);
  }
  return out;
}

void write_guiding_metrics_csv(std::span<const GuidingMetricsRow> rows, const std::filesystem::path& path) {
  table::write_file_atomically(path, format_guiding_metrics_csv(rows));
}

}  // namespace guiding::products

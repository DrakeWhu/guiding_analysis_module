#include "guiding/products/triplet_tables.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <stdexcept>

#include <fmt/format.h>

#include "guiding/campaign/case_metadata.hpp"
#include "guiding/numeric/npcompat.hpp"
#include "guiding/table/py_format.hpp"
#include "guiding/table/record.hpp"

namespace guiding::products {
namespace {

namespace fs = std::filesystem;
using table::Frame;
using table::Record;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
const std::array<std::string, 3> kCaseOrder{"channel", "uniform", "vacuum"};
const std::vector<std::string> kOptionalColumns{"Eperp_peak_Vm", "a0_peak"};

// pandas.to_numeric(series, errors="coerce")
Frame::Column to_numeric(const Frame::Column& column) {
  const auto* texts = std::get_if<Frame::Strings>(&column);
  if (texts == nullptr) {
    return column;
  }
  Frame::Integers integers;
  for (const auto& text : *texts) {
    const auto value = table::pandas_int64_field(text);
    if (!value) {
      break;
    }
    integers.push_back(*value);
  }
  if (!texts->empty() && integers.size() == texts->size()) {
    return integers;
  }
  Frame::Doubles reals;
  for (const auto& text : *texts) {
    reals.push_back(table::parse_pandas_number(text));
  }
  return reals;
}

Frame read_case_csv(const fs::path& path, const std::string& case_type) {
  Frame df = table::frame_from_csv(table::read_csv_file(path));
  std::vector<std::string> missing;
  for (const auto& column : triplet_required_columns()) {
    if (!df.has(column)) {
      missing.push_back(column);
    }
  }
  if (!missing.empty()) {
    throw std::runtime_error(fmt::format("{} is missing required columns: {}", table::python_path_string(path),
                                         table::python_list_repr(missing)));
  }
  df.set("case_type", Frame::Strings(df.row_count(), case_type));
  df.set("source_csv", Frame::Strings(df.row_count(), table::python_path_string(path)));
  for (const auto& column : triplet_required_columns()) {
    df.set(column, to_numeric(df.column(column)));
  }
  for (const auto& column : kOptionalColumns) {
    if (df.has(column)) {
      df.set(column, to_numeric(df.column(column)));
    }
  }
  return df;
}

std::vector<std::size_t> stable_ascending_order(const std::vector<double>& keys) {
  std::vector<std::size_t> order(keys.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    const bool a_nan = std::isnan(keys[a]);
    const bool b_nan = std::isnan(keys[b]);
    if (a_nan != b_nan) {
      return b_nan;
    }
    return !a_nan && keys[a] < keys[b];
  });
  return order;
}

std::size_t first_valid_index(const Frame& df) {
  const auto peak = df.doubles("peak_I_proxy");
  const auto energy = df.doubles("energy_proxy");
  const auto waist = df.doubles("waist_um");
  for (std::size_t i = 0; i < peak.size(); ++i) {
    if (std::isfinite(peak[i]) && std::isfinite(energy[i]) && std::isfinite(waist[i]) && peak[i] > 0.0 &&
        energy[i] > 0.0) {
      return i;
    }
  }
  throw std::runtime_error("No valid laser dump found in one case CSV");
}

Frame::Doubles divide(const Frame::Doubles& values, double denominator) {
  Frame::Doubles out(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    out[i] = values[i] / denominator;
  }
  return out;
}

Frame::Doubles divide(const Frame::Doubles& numerator, const Frame::Doubles& denominator) {
  Frame::Doubles out(numerator.size());
  for (std::size_t i = 0; i < numerator.size(); ++i) {
    out[i] = numerator[i] / denominator[i];
  }
  return out;
}

Frame add_case_normalizations(const Frame& input) {
  Frame df = input.take(stable_ascending_order(input.doubles("iteration")));
  const std::size_t i0 = first_valid_index(df);
  const std::size_t n = df.row_count();

  const auto peak = df.doubles("peak_I_proxy");
  const auto energy = df.doubles("energy_proxy");
  const auto waist = df.doubles("waist_um");
  df.set("peak_I_norm", divide(peak, peak[i0]));
  df.set("energy_norm", divide(energy, energy[i0]));
  df.set("waist_norm", divide(waist, waist[i0]));
  df.set("Ez_wake_absmax_GVm", divide(df.doubles("Ez_wake_absmax"), 1.0e9));

  if (df.has("a0_peak")) {
    const auto a0 = df.doubles("a0_peak");
    const double a0_0 = a0[i0];
    if (std::isfinite(a0_0) && a0_0 > 0.0) {
      df.set("a0_norm", divide(a0, a0_0));
    } else {
      df.set("a0_norm", Frame::Doubles(n, kNaN));
    }
  }
  const double reference_iteration = df.doubles("iteration")[i0];
  df.set("ref_iteration", Frame::Integers(n, static_cast<std::int64_t>(reference_iteration)));
  return df;
}

std::array<Frame, 3> align_common_iterations(const std::array<Frame, 3>& cases) {
  std::array<std::set<std::int64_t>, 3> iterations;
  for (std::size_t k = 0; k < 3; ++k) {
    for (double value : cases[k].doubles("iteration")) {
      if (!std::isnan(value)) {
        iterations[k].insert(static_cast<std::int64_t>(value));
      }
    }
  }
  std::set<std::int64_t> common = iterations[0];
  for (std::size_t k = 1; k < 3; ++k) {
    std::set<std::int64_t> kept;
    std::set_intersection(common.begin(), common.end(), iterations[k].begin(), iterations[k].end(),
                          std::inserter(kept, kept.begin()));
    common = std::move(kept);
  }
  if (common.empty()) {
    std::string details;
    for (std::size_t k = 0; k < 3; ++k) {
      const auto& its = iterations[k];
      details += (k == 0 ? "" : "; ") +
                 (its.empty() ? fmt::format("{}: n=0", kCaseOrder[k])
                              : fmt::format("{}: n={}, min={}, max={}", kCaseOrder[k], its.size(), *its.begin(),
                                            *its.rbegin()));
    }
    throw std::runtime_error("No common iterations across channel/uniform/vacuum. " + details);
  }

  // Channel is the limiting/reference case.
  const auto channel_iterations = cases[0].doubles("iteration");
  const double channel_last = np::max_value<double>(channel_iterations);
  std::set<std::int64_t> kept;
  for (std::int64_t iteration : common) {
    if (static_cast<double>(iteration) <= channel_last) {
      kept.insert(iteration);
    }
  }

  std::array<Frame, 3> aligned;
  for (std::size_t k = 0; k < 3; ++k) {
    const auto values = cases[k].doubles("iteration");
    std::vector<std::size_t> rows;
    for (std::size_t i = 0; i < values.size(); ++i) {
      if (!std::isnan(values[i]) && kept.contains(static_cast<std::int64_t>(values[i]))) {
        rows.push_back(i);
      }
    }
    const Frame subset = cases[k].take(rows);
    aligned[k] = subset.take(stable_ascending_order(subset.doubles("iteration")));
  }
  if (aligned[0].row_count() != aligned[1].row_count() || aligned[1].row_count() != aligned[2].row_count()) {
    throw std::runtime_error(fmt::format("Internal alignment error: {{'channel': {}, 'uniform': {}, 'vacuum': {}}}",
                                         aligned[0].row_count(), aligned[1].row_count(), aligned[2].row_count()));
  }
  return aligned;
}

// pd.concat(frames, ignore_index=True): outer union of columns in order of
// first appearance; int64 columns with missing rows become float64.
Frame concat_frames(const std::array<Frame, 3>& frames) {
  std::vector<std::string> names;
  for (const auto& frame : frames) {
    for (const auto& name : frame.names()) {
      if (std::find(names.begin(), names.end(), name) == names.end()) {
        names.push_back(name);
      }
    }
  }
  Frame out;
  for (const auto& name : names) {
    bool all_integers = true;
    bool any_strings = false;
    for (const auto& frame : frames) {
      if (!frame.has(name)) {
        all_integers = false;
        continue;
      }
      const auto& column = frame.column(name);
      all_integers = all_integers && std::holds_alternative<Frame::Integers>(column);
      any_strings = any_strings || std::holds_alternative<Frame::Strings>(column);
    }
    if (any_strings) {
      Frame::Strings values;
      for (const auto& frame : frames) {
        if (!frame.has(name)) {
          values.insert(values.end(), frame.row_count(), std::string());
          continue;
        }
        const auto& column = frame.column(name);
        if (const auto* texts = std::get_if<Frame::Strings>(&column)) {
          values.insert(values.end(), texts->begin(), texts->end());
        } else {
          for (double v : frame.doubles(name)) {
            values.push_back(std::isnan(v) ? std::string() : table::py_float_repr(v));
          }
        }
      }
      out.set(name, std::move(values));
    } else if (all_integers) {
      Frame::Integers values;
      for (const auto& frame : frames) {
        const auto& column = std::get<Frame::Integers>(frame.column(name));
        values.insert(values.end(), column.begin(), column.end());
      }
      out.set(name, std::move(values));
    } else {
      Frame::Doubles values;
      for (const auto& frame : frames) {
        if (!frame.has(name)) {
          values.insert(values.end(), frame.row_count(), kNaN);
          continue;
        }
        const auto column = frame.doubles(name);
        values.insert(values.end(), column.begin(), column.end());
      }
      out.set(name, std::move(values));
    }
  }
  return out;
}

// left.merge(right, on="iteration", how="inner"), keeping the left key order.
Frame merge_on_iteration(const Frame& left, const Frame& right) {
  const auto left_keys = left.doubles("iteration");
  const auto right_keys = right.doubles("iteration");
  std::map<double, std::vector<std::size_t>> right_rows;
  for (std::size_t j = 0; j < right_keys.size(); ++j) {
    right_rows[right_keys[j]].push_back(j);
  }
  std::vector<std::size_t> left_index;
  std::vector<std::size_t> right_index;
  for (std::size_t i = 0; i < left_keys.size(); ++i) {
    const auto it = right_rows.find(left_keys[i]);
    if (it == right_rows.end()) {
      continue;
    }
    for (std::size_t j : it->second) {
      left_index.push_back(i);
      right_index.push_back(j);
    }
  }
  Frame merged = left.take(left_index);
  const Frame right_rows_taken = right.take(right_index);
  for (const auto& name : right.names()) {
    if (name != "iteration") {
      merged.set(name, right_rows_taken.column(name));
    }
  }
  return merged;
}

Frame build_wide(const std::array<Frame, 3>& aligned) {
  std::vector<std::string> metric_cols{"waist_um",       "waist_norm",         "peak_I_proxy",
                                       "peak_I_norm",    "energy_proxy",       "energy_norm",
                                       "Ez_wake_absmax", "Ez_wake_absmax_GVm", "front_margin_um"};
  for (const char* optional : {"Eperp_peak_Vm", "a0_peak", "a0_norm"}) {
    if (std::all_of(aligned.begin(), aligned.end(), [&](const Frame& f) { return f.has(optional); })) {
      metric_cols.emplace_back(optional);
    }
  }

  const std::vector<std::string> base_cols{"iteration", "propagation_mm"};
  Frame wide = aligned[0].select(base_cols);
  for (std::size_t k = 0; k < 3; ++k) {
    std::vector<std::string> cols{"iteration"};
    cols.insert(cols.end(), metric_cols.begin(), metric_cols.end());
    Frame d = aligned[k].select(cols);
    for (const auto& col : metric_cols) {
      d.rename(col, col + "_" + kCaseOrder[k]);
    }
    wide = merge_on_iteration(wide, d);
  }

  const std::array<std::pair<std::string, std::string>, 3> pairs{
      {{"channel", "vacuum"}, {"uniform", "vacuum"}, {"channel", "uniform"}}};
  for (const auto& [a, b] : pairs) {
    wide.set(fmt::format("waist_{}_over_{}", a, b), divide(wide.doubles("waist_um_" + a), wide.doubles("waist_um_" + b)));
    wide.set(fmt::format("peakI_{}_over_{}", a, b),
             divide(wide.doubles("peak_I_proxy_" + a), wide.doubles("peak_I_proxy_" + b)));
    wide.set(fmt::format("energy_{}_over_{}", a, b),
             divide(wide.doubles("energy_proxy_" + a), wide.doubles("energy_proxy_" + b)));
    wide.set(fmt::format("Ezabs_{}_over_{}", a, b),
             divide(wide.doubles("Ez_wake_absmax_" + a), wide.doubles("Ez_wake_absmax_" + b)));
    if (wide.has("a0_peak_" + a) && wide.has("a0_peak_" + b)) {
      wide.set(fmt::format("a0_{}_over_{}", a, b), divide(wide.doubles("a0_peak_" + a), wide.doubles("a0_peak_" + b)));
    }
  }
  return wide;
}

std::vector<std::size_t> late_rows(const Frame& wide, double late_fraction) {
  if (!(0.0 < late_fraction && late_fraction <= 1.0)) {
    throw std::invalid_argument("late_fraction must be in (0, 1]");
  }
  const std::size_t n = wide.row_count();
  const auto n_late = static_cast<std::size_t>(
      std::max<double>(1.0, std::ceil(late_fraction * static_cast<double>(n))));
  std::vector<std::size_t> rows;
  for (std::size_t i = n_late >= n ? 0 : n - n_late; i < n; ++i) {
    rows.push_back(i);
  }
  return rows;
}

// triplet.nanmedian: NaN when no value is finite, else np.nanmedian.
double late_nanmedian(const Frame& late, const std::string& column) {
  const auto values = late.doubles(column);
  if (std::none_of(values.begin(), values.end(), [](double v) { return std::isfinite(v); })) {
    return kNaN;
  }
  return np::nanmedian(values);
}

Frame build_late_summary(const Frame& wide, double late_fraction) {
  const Frame late = wide.take(late_rows(wide, late_fraction));
  const auto propagation = wide.doubles("propagation_mm");
  const auto late_propagation = late.doubles("propagation_mm");

  std::vector<Record> rows;
  for (const auto& c : kCaseOrder) {
    Record row;
    row.set("case_type", c);
    row.set("n_common_dumps", static_cast<std::int64_t>(wide.row_count()));
    row.set("n_late_dumps", static_cast<std::int64_t>(late.row_count()));
    row.set("late_fraction", late_fraction);
    row.set("propagation_start_mm", propagation.front());
    row.set("propagation_end_mm", propagation.back());
    row.set("late_start_mm", late_propagation.front());
    row.set("late_end_mm", late_propagation.back());
    row.set("waist_late_median_um", late_nanmedian(late, "waist_um_" + c));
    row.set("waist_norm_late_median", late_nanmedian(late, "waist_norm_" + c));
    row.set("peakI_norm_late_median", late_nanmedian(late, "peak_I_norm_" + c));
    row.set("energy_norm_late_median", late_nanmedian(late, "energy_norm_" + c));
    row.set("Ezabs_late_median_GVm", late_nanmedian(late, "Ez_wake_absmax_GVm_" + c));
    row.set("front_margin_late_median_um", late_nanmedian(late, "front_margin_um_" + c));
    if (late.has("a0_peak_" + c)) {
      row.set("a0_peak_late_median", late_nanmedian(late, "a0_peak_" + c));
    }
    if (late.has("a0_norm_" + c)) {
      row.set("a0_norm_late_median", late_nanmedian(late, "a0_norm_" + c));
    }
    rows.push_back(std::move(row));
  }
  return table::frame_from_records(rows);
}

Frame build_late_ratios(const Frame& wide, double late_fraction) {
  const Frame late = wide.take(late_rows(wide, late_fraction));
  Record row;
  row.set("late_fraction", late_fraction);
  row.set("n_late_dumps", static_cast<std::int64_t>(late.row_count()));
  for (const char* column : {"waist_channel_over_vacuum", "waist_uniform_over_vacuum", "waist_channel_over_uniform",
                             "peakI_channel_over_vacuum", "peakI_uniform_over_vacuum", "peakI_channel_over_uniform",
                             "energy_channel_over_vacuum", "energy_uniform_over_vacuum",
                             "energy_channel_over_uniform", "Ezabs_channel_over_vacuum", "Ezabs_uniform_over_vacuum",
                             "Ezabs_channel_over_uniform"}) {
    row.set(std::string(column) + "_late_median", late_nanmedian(late, column));
  }
  for (const char* column : {"a0_channel_over_vacuum", "a0_uniform_over_vacuum", "a0_channel_over_uniform"}) {
    if (late.has(column)) {
      row.set(std::string(column) + "_late_median", late_nanmedian(late, column));
    }
  }
  const double peak_gain = std::get<double>(*row.get("peakI_channel_over_uniform_late_median"));
  const double waist_ratio = std::get<double>(*row.get("waist_channel_over_uniform_late_median"));
  row.set("optical_guiding_peakI_gain_ch_over_uni", peak_gain);
  row.set("optical_guiding_waist_reduction_ch_over_uni", 1.0 - waist_ratio);
  const std::vector<Record> rows{row};
  return table::frame_from_records(rows);
}

void add_plateau_columns(Frame& frame, const std::pair<double, double>& window) {
  frame.set("plateau_start_mm", Frame::Doubles(frame.row_count(), window.first));
  frame.set("plateau_end_mm", Frame::Doubles(frame.row_count(), window.second));
}

}  // namespace

const std::vector<std::string>& triplet_required_columns() {
  static const std::vector<std::string> columns{"iteration",      "propagation_mm", "waist_um",       "peak_I_proxy",
                                                "energy_proxy",   "Ez_wake_absmax", "front_margin_um"};
  return columns;
}

TripletTables build_triplet_tables(const fs::path& channel_csv, const fs::path& uniform_csv,
                                   const fs::path& vacuum_csv, const std::string& label, double late_fraction) {
  const std::array<Frame, 3> cases{add_case_normalizations(read_case_csv(channel_csv, "channel")),
                                   add_case_normalizations(read_case_csv(uniform_csv, "uniform")),
                                   add_case_normalizations(read_case_csv(vacuum_csv, "vacuum"))};
  const auto aligned = align_common_iterations(cases);

  std::vector<std::string> sources;
  for (const auto& frame : aligned) {
    if (frame.row_count() > 0) {
      sources.push_back(frame.strings("source_csv").front());
    }
  }
  const auto plateau_window = campaign::infer_plateau_window_mm_from_sources(sources);

  TripletTables tables;
  tables.long_table = concat_frames(aligned);
  tables.wide = build_wide(aligned);
  if (plateau_window) {
    add_plateau_columns(tables.wide, *plateau_window);
  }
  tables.late_summary = build_late_summary(tables.wide, late_fraction);
  tables.late_ratios = build_late_ratios(tables.wide, late_fraction);
  if (plateau_window) {
    add_plateau_columns(tables.late_summary, *plateau_window);
    add_plateau_columns(tables.late_ratios, *plateau_window);
  }
  for (Frame* frame : {&tables.long_table, &tables.wide, &tables.late_summary, &tables.late_ratios}) {
    frame->insert(0, "triplet", Frame::Strings(frame->row_count(), label));
  }
  return tables;
}

TripletTablePaths write_triplet_tables(const TripletTables& tables, const fs::path& outdir) {
  fs::create_directories(outdir);
  const TripletTablePaths paths{outdir / "guiding_triplet_long.csv", outdir / "guiding_triplet_wide.csv",
                                outdir / "guiding_triplet_late_summary.csv",
                                outdir / "guiding_triplet_late_ratios.csv"};
  table::write_file_atomically(paths.long_table, tables.long_table.to_pandas_csv());
  table::write_file_atomically(paths.wide, tables.wide.to_pandas_csv());
  table::write_file_atomically(paths.late_summary, tables.late_summary.to_pandas_csv());
  table::write_file_atomically(paths.late_ratios, tables.late_ratios.to_pandas_csv());
  return paths;
}

}  // namespace guiding::products

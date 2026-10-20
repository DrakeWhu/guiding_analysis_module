#include "guiding/products/scoring.hpp"

#include <algorithm>
#include <limits>
#include <numeric>
#include <stdexcept>

#include <fmt/format.h>

#include "guiding/campaign/case_metadata.hpp"
#include "guiding/numeric/npcompat.hpp"
#include "guiding/table/csv.hpp"
#include "guiding/table/py_format.hpp"

namespace guiding::products {
namespace {

namespace fs = std::filesystem;
using table::Cell;
using table::Record;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
const std::vector<std::string> kRequiredScoreColumns{"iteration", "propagation_mm", "a0_peak", "waist_um"};

// Python min(max(value, low), high) with NaN passed through.
double clip(double value, double low, double high) {
  if (!std::isfinite(value)) {
    return kNaN;
  }
  const double lower = low > value ? low : value;
  return high < lower ? high : lower;
}

double exp_score(double value) { return std::isfinite(value) ? std::exp(-(value * value)) : kNaN; }

double finite_median(std::span<const double> values) {
  std::vector<double> finite;
  for (double v : values) {
    if (std::isfinite(v)) {
      finite.push_back(v);
    }
  }
  return finite.empty() ? kNaN : np::median(finite);
}

double safe_ratio(double numerator, double denominator) {
  if (!std::isfinite(numerator) || !std::isfinite(denominator) || denominator == 0.0) {
    return kNaN;
  }
  return numerator / denominator;
}

double positive_log_ratio(double numerator, double denominator) {
  if (!std::isfinite(numerator) || !std::isfinite(denominator) || numerator <= 0.0 || denominator <= 0.0) {
    return kNaN;
  }
  return std::log(numerator / denominator);
}

double signed_deadband(double delta, double deadband) {
  if (!std::isfinite(delta)) {
    return kNaN;
  }
  const double magnitude = std::abs(delta);
  if (magnitude <= deadband) {
    return 0.0;
  }
  // np.sign(delta) * (abs - deadband)
  return (delta > 0.0 ? 1.0 : -1.0) * (magnitude - deadband);
}

Cell token_cell(const campaign::Token& token) { return token ? Cell{*token} : Cell{}; }

// record.get(key, nan) as float
double record_float(const Record& record, std::string_view key) {
  const auto cell = record.get(key);
  if (!cell) {
    return kNaN;
  }
  return py_float(*cell).value_or(kNaN);
}

std::string record_string(const Record& record, std::string_view key) {
  const auto cell = record.get(key);
  if (!cell) {
    return "None";
  }
  if (const auto* text = std::get_if<std::string>(&*cell)) {
    return *text;
  }
  return std::holds_alternative<std::monostate>(*cell) ? "None" : table::py_float_repr(py_float(*cell).value_or(kNaN));
}

struct WindowRows {
  std::vector<double> iteration;
  std::vector<double> a0;
  std::vector<double> waist;
};

// scoring._read_reference_window_df: rows inside the window with finite
// positive a0/waist, sorted by iteration.
WindowRows read_reference_window(const fs::path& csv_path, double start_mm, double end_mm) {
  const auto table = table::read_csv_file(csv_path);
  std::vector<std::string> missing;
  for (const char* column : {"iteration", "propagation_mm", "a0_peak", "waist_um"}) {
    if (!table.has_column(column)) {
      missing.emplace_back(column);
    }
  }
  if (!missing.empty()) {
    throw std::invalid_argument(fmt::format("missing columns in {}: {}", table::python_path_string(csv_path),
                                            table::python_list_repr(missing)));
  }
  const auto iteration = table.numeric_column("iteration");
  const auto propagation = table.numeric_column("propagation_mm");
  const auto a0 = table.numeric_column("a0_peak");
  const auto waist = table.numeric_column("waist_um");
  std::vector<std::size_t> kept;
  for (std::size_t i = 0; i < iteration.size(); ++i) {
    if (propagation[i] >= start_mm && propagation[i] <= end_mm && !std::isnan(iteration[i]) && !std::isnan(a0[i]) &&
        !std::isnan(waist[i]) && a0[i] > 0.0 && waist[i] > 0.0) {
      kept.push_back(i);
    }
  }
  std::stable_sort(kept.begin(), kept.end(), [&](std::size_t a, std::size_t b) { return iteration[a] < iteration[b]; });
  WindowRows rows;
  for (std::size_t i : kept) {
    rows.iteration.push_back(iteration[i]);
    rows.a0.push_back(a0[i]);
    rows.waist.push_back(waist[i]);
  }
  return rows;
}

struct ReferenceAdvantage {
  double a0_log = kNaN;
  double waist_log = kNaN;
  std::int64_t n_rows = 0;
  double fraction_a0 = kNaN;
  double fraction_waist = kNaN;
};

ReferenceAdvantage rowwise_exit_reference_advantage(const fs::path& channel_csv, const fs::path& uniform_csv,
                                                    const fs::path& vacuum_csv, double exit_start_mm,
                                                    double exit_end_mm) {
  const auto channel = read_reference_window(channel_csv, exit_start_mm, exit_end_mm);
  const auto uniform = read_reference_window(uniform_csv, exit_start_mm, exit_end_mm);
  const auto vacuum = read_reference_window(vacuum_csv, exit_start_mm, exit_end_mm);

  const auto index_of = [](const WindowRows& rows, double iteration) -> std::optional<std::size_t> {
    const auto it = std::lower_bound(rows.iteration.begin(), rows.iteration.end(), iteration);
    if (it == rows.iteration.end() || *it != iteration) {
      return std::nullopt;
    }
    return static_cast<std::size_t>(it - rows.iteration.begin());
  };

  std::vector<double> a0_log;
  std::vector<double> waist_log;
  bool any_common = false;
  double previous = kNaN;
  for (std::size_t i = 0; i < channel.iteration.size(); ++i) {
    const double iteration = channel.iteration[i];
    if (iteration == previous) {
      continue;  // Index.intersection is unique
    }
    previous = iteration;
    const auto u = index_of(uniform, iteration);
    const auto v = index_of(vacuum, iteration);
    if (!u || !v) {
      continue;
    }
    any_common = true;
    const bool use_uniform = uniform.a0[*u] >= vacuum.a0[*v];
    const double ref_a0 = use_uniform ? uniform.a0[*u] : vacuum.a0[*v];
    const double ref_waist = use_uniform ? uniform.waist[*u] : vacuum.waist[*v];
    const double a = std::log(channel.a0[i] / ref_a0);
    const double w = std::log(ref_waist / channel.waist[i]);
    if (std::isfinite(a) && std::isfinite(w)) {
      a0_log.push_back(a);
      waist_log.push_back(w);
    }
  }
  if (!any_common) {
    throw std::invalid_argument("no common iterations in triplet exit window");
  }
  if (a0_log.empty()) {
    throw std::invalid_argument("no finite row-wise reference advantages");
  }
  ReferenceAdvantage advantage;
  advantage.a0_log = np::median(a0_log);
  advantage.waist_log = np::median(waist_log);
  advantage.n_rows = static_cast<std::int64_t>(a0_log.size());
  const auto positive_fraction = [](const std::vector<double>& values) {
    const auto count = std::count_if(values.begin(), values.end(), [](double v) { return v > 0.0; });
    return static_cast<double>(count) / static_cast<double>(values.size());
  };
  advantage.fraction_a0 = positive_fraction(a0_log);
  advantage.fraction_waist = positive_fraction(waist_log);
  return advantage;
}

ReferenceAdvantage median_exit_reference_advantage(const Record& channel, const Record& uniform,
                                                   const Record& vacuum) {
  const double a0_channel = record_float(channel, "a0_exit");
  const double a0_uniform = record_float(uniform, "a0_exit");
  const double a0_vacuum = record_float(vacuum, "a0_exit");
  const double waist_channel = record_float(channel, "waist_exit_um");
  const double waist_uniform = record_float(uniform, "waist_exit_um");
  const double waist_vacuum = record_float(vacuum, "waist_exit_um");
  const bool use_uniform = a0_uniform >= a0_vacuum;
  const double a0_reference = use_uniform ? a0_uniform : a0_vacuum;
  const double waist_reference = use_uniform ? waist_uniform : waist_vacuum;
  const double a0_log = positive_log_ratio(a0_channel, a0_reference);
  const double waist_log = positive_log_ratio(waist_reference, waist_channel);
  if (!std::isfinite(a0_log) || !std::isfinite(waist_log)) {
    throw std::invalid_argument("non_finite_median_reference_advantage");
  }
  return {a0_log, waist_log, 0, a0_log > 0.0 ? 1.0 : 0.0, waist_log > 0.0 ? 1.0 : 0.0};
}

}  // namespace

std::optional<double> py_float(const Cell& cell) {
  if (const auto* real = std::get_if<double>(&cell)) {
    return *real;
  }
  if (const auto* integer = std::get_if<std::int64_t>(&cell)) {
    return static_cast<double>(*integer);
  }
  if (const auto* flag = std::get_if<bool>(&cell)) {
    return *flag ? 1.0 : 0.0;
  }
  if (const auto* text = std::get_if<std::string>(&cell)) {
    return table::parse_py_float(*text);
  }
  return std::nullopt;
}

Record score_case_csv(const fs::path& path, const std::optional<std::string>& case_id, const ScoreConfig& cfg) {
  std::string id;
  if (case_id && !case_id->empty()) {
    id = *case_id;
  } else {
    const std::string parent = table::python_path_string(path.parent_path());
    id = parent == "." ? std::string() : fs::path(parent).filename().string();
  }
  Record base{{"case_id", id},
              {"csv_path", table::python_path_string(path)},
              {"status", std::string("failed")},
              {"failure_reason", std::string()},
              {"score", kNaN}};
  const auto failure = [&](const std::string& reason, const Record& extra = {}) {
    Record out = base;
    out.set("failure_reason", reason);
    out.merge(extra);
    return out;
  };

  std::error_code error;
  if (!fs::is_regular_file(path, error)) {
    return failure("missing_csv");
  }
  table::CsvTable table;
  try {
    table = table::read_csv_file(path);
  } catch (const std::exception& exception) {
    return failure(fmt::format("read_csv_failed: {}", exception.what()));
  }
  if (table.columns.empty()) {
    return failure("read_csv_failed: No columns to parse from file");
  }
  std::vector<std::string> missing;
  for (const auto& column : kRequiredScoreColumns) {
    if (!table.has_column(column)) {
      missing.push_back(column);
    }
  }
  if (!missing.empty()) {
    return failure(fmt::format("missing_columns: {}", table::python_list_repr(missing)));
  }
  if (table.rows.empty()) {
    return failure("empty_csv");
  }

  const auto z = table.numeric_column("propagation_mm");
  const auto a0 = table.numeric_column("a0_peak");
  const auto waist = table.numeric_column("waist_um");

  const auto window = campaign::infer_plateau_window_mm_from_text(table::python_path_string(path));
  if (!window) {
    return failure("could_not_infer_plateau_window");
  }
  const auto [plateau_start_mm, plateau_end_mm] = *window;
  const double entry_start_mm = plateau_start_mm;
  const double entry_end_mm = plateau_start_mm + cfg.entry_window_mm;
  const double exit_start_mm = plateau_end_mm - cfg.exit_before_mm;
  const double exit_end_mm = plateau_end_mm + cfg.exit_after_mm;
  const double analysis_start_mm = plateau_start_mm;
  const double analysis_end_mm = exit_end_mm;

  const auto rows_in = [&](double start, double end) {
    std::vector<std::size_t> rows;
    for (std::size_t i = 0; i < z.size(); ++i) {
      if (z[i] >= start && z[i] <= end) {
        rows.push_back(i);
      }
    }
    return rows;
  };
  const auto gather = [](const std::vector<double>& column, const std::vector<std::size_t>& rows) {
    std::vector<double> out;
    out.reserve(rows.size());
    for (std::size_t i : rows) {
      out.push_back(column[i]);
    }
    return out;
  };
  const auto entry = rows_in(entry_start_mm, entry_end_mm);
  const auto exit = rows_in(exit_start_mm, exit_end_mm);
  const auto analysis = rows_in(analysis_start_mm, analysis_end_mm);
  if (entry.empty()) {
    return failure("empty_entry_window");
  }
  if (exit.empty()) {
    return failure("empty_exit_window");
  }
  if (analysis.empty()) {
    return failure("empty_analysis_window");
  }

  const double a0_exit = finite_median(gather(a0, exit));
  const double waist_entry_um = finite_median(gather(waist, entry));
  const double waist_exit_um = finite_median(gather(waist, exit));

  // _finite_max: nanargmax unless every value is non-finite
  const auto analysis_a0 = gather(a0, analysis);
  double a0_max = kNaN;
  double a0_max_mm = kNaN;
  if (std::any_of(analysis_a0.begin(), analysis_a0.end(), [](double v) { return std::isfinite(v); })) {
    std::size_t best = analysis_a0.size();
    for (std::size_t i = 0; i < analysis_a0.size(); ++i) {
      if (!std::isnan(analysis_a0[i]) && (best == analysis_a0.size() || analysis_a0[i] > analysis_a0[best])) {
        best = i;
      }
    }
    a0_max = analysis_a0[best];
    a0_max_mm = z[analysis[best]];
  }

  std::vector<double> log_waist;
  for (std::size_t i : analysis) {
    if (std::isfinite(waist[i]) && waist[i] > 0.0) {
      log_waist.push_back(std::log(waist[i]));
    }
  }
  const double waist_jitter_log = log_waist.size() >= 2 ? np::std_dev(log_waist) : kNaN;

  std::size_t valid = 0;
  for (std::size_t i : analysis) {
    valid += std::isfinite(a0[i]) && std::isfinite(waist[i]) ? 1 : 0;
  }
  const double valid_fraction = static_cast<double>(valid) / static_cast<double>(analysis.size());
  const double nan_fraction = 1.0 - valid_fraction;

  for (double value : {a0_exit, waist_entry_um, waist_exit_um, a0_max, waist_jitter_log}) {
    if (!std::isfinite(value)) {
      return failure("non_finite_score_metric", Record{{"plateau_start_mm", plateau_start_mm},
                                                       {"plateau_end_mm", plateau_end_mm},
                                                       {"a0_exit", a0_exit},
                                                       {"waist_entry_um", waist_entry_um},
                                                       {"waist_exit_um", waist_exit_um},
                                                       {"a0_max_analysis", a0_max},
                                                       {"waist_jitter_log", waist_jitter_log},
                                                       {"valid_fraction", valid_fraction},
                                                       {"nan_fraction", nan_fraction}});
    }
  }
  if (waist_entry_um <= 0.0 || a0_max <= 0.0) {
    return failure("non_positive_reference_metric",
                   Record{{"waist_entry_um", waist_entry_um}, {"a0_max_analysis", a0_max}});
  }

  const double waist_growth = waist_exit_um / waist_entry_um;
  const double a0_exit_over_analysis_max = a0_exit / a0_max;
  const double a0_exit_component = clip(a0_exit / cfg.a0_target, 0.0, cfg.a0_component_cap);
  const double a0_retention_component = clip(a0_exit_over_analysis_max, 0.0, 1.0);
  const double growth_excess = waist_growth - 1.0 > 0.0 ? waist_growth - 1.0 : 0.0;  // max(0.0, x)
  const double waist_growth_component = exp_score(growth_excess / cfg.waist_growth_sigma);
  const double waist_stability_component = exp_score(waist_jitter_log / cfg.waist_jitter_sigma);
  const double weighted = cfg.weight_a0_exit * a0_exit_component + cfg.weight_a0_retention * a0_retention_component +
                          cfg.weight_waist_growth * waist_growth_component +
                          cfg.weight_waist_stability * waist_stability_component;
  const double score = 100.0 * weighted * valid_fraction;

  Record out = base;
  out.set("status", std::string("ok"));
  out.set("failure_reason", std::string());
  out.set("score", score);
  out.merge(Record{{"plateau_start_mm", plateau_start_mm},
                   {"plateau_end_mm", plateau_end_mm},
                   {"entry_start_mm", entry_start_mm},
                   {"entry_end_mm", entry_end_mm},
                   {"exit_start_mm", exit_start_mm},
                   {"exit_end_mm", exit_end_mm},
                   {"analysis_start_mm", analysis_start_mm},
                   {"analysis_end_mm", analysis_end_mm},
                   {"n_rows", static_cast<std::int64_t>(table.rows.size())},
                   {"n_entry_rows", static_cast<std::int64_t>(entry.size())},
                   {"n_exit_rows", static_cast<std::int64_t>(exit.size())},
                   {"n_analysis_rows", static_cast<std::int64_t>(analysis.size())},
                   {"valid_fraction", valid_fraction},
                   {"nan_fraction", nan_fraction},
                   {"a0_exit", a0_exit},
                   {"a0_max_analysis", a0_max},
                   {"a0_max_analysis_mm", a0_max_mm},
                   {"a0_exit_over_analysis_max", a0_exit_over_analysis_max},
                   {"waist_entry_um", waist_entry_um},
                   {"waist_exit_um", waist_exit_um},
                   {"waist_growth", waist_growth},
                   {"waist_jitter_log", waist_jitter_log},
                   {"component_a0_exit", a0_exit_component},
                   {"component_a0_retention", a0_retention_component},
                   {"component_waist_growth", waist_growth_component},
                   {"component_waist_stability", waist_stability_component},
                   {"weight_a0_exit", cfg.weight_a0_exit},
                   {"weight_a0_retention", cfg.weight_a0_retention},
                   {"weight_waist_growth", cfg.weight_waist_growth},
                   {"weight_waist_stability", cfg.weight_waist_stability}});
  return out;
}

std::pair<double, double> reference_factor_from_local_advantage(double a0_log_advantage, double waist_log_advantage,
                                                                double deadband_log, double scale_log,
                                                                double a0_weight, double waist_weight) {
  if (!std::isfinite(a0_log_advantage) || !std::isfinite(waist_log_advantage)) {
    return {kNaN, kNaN};
  }
  if (scale_log <= 0.0) {
    throw std::invalid_argument("scale_log must be positive");
  }
  double combined = a0_weight * a0_log_advantage + waist_weight * waist_log_advantage;
  if (a0_log_advantage < -deadband_log) {
    combined = a0_log_advantage < combined ? a0_log_advantage : combined;  // min(combined, a0)
  }
  const double effective = signed_deadband(combined, deadband_log);
  return {std::tanh(effective / scale_log), combined};
}

Record score_triplet_csvs(const fs::path& channel_csv, const fs::path& uniform_csv, const fs::path& vacuum_csv,
                          const std::optional<std::string>& channel_case_id,
                          const std::optional<std::string>& uniform_case_id,
                          const std::optional<std::string>& vacuum_case_id, const ScoreConfig& case_config,
                          const TripletScoreConfig& tcfg) {
  const Record channel = score_case_csv(channel_csv, channel_case_id, case_config);
  const Record uniform = score_case_csv(uniform_csv, uniform_case_id, case_config);
  const Record vacuum = score_case_csv(vacuum_csv, vacuum_case_id, case_config);

  Record base{{"status", std::string("failed")},
              {"failure_reason", std::string()},
              {"channel_case_id", *channel.get("case_id")},
              {"uniform_case_id", *uniform.get("case_id")},
              {"vacuum_case_id", *vacuum.get("case_id")},
              {"score_channel", *channel.get("score")},
              {"score_uniform", *uniform.get("score")},
              {"score_vacuum", *vacuum.get("score")},
              {"final_score", kNaN}};

  std::vector<std::string> failed;
  for (const auto& [name, result] : {std::pair<const char*, const Record*>{"channel", &channel},
                                     {"uniform", &uniform},
                                     {"vacuum", &vacuum}}) {
    if (record_string(*result, "status") != "ok") {
      failed.push_back(fmt::format("{}: {}", name, record_string(*result, "failure_reason")));
    }
  }
  if (!failed.empty()) {
    std::string reason;
    for (const auto& item : failed) {
      reason += (reason.empty() ? "" : "; ") + item;
    }
    Record out = base;
    out.set("failure_reason", reason);
    return out;
  }

  const double score_channel = record_float(channel, "score");
  const double score_uniform = record_float(uniform, "score");
  const double score_vacuum = record_float(vacuum, "score");
  const double score_delta_vs_uniform = score_channel - score_uniform;
  const double score_delta_vs_vacuum = score_channel - score_vacuum;
  const bool score_uniform_reference = score_uniform >= score_vacuum;
  const std::string score_reference_kind = score_uniform_reference ? "uniform" : "vacuum";
  const double score_reference = score_uniform_reference ? score_uniform : score_vacuum;
  const double score_delta_vs_reference = score_channel - score_reference;

  const double a0_exit_channel = record_float(channel, "a0_exit");
  const double a0_exit_uniform = record_float(uniform, "a0_exit");
  const double a0_exit_vacuum = record_float(vacuum, "a0_exit");
  const double waist_exit_channel = record_float(channel, "waist_exit_um");
  const double waist_exit_uniform = record_float(uniform, "waist_exit_um");
  const double waist_exit_vacuum = record_float(vacuum, "waist_exit_um");
  const bool a0_uniform_reference = a0_exit_uniform >= a0_exit_vacuum;
  const std::string reference_kind = a0_uniform_reference ? "uniform" : "vacuum";
  const double a0_exit_reference = a0_uniform_reference ? a0_exit_uniform : a0_exit_vacuum;
  const double waist_exit_reference = a0_uniform_reference ? waist_exit_uniform : waist_exit_vacuum;

  std::string reference_factor_mode = "rowwise_exit_median";
  std::string rowwise_failure;
  ReferenceAdvantage advantage;
  try {
    advantage = rowwise_exit_reference_advantage(channel_csv, uniform_csv, vacuum_csv,
                                                 record_float(channel, "exit_start_mm"),
                                                 record_float(channel, "exit_end_mm"));
  } catch (const std::invalid_argument& rowwise_error) {
    rowwise_failure = rowwise_error.what();
    reference_factor_mode = "exit_median_fallback";
    try {
      advantage = median_exit_reference_advantage(channel, uniform, vacuum);
    } catch (const std::invalid_argument& fallback_error) {
      Record out = base;
      out.set("failure_reason", fmt::format("rowwise_reference_failed: {}; median_reference_failed: {}",
                                            rowwise_failure, fallback_error.what()));
      out.set("reference_factor_mode", std::string("failed"));
      out.set("rowwise_reference_failure_reason", rowwise_failure);
      return out;
    }
  }

  const auto [reference_factor, combined] = reference_factor_from_local_advantage(
      advantage.a0_log, advantage.waist_log, tcfg.reference_deadband_log, tcfg.reference_scale_log,
      tcfg.reference_a0_weight, tcfg.reference_waist_weight);
  const double final_score = score_channel * reference_factor;

  Record out = base;
  out.set("status", std::string("ok"));
  out.set("failure_reason", std::string());
  out.set("final_score", final_score);
  const auto get = [](const Record& record, const char* key) { return record.get(key).value_or(Cell{kNaN}); };
  out.merge(Record{
      {"reference_kind", reference_kind},
      {"score_reference_kind", score_reference_kind},
      {"score_reference", score_reference},
      {"score_delta_vs_reference", score_delta_vs_reference},
      {"score_delta_vs_uniform", score_delta_vs_uniform},
      {"score_delta_vs_vacuum", score_delta_vs_vacuum},
      {"reference_factor", reference_factor},
      {"a0_exit_reference", a0_exit_reference},
      {"waist_exit_reference_um", waist_exit_reference},
      {"a0_log_advantage_vs_reference", advantage.a0_log},
      {"waist_log_advantage_vs_reference", advantage.waist_log},
      {"combined_log_advantage", combined},
      {"reference_deadband_log", tcfg.reference_deadband_log},
      {"reference_scale_log", tcfg.reference_scale_log},
      {"reference_a0_weight", tcfg.reference_a0_weight},
      {"reference_waist_weight", tcfg.reference_waist_weight},
      {"waist_exit_channel_um", waist_exit_channel},
      {"waist_exit_uniform_um", waist_exit_uniform},
      {"waist_exit_vacuum_um", waist_exit_vacuum},
      {"reference_deadband", tcfg.reference_deadband},
      {"reference_scale", tcfg.reference_scale},
      {"final_score", final_score},
      {"a0_exit_channel", get(channel, "a0_exit")},
      {"a0_exit_uniform", get(uniform, "a0_exit")},
      {"a0_exit_vacuum", get(vacuum, "a0_exit")},
      {"a0_exit_channel_over_uniform", safe_ratio(a0_exit_channel, a0_exit_uniform)},
      {"a0_exit_channel_over_vacuum", safe_ratio(a0_exit_channel, a0_exit_vacuum)},
      {"waist_growth_channel", get(channel, "waist_growth")},
      {"waist_growth_uniform", get(uniform, "waist_growth")},
      {"waist_growth_vacuum", get(vacuum, "waist_growth")},
      {"waist_growth_channel_over_uniform",
       safe_ratio(record_float(channel, "waist_growth"), record_float(uniform, "waist_growth"))},
      {"waist_growth_channel_over_vacuum",
       safe_ratio(record_float(channel, "waist_growth"), record_float(vacuum, "waist_growth"))},
      {"a0_retention_channel", get(channel, "a0_exit_over_analysis_max")},
      {"a0_retention_uniform", get(uniform, "a0_exit_over_analysis_max")},
      {"a0_retention_vacuum", get(vacuum, "a0_exit_over_analysis_max")},
      {"valid_fraction_channel", get(channel, "valid_fraction")},
      {"valid_fraction_uniform", get(uniform, "valid_fraction")},
      {"valid_fraction_vacuum", get(vacuum, "valid_fraction")},
      {"reference_factor_mode", reference_factor_mode},
      {"rowwise_reference_failure_reason", rowwise_failure},
      {"n_reference_rows", advantage.n_rows},
      {"fraction_a0_beats_reference", advantage.fraction_a0},
      {"fraction_waist_beats_reference", advantage.fraction_waist},
  });
  return out;
}

table::Frame top_rows(const table::Frame& frame, const std::string& column, std::size_t top) {
  std::vector<std::size_t> rows;
  if (frame.has("status")) {
    const auto& status = frame.strings("status");
    for (std::size_t i = 0; i < status.size(); ++i) {
      if (status[i] == "ok") {
        rows.push_back(i);
      }
    }
  }
  const auto values = frame.has(column) ? frame.doubles(column) : std::vector<double>(frame.row_count(), kNaN);
  // sort_values(ascending=False): NaN last, equal values keep their order.
  std::stable_sort(rows.begin(), rows.end(), [&](std::size_t a, std::size_t b) {
    if (std::isnan(values[a]) || std::isnan(values[b])) {
      return !std::isnan(values[a]) && std::isnan(values[b]);
    }
    return values[a] > values[b];
  });
  if (rows.size() > top) {
    rows.resize(top);
  }
  table::Frame out = frame.take(rows);
  table::Frame::Integers rank(rows.size());
  std::iota(rank.begin(), rank.end(), std::int64_t{1});
  out.insert(0, "rank", std::move(rank));
  return out;
}

ScoreTables score_campaign_cases(const std::vector<campaign::CaseInfo>& cases, const fs::path& case_metrics_root,
                                 const std::string& case_type, const ScoreConfig& config, std::size_t top) {
  std::vector<Record> rows;
  for (const auto& info : cases) {
    if (case_type != "all" && case_type != campaign::case_type_name(info.case_type)) {
      continue;
    }
    Record row = score_case_csv(case_metrics_root / info.case_id / "guiding_metrics.csv", info.case_id, config);
    row.set("case_type", std::string(campaign::case_type_name(info.case_type)));
    row.set("laser_case", token_cell(info.tokens.laser_case));
    row.set("density", token_cell(info.tokens.density));
    row.set("ref_density", token_cell(info.tokens.ref_density));
    row.set("plateau", token_cell(info.tokens.plateau));
    row.set("focus", token_cell(info.tokens.focus));
    row.set("diameter", token_cell(info.tokens.diameter));
    rows.push_back(std::move(row));
  }
  ScoreTables tables;
  tables.scores = table::frame_from_records(rows);
  tables.top = top_rows(tables.scores, "score", top);
  for (const auto& row : rows) {
    tables.ok += record_string(row, "status") == "ok" ? 1 : 0;
  }
  return tables;
}

TripletScoreTables score_campaign_triplets(const std::vector<campaign::TripletInfo>& triplets,
                                           const fs::path& case_metrics_root, const ScoreConfig& case_config,
                                           const TripletScoreConfig& triplet_config, std::size_t top) {
  TripletScoreTables result;
  std::vector<Record> rows;
  for (const auto& triplet : triplets) {
    if (!triplet.complete()) {
      ++result.skipped_incomplete;
      continue;
    }
    const fs::path channel_csv = case_metrics_root / triplet.channel->case_id / "guiding_metrics.csv";
    const fs::path uniform_csv = case_metrics_root / triplet.uniform->case_id / "guiding_metrics.csv";
    const fs::path vacuum_csv = case_metrics_root / triplet.vacuum->case_id / "guiding_metrics.csv";
    Record row;
    try {
      row = score_triplet_csvs(channel_csv, uniform_csv, vacuum_csv, triplet.channel->case_id,
                               triplet.uniform->case_id, triplet.vacuum->case_id, case_config, triplet_config);
    } catch (const std::exception& error) {
      row = Record{{"status", std::string("failed")},
                   {"failure_reason", fmt::format("score_triplet_exception: {}", error.what())},
                   {"channel_case_id", triplet.channel->case_id},
                   {"uniform_case_id", triplet.uniform->case_id},
                   {"vacuum_case_id", triplet.vacuum->case_id},
                   {"score_channel", kNaN},
                   {"score_uniform", kNaN},
                   {"score_vacuum", kNaN},
                   {"final_score", kNaN}};
    }
    const auto& tokens = triplet.channel->tokens;
    row.set("triplet_label", triplet.label());
    row.set("laser_case", token_cell(tokens.laser_case));
    row.set("density", token_cell(tokens.density));
    row.set("plateau", token_cell(tokens.plateau));
    row.set("focus", token_cell(tokens.focus));
    row.set("diameter", token_cell(tokens.diameter));
    row.set("channel_csv", table::python_path_string(channel_csv));
    row.set("uniform_csv", table::python_path_string(uniform_csv));
    row.set("vacuum_csv", table::python_path_string(vacuum_csv));
    rows.push_back(std::move(row));
  }
  result.tables.scores = table::frame_from_records(rows);
  result.tables.top = top_rows(result.tables.scores, "final_score", top);
  for (const auto& row : rows) {
    result.tables.ok += record_string(row, "status") == "ok" ? 1 : 0;
  }
  return result;
}

}  // namespace guiding::products

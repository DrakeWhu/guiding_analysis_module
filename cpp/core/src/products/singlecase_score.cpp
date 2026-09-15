#include "guiding/products/singlecase_score.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numeric>
#include <vector>

#include <fmt/format.h>

#include "guiding/campaign/case_metadata.hpp"
#include "guiding/numeric/npcompat.hpp"

namespace guiding::products {
namespace {

using table::Cell;
using table::Record;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr std::array<const char*, 4> kRequiredColumns = {"iteration", "propagation_mm", "a0_peak", "waist_um"};

Record base_result(const std::string& case_id, const std::optional<std::string>& csv_path,
                   const SingleCaseGuidingConfig& config) {
  Record record;
  record.set("case_id", std::string(case_id));
  record.set("csv_path", csv_path.value_or(std::string()));
  record.set("metric_guiding_singlecase_schema_version", config.schema_version);
  record.set("metric_guiding_singlecase_config_id", config.config_id);
  record.set("metric_guiding_singlecase_status", std::string("failed"));
  record.set("metric_guiding_singlecase_failure_reason", std::string());
  record.set("metric_guiding_singlecase_score_v1", kNaN);
  return record;
}

Record failure(Record base, const Record& extra, const std::string& reason) {
  base.merge(extra);
  base.set("metric_guiding_singlecase_failure_reason", reason);
  return base;
}

std::string python_list_repr(const std::vector<std::string>& items) {
  std::string out = "[";
  for (std::size_t i = 0; i < items.size(); ++i) {
    out += fmt::format("{}'{}'", i == 0 ? "" : ", ", items[i]);
  }
  return out + "]";
}

double finite_median(const std::vector<double>& values) {
  std::vector<double> finite;
  for (double v : values) {
    if (std::isfinite(v)) {
      finite.push_back(v);
    }
  }
  return finite.empty() ? kNaN : np::median(finite);
}

double clip01(double value) {
  if (!std::isfinite(value)) {
    return kNaN;
  }
  const double lower = (0.0 > value) ? 0.0 : value;  // Python max(value, 0.0)
  return (1.0 < lower) ? 1.0 : lower;                // Python min(..., 1.0)
}

double exp_quadratic_component(double value, double sigma) {
  if (!std::isfinite(value) || !std::isfinite(sigma) || sigma <= 0.0) {
    return kNaN;
  }
  return std::exp(-std::pow(value / sigma, 2.0));
}

double positive_log_ratio(double numerator, double denominator) {
  if (!std::isfinite(numerator) || !std::isfinite(denominator)) {
    return kNaN;
  }
  if (numerator <= 0.0 || denominator <= 0.0) {
    return kNaN;
  }
  return std::log(numerator / denominator);
}

struct PlateauWindow {
  double start;
  double end;
  std::string policy;
};

std::optional<PlateauWindow> resolve_plateau_window(const std::optional<std::string>& csv_path,
                                                    const PlateauOverride& plateau,
                                                    const SingleCaseGuidingConfig& config) {
  if (plateau.start_mm && plateau.end_mm) {
    return PlateauWindow{*plateau.start_mm, *plateau.end_mm, "explicit_start_end"};
  }
  if (plateau.length_mm) {
    const double start = plateau.start_mm.value_or(config.plateau_start_default_mm);
    return PlateauWindow{start, start + *plateau.length_mm, "explicit_length"};
  }
  if (csv_path) {
    if (const auto inferred = campaign::infer_plateau_window_mm_from_text(*csv_path)) {
      return PlateauWindow{inferred->first, inferred->second, "path_inference"};
    }
  }
  return std::nullopt;
}

std::vector<double> gather(const std::vector<double>& column, const std::vector<std::size_t>& rows) {
  std::vector<double> out;
  out.reserve(rows.size());
  for (std::size_t row : rows) {
    out.push_back(column[row]);
  }
  return out;
}

}  // namespace

Record score_singlecase_guiding_table(const table::CsvTable& table, const std::string& case_id,
                                      const std::optional<std::string>& csv_path, const PlateauOverride& plateau,
                                      const SingleCaseGuidingConfig& config) {
  const Record base = base_result(case_id, csv_path, config);

  std::vector<std::string> missing;
  for (const char* column : kRequiredColumns) {
    if (!table.has_column(column)) {
      missing.emplace_back(column);
    }
  }
  if (!missing.empty()) {
    return failure(base, {}, fmt::format("missing_columns: {}", python_list_repr(missing)));
  }
  if (table.rows.empty()) {
    return failure(base, {}, "empty_csv");
  }

  const auto window = resolve_plateau_window(csv_path, plateau, config);
  if (!window) {
    return failure(base, {}, "could_not_determine_plateau_window");
  }
  const double plateau_start = window->start;
  const double plateau_end = window->end;
  const Record window_keys{{"metric_guiding_plateau_start_mm", plateau_start},
                           {"metric_guiding_plateau_end_mm", plateau_end},
                           {"metric_guiding_plateau_policy", window->policy}};
  if (!std::isfinite(plateau_start) || !std::isfinite(plateau_end)) {
    return failure(base, window_keys, "non_finite_plateau_window");
  }
  if (plateau_end <= plateau_start) {
    return failure(base, window_keys, "invalid_plateau_window");
  }

  const auto propagation = table.numeric_column("propagation_mm");
  const auto a0 = table.numeric_column("a0_peak");
  const auto waist = table.numeric_column("waist_um");

  // work.sort_values("propagation_mm"): ascending, NaN last.
  std::vector<std::size_t> order(table.rows.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    const bool a_nan = std::isnan(propagation[a]);
    const bool b_nan = std::isnan(propagation[b]);
    if (a_nan != b_nan) {
      return b_nan;
    }
    return !a_nan && propagation[a] < propagation[b];
  });

  std::vector<std::size_t> plateau_rows;
  for (std::size_t row : order) {
    if (propagation[row] >= plateau_start && propagation[row] <= plateau_end) {
      plateau_rows.push_back(row);
    }
  }

  Record common = window_keys;
  common.set("metric_guiding_n_rows", static_cast<std::int64_t>(table.rows.size()));
  common.set("metric_guiding_n_plateau_rows", static_cast<std::int64_t>(plateau_rows.size()));
  if (plateau_rows.empty()) {
    return failure(base, common, "empty_plateau_window");
  }

  std::vector<std::size_t> valid;
  for (std::size_t row : plateau_rows) {
    if (std::isfinite(propagation[row]) && std::isfinite(a0[row]) && std::isfinite(waist[row]) && a0[row] > 0.0 &&
        waist[row] > 0.0) {
      valid.push_back(row);
    }
  }
  const double valid_fraction =
      plateau_rows.empty() ? 0.0 : static_cast<double>(valid.size()) / static_cast<double>(plateau_rows.size());
  common.set("metric_guiding_n_valid_plateau_rows", static_cast<std::int64_t>(valid.size()));
  common.set("metric_guiding_valid_fraction", valid_fraction);
  if (static_cast<int>(valid.size()) < config.min_valid_plateau_rows) {
    return failure(base, common, "not_enough_valid_plateau_rows");
  }

  const auto z_valid = gather(propagation, valid);
  const double span_fraction = clip01((np::max_value<double>(z_valid) - np::min_value<double>(z_valid)) /
                                      (plateau_end - plateau_start));
  const double coverage_component = span_fraction * valid_fraction;

  auto window_rows = [&](double start, double end, bool entry) {
    std::vector<std::size_t> rows;
    for (std::size_t row : valid) {
      if (propagation[row] >= start && propagation[row] <= end) {
        rows.push_back(row);
      }
    }
    if (!rows.empty()) {
      return std::pair{rows, std::string("window")};
    }
    return entry ? std::pair{std::vector<std::size_t>{valid.front()}, std::string("first_valid_fallback")}
                 : std::pair{std::vector<std::size_t>{valid.back()}, std::string("last_valid_fallback")};
  };
  const auto [entry_rows, entry_policy] =
      window_rows(plateau_start, plateau_start + config.entry_window_mm, true);
  const auto [exit_rows, exit_policy] = window_rows(plateau_end - config.exit_window_mm, plateau_end, false);

  const double a0_entry = finite_median(gather(a0, entry_rows));
  const double a0_exit = finite_median(gather(a0, exit_rows));
  const auto a0_values = gather(a0, valid);
  const double a0_max = np::max_value<double>(a0_values);
  const double waist_entry = finite_median(gather(waist, entry_rows));
  const double waist_exit = finite_median(gather(waist, exit_rows));
  const auto waist_values = gather(waist, valid);
  const double waist_max = np::max_value<double>(waist_values);

  const std::array<double, 6> required_positive{a0_entry, a0_exit, a0_max, waist_entry, waist_exit, waist_max};
  if (!std::all_of(required_positive.begin(), required_positive.end(),
                   [](double v) { return std::isfinite(v) && v > 0.0; })) {
    Record extra = common;
    extra.set("metric_guiding_a0_entry", a0_entry);
    extra.set("metric_guiding_a0_exit", a0_exit);
    extra.set("metric_guiding_a0_max", a0_max);
    extra.set("metric_guiding_waist_entry_um", waist_entry);
    extra.set("metric_guiding_waist_exit_um", waist_exit);
    extra.set("metric_guiding_waist_max_um", waist_max);
    return failure(base, extra, "non_positive_reference_metric");
  }

  const double a0_retention_ratio = a0_exit / a0_entry;
  const double a0_retention_component = clip01(a0_retention_ratio);

  double a0_drop_rms_log = kNaN;
  if (a0_values.size() >= 2) {
    std::vector<double> squared_drops(a0_values.size() - 1);
    for (std::size_t k = 0; k + 1 < a0_values.size(); ++k) {
      const double drop = np::maximum(0.0, std::log(a0_values[k] / a0_values[k + 1]));
      squared_drops[k] = drop * drop;
    }
    a0_drop_rms_log = std::sqrt(np::mean<double>(squared_drops));
  }
  const double a0_stability_component = exp_quadratic_component(a0_drop_rms_log, config.sigma_a0_drop_log);

  const double waist_growth_factor = waist_max / waist_entry;
  const double growth_log = positive_log_ratio(waist_growth_factor, config.waist_growth_deadband);
  const double waist_growth_excess_log = (growth_log > 0.0) ? growth_log : 0.0;  // Python max(0.0, x)
  const double waist_growth_component =
      exp_quadratic_component(waist_growth_excess_log, config.sigma_waist_growth_log);

  std::vector<double> log_waist(waist_values.size());
  std::transform(waist_values.begin(), waist_values.end(), log_waist.begin(), [](double w) { return std::log(w); });
  const double waist_jitter_log = np::std_dev(log_waist);
  const double waist_stability_component = exp_quadratic_component(waist_jitter_log, config.sigma_waist_jitter_log);

  const std::array<double, 5> components{a0_retention_component, a0_stability_component, waist_growth_component,
                                         waist_stability_component, coverage_component};
  if (!std::all_of(components.begin(), components.end(), [](double v) { return std::isfinite(v); })) {
    Record extra = common;
    extra.set("metric_guiding_a0_retention_component_v1", a0_retention_component);
    extra.set("metric_guiding_a0_stability_component_v1", a0_stability_component);
    extra.set("metric_guiding_waist_growth_component_v1", waist_growth_component);
    extra.set("metric_guiding_waist_stability_component_v1", waist_stability_component);
    extra.set("metric_guiding_plateau_coverage_component_v1", coverage_component);
    return failure(base, extra, "non_finite_component");
  }

  std::array<double, 4> weights{config.weight_a0_retention, config.weight_a0_stability, config.weight_waist_growth,
                                config.weight_waist_stability};
  const double weight_sum = np::pairwise_sum<double>(weights);
  if (std::any_of(weights.begin(), weights.end(), [](double w) { return !std::isfinite(w) || w < 0.0; }) ||
      weight_sum <= 0.0) {
    return failure(base, common, "invalid_component_weights");
  }
  for (double& w : weights) {
    w = w / weight_sum;
  }

  const double weighted_shape_score = weights[0] * a0_retention_component + weights[1] * a0_stability_component +
                                      weights[2] * waist_growth_component + weights[3] * waist_stability_component;
  const double score = 100.0 * coverage_component * weighted_shape_score;

  Record result = base;
  result.merge(common);
  result.set("metric_guiding_singlecase_status", std::string("ok"));
  result.set("metric_guiding_singlecase_failure_reason", std::string());
  result.set("metric_guiding_singlecase_score_v1", score);
  result.set("metric_guiding_a0_retention_component_v1", a0_retention_component);
  result.set("metric_guiding_a0_stability_component_v1", a0_stability_component);
  result.set("metric_guiding_waist_growth_component_v1", waist_growth_component);
  result.set("metric_guiding_waist_stability_component_v1", waist_stability_component);
  result.set("metric_guiding_plateau_coverage_component_v1", coverage_component);
  result.set("metric_guiding_a0_entry", a0_entry);
  result.set("metric_guiding_a0_exit", a0_exit);
  result.set("metric_guiding_a0_max", a0_max);
  result.set("metric_guiding_a0_retention_ratio", a0_retention_ratio);
  result.set("metric_guiding_a0_drop_rms_log", a0_drop_rms_log);
  result.set("metric_guiding_waist_entry_um", waist_entry);
  result.set("metric_guiding_waist_exit_um", waist_exit);
  result.set("metric_guiding_waist_max_um", waist_max);
  result.set("metric_guiding_waist_growth_factor", waist_growth_factor);
  result.set("metric_guiding_waist_growth_excess_log", waist_growth_excess_log);
  result.set("metric_guiding_waist_jitter_log", waist_jitter_log);
  result.set("metric_guiding_plateau_span_covered", span_fraction);
  result.set("metric_guiding_entry_policy", entry_policy);
  result.set("metric_guiding_exit_policy", exit_policy);
  result.set("metric_guiding_entry_window_mm", config.entry_window_mm);
  result.set("metric_guiding_exit_window_mm", config.exit_window_mm);
  result.set("metric_guiding_min_valid_plateau_rows", static_cast<std::int64_t>(config.min_valid_plateau_rows));
  result.set("metric_guiding_sigma_a0_drop_log", config.sigma_a0_drop_log);
  result.set("metric_guiding_waist_growth_deadband", config.waist_growth_deadband);
  result.set("metric_guiding_sigma_waist_growth_log", config.sigma_waist_growth_log);
  result.set("metric_guiding_sigma_waist_jitter_log", config.sigma_waist_jitter_log);
  result.set("metric_guiding_weight_a0_retention", weights[0]);
  result.set("metric_guiding_weight_a0_stability", weights[1]);
  result.set("metric_guiding_weight_waist_growth", weights[2]);
  result.set("metric_guiding_weight_waist_stability", weights[3]);
  return result;
}

Record score_singlecase_guiding_csv(const std::filesystem::path& csv_path, const std::optional<std::string>& case_id,
                                    const PlateauOverride& plateau, const SingleCaseGuidingConfig& config) {
  const std::string path_text = python_path_string(csv_path);
  const std::string cid = (case_id && !case_id->empty())
                              ? *case_id
                              : std::filesystem::path(path_text).parent_path().filename().string();
  const Record base = base_result(cid, path_text, config);
  if (!std::filesystem::is_regular_file(csv_path)) {
    return failure(base, {}, "missing_csv");
  }
  table::CsvTable table;
  try {
    table = table::read_csv_file(csv_path);
  } catch (const std::exception& error) {
    return failure(base, {}, fmt::format("read_csv_failed: {}", error.what()));
  }
  return score_singlecase_guiding_table(table, cid, path_text, plateau, config);
}

bool ensure_singlecase_guiding_score_csv(const std::filesystem::path& guiding_metrics_csv,
                                         const std::optional<std::string>& case_id, bool overwrite,
                                         std::filesystem::path* score_path_out) {
  const auto score_path = guiding_metrics_csv.parent_path() / kSingleCaseScoreFilename;
  if (score_path_out != nullptr) {
    *score_path_out = score_path;
  }
  if (std::filesystem::exists(score_path) && !overwrite) {
    return false;
  }
  const Record result = score_singlecase_guiding_csv(guiding_metrics_csv, case_id);
  const std::array<Record, 1> records{result};
  table::write_file_atomically(score_path, table::format_records_pandas(records));
  return true;
}

std::string python_path_string(const std::filesystem::path& path) {
  const std::string text = path.generic_string();
  if (text.empty()) {
    return ".";
  }
  const bool absolute = text.front() == '/';
  std::string out = absolute ? "/" : "";
  std::size_t start = 0;
  bool first = true;
  while (start <= text.size()) {
    const std::size_t end = text.find('/', start);
    const std::string_view part(text.data() + start,
                                (end == std::string::npos ? text.size() : end) - start);
    if (!part.empty() && part != ".") {
      if (!first) {
        out.push_back('/');
      }
      out.append(part);
      first = false;
    }
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return out.empty() ? "." : out;
}

}  // namespace guiding::products

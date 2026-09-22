#include "guiding/products/beamlike_pairs.hpp"

#include <algorithm>
#include <cctype>
#include <limits>
#include <stdexcept>
#include <tuple>

#include <fmt/format.h>

#include "guiding/physics/beamlike.hpp"
#include "guiding/products/scoring.hpp"
#include "guiding/table/csv.hpp"
#include "guiding/table/py_format.hpp"

namespace guiding::products {
namespace {

namespace fs = std::filesystem;
using table::Cell;
using table::Record;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kInf = std::numeric_limits<double>::infinity();

const std::vector<std::string> kTransversePairMetrics{
    "n_macroparticles_transverse", "weight_transverse",       "theta_x_rms_mrad",
    "theta_y_rms_mrad",            "theta_rms_mrad",          "theta_x_p95_mrad",
    "theta_y_p95_mrad",            "theta_r_p95_mrad",        "x_rms_um",
    "y_rms_um",                    "x_p95_um",                "y_p95_um",
    "emit_x_norm_mm_mrad",         "emit_y_norm_mm_mrad",     "emit_geom_norm_mm_mrad",
    "transverse_theta_rms_component", "transverse_theta_p95_component", "transverse_emit_component",
    "beam_transverse_quality_score"};

const std::vector<std::string> kTransverseLowerIsBetter{
    "theta_x_rms_mrad", "theta_y_rms_mrad", "theta_rms_mrad", "theta_x_p95_mrad",      "theta_y_p95_mrad",
    "theta_r_p95_mrad", "x_rms_um",         "y_rms_um",       "x_p95_um",              "y_p95_um",
    "emit_x_norm_mm_mrad", "emit_y_norm_mm_mrad", "emit_geom_norm_mm_mrad"};

const std::vector<std::string> kBeamPairMetrics{"beam_yield_score", "charge_hot_pC",  "n_macroparticles_hot",
                                                "E95_hot_MeV",      "Emean_hot_MeV",  "Emax_hot_MeV",
                                                "mono_proxy_E95_over_Emax", "z_span_hot_mm"};

// _finite_float(row, key, default)
double finite_float(const Record& row, std::string_view key, double fallback = kNaN) {
  const auto cell = row.get(key);
  if (!cell) {
    return fallback;
  }
  const auto value = py_float(*cell);
  return value && std::isfinite(*value) ? *value : fallback;
}

std::string string_value(const Record& row, std::string_view key) {
  const auto cell = row.get(key);
  if (!cell || std::holds_alternative<std::monostate>(*cell)) {
    return "";
  }
  return py_str(*cell);
}

Cell boolish(const Record& row, std::string_view key) {
  const auto cell = row.get(key);
  if (cell) {
    if (const auto* flag = std::get_if<bool>(&*cell)) {
      return *flag;
    }
  }
  std::string text = cell ? py_str(*cell) : std::string();
  const auto first = text.find_first_not_of(" \t\n\r\v\f");
  text = first == std::string::npos ? std::string() : text.substr(first, text.find_last_not_of(" \t\n\r\v\f") - first + 1);
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });
  if (text == "true" || text == "1" || text == "yes" || text == "y") {
    return true;
  }
  if (text == "false" || text == "0" || text == "no" || text == "n") {
    return false;
  }
  return std::string();
}

double safe_ratio(double numerator, double denominator) {
  if (!std::isfinite(numerator) || !std::isfinite(denominator) || denominator == 0.0) {
    return kNaN;
  }
  return numerator / denominator;
}

double reference_factor_from_log_advantage(double log_advantage, double deadband_log, double scale_log) {
  if (!std::isfinite(log_advantage)) {
    return kNaN;
  }
  if (scale_log <= 0.0) {
    throw std::invalid_argument("scale_log must be positive");
  }
  const double magnitude = std::abs(log_advantage);
  const double deadbanded = magnitude <= deadband_log ? 0.0 : std::copysign(magnitude - deadband_log, log_advantage);
  return std::tanh(deadbanded / scale_log);
}

std::string lower_stripped(std::string text) {
  const auto first = text.find_first_not_of(" \t\n\r\v\f");
  text = first == std::string::npos ? std::string() : text.substr(first, text.find_last_not_of(" \t\n\r\v\f") - first + 1);
  std::transform(text.begin(), text.end(), text.begin(), [](unsigned char c) { return std::tolower(c); });
  return text;
}

std::string classify_by(const Record& row, std::string_view status_key, std::string_view factor_key) {
  const auto status = row.get(status_key);
  if (lower_stripped(status ? py_str(*status) : std::string()) != "ok") {
    return "failed";
  }
  const double factor = finite_float(row, factor_key);
  if (!std::isfinite(factor)) {
    return "failed";
  }
  return factor > 0.0 ? "positive" : factor < 0.0 ? "negative" : "neutral";
}

void metric_pair(Record& out, const Record& channel, const Record& uniform, const std::string& metric) {
  const double ch = finite_float(channel, metric);
  const double uni = finite_float(uniform, metric);
  out.set(metric + "_channel", ch);
  out.set(metric + "_uniform", uni);
  out.set(metric + "_delta", std::isfinite(ch) && std::isfinite(uni) ? ch - uni : kNaN);
  out.set(metric + "_ratio", safe_ratio(ch, uni));
}

// float(row.get(key, "nan")) with NaN for errors, as the script's _score_float.
double score_float(const Record& row, std::string_view key) {
  const auto cell = row.get(key);
  if (!cell) {
    return kNaN;
  }
  return py_float(*cell).value_or(kNaN);
}

std::string row_text(const Record& row, std::string_view key) {
  const auto cell = row.get(key);
  return cell ? py_str(*cell) : std::string();
}

}  // namespace

std::string py_str(const Cell& cell) {
  struct Visitor {
    std::string operator()(std::monostate) const { return "None"; }
    std::string operator()(bool v) const { return table::py_bool(v); }
    std::string operator()(std::int64_t v) const { return std::to_string(v); }
    std::string operator()(double v) const { return table::py_float_repr(v); }
    std::string operator()(const std::string& v) const { return v; }
  };
  return std::visit(Visitor{}, cell);
}

RowSelection parse_row_selection(std::string_view name) {
  if (name == "single") {
    return RowSelection::Single;
  }
  if (name == "last") {
    return RowSelection::Last;
  }
  if (name == "max-beamlike") {
    return RowSelection::MaxBeamlike;
  }
  throw std::invalid_argument(fmt::format("Unknown row_selection: {}", name));
}

const char* row_selection_name(RowSelection selection) noexcept {
  switch (selection) {
    case RowSelection::Single:
      return "single";
    case RowSelection::Last:
      return "last";
    case RowSelection::MaxBeamlike:
      return "max-beamlike";
  }
  return "single";
}

std::vector<Record> read_dict_rows(const fs::path& path) {
  const auto table = table::read_csv_file(path);
  std::vector<Record> rows;
  for (const auto& values : table.rows) {
    if (values.empty() || (values.size() == 1 && values.front().empty() && table.columns.size() > 1)) {
      continue;  // DictReader skips blank lines
    }
    Record row;
    for (std::size_t c = 0; c < table.columns.size(); ++c) {
      row.set(table.columns[c], c < values.size() ? Cell{values[c]} : Cell{});
    }
    rows.push_back(std::move(row));
  }
  return rows;
}

Record read_particle_summary_row(const fs::path& csv_path, RowSelection selection) {
  const std::string shown = table::python_path_string(csv_path);
  std::error_code error;
  if (!fs::is_regular_file(csv_path, error)) {
    throw std::runtime_error(fmt::format("missing_csv: {}", shown));
  }
  const auto rows = read_dict_rows(csv_path);
  if (rows.empty()) {
    throw std::invalid_argument(fmt::format("empty_csv: {}", shown));
  }
  switch (selection) {
    case RowSelection::Single:
      if (rows.size() != 1) {
        throw std::invalid_argument(fmt::format(
            "expected_single_row_but_found_{}: {}. Use --row-selection last or --row-selection max-beamlike explicitly.",
            rows.size(), shown));
      }
      return rows.front();
    case RowSelection::Last:
    case RowSelection::MaxBeamlike: {
      // sorted(rows, key=...)[-1]: the last of the rows with the largest key
      const char* key = selection == RowSelection::Last ? "iteration" : "beamlike_score";
      std::size_t best = 0;
      for (std::size_t i = 1; i < rows.size(); ++i) {
        if (finite_float(rows[i], key, -kInf) >= finite_float(rows[best], key, -kInf)) {
          best = i;
        }
      }
      return rows[best];
    }
  }
  return rows.front();
}

std::string classify_pair_row(const Record& row) { return classify_by(row, "status", "beamlike_reference_factor"); }

Record compare_beamlike_pair_rows(const Record& channel, const Record& uniform, const std::string& channel_case_id,
                                  const std::string& uniform_case_id, const std::string& channel_csv,
                                  const std::string& uniform_csv, RowSelection selection,
                                  const BeamlikePairConfig& cfg) {
  const double score_ch = finite_float(channel, "beamlike_score", 0.0);
  const double score_uni = finite_float(uniform, "beamlike_score", 0.0);
  const double log_advantage = std::log((score_ch + cfg.score_floor) / (score_uni + cfg.score_floor));
  const double reference_factor =
      reference_factor_from_log_advantage(log_advantage, cfg.reference_deadband_log, cfg.reference_scale_log);
  const double reference_scale_score = score_uni > score_ch ? score_uni : score_ch;  // max(ch, uni)
  const double gain_score = std::isfinite(reference_factor) ? reference_scale_score * reference_factor : kNaN;

  Record out{{"status", std::string("ok")},
             {"failure_reason", std::string()},
             {"comparison_status", std::string("ok")},
             {"comparison_bucket", std::string()},
             {"transverse_comparison_status", std::string()},
             {"transverse_comparison_bucket", std::string()},
             {"channel_case_id", channel_case_id.empty() ? string_value(channel, "case_id") : channel_case_id},
             {"uniform_case_id", uniform_case_id.empty() ? string_value(uniform, "case_id") : uniform_case_id},
             {"channel_csv", channel_csv},
             {"uniform_csv", uniform_csv},
             {"row_selection", std::string(row_selection_name(selection))},
             {"channel_iteration", finite_float(channel, "iteration")},
             {"uniform_iteration", finite_float(uniform, "iteration")},
             {"eligible_beamlike_channel", boolish(channel, "eligible_beamlike")},
             {"eligible_beamlike_uniform", boolish(uniform, "eligible_beamlike")},
             {"beamlike_status_channel", string_value(channel, "beamlike_status")},
             {"beamlike_status_uniform", string_value(uniform, "beamlike_status")},
             {"beamlike_rejection_reasons_channel", string_value(channel, "beamlike_rejection_reasons")},
             {"beamlike_rejection_reasons_uniform", string_value(uniform, "beamlike_rejection_reasons")},
             {"beamlike_score_channel", score_ch},
             {"beamlike_score_uniform", score_uni},
             {"beamlike_score_delta", score_ch - score_uni},
             {"beamlike_score_ratio", safe_ratio(score_ch, score_uni)},
             {"beamlike_score_log_advantage", log_advantage},
             {"beamlike_reference_factor", reference_factor},
             {"beamlike_gain_score", gain_score},
             {"beamlike_reference_scale_score", reference_scale_score}};

  for (const auto& metric : kBeamPairMetrics) {
    metric_pair(out, channel, uniform, metric);
  }

  // _add_transverse_comparison_metrics
  out.set("transverse_status_channel", string_value(channel, "transverse_status"));
  out.set("transverse_status_uniform", string_value(uniform, "transverse_status"));
  for (const auto& metric : kTransversePairMetrics) {
    metric_pair(out, channel, uniform, metric);
  }
  for (const auto& metric : kTransverseLowerIsBetter) {
    const double ch = finite_float(out, metric + "_channel");
    const double uni = finite_float(out, metric + "_uniform");
    out.set(metric + "_improvement", std::isfinite(ch) && std::isfinite(uni) ? uni - ch : kNaN);
  }
  const double transverse_ch = finite_float(channel, "beam_transverse_quality_score");
  const double transverse_uni = finite_float(uniform, "beam_transverse_quality_score");
  if (std::isfinite(transverse_ch) && std::isfinite(transverse_uni)) {
    const double advantage = std::log((transverse_ch + cfg.transverse_score_floor) /
                                      (transverse_uni + cfg.transverse_score_floor));
    const double factor = reference_factor_from_log_advantage(advantage, cfg.transverse_reference_deadband_log,
                                                               cfg.transverse_reference_scale_log);
    const double scale = transverse_uni > transverse_ch ? transverse_uni : transverse_ch;
    out.set("transverse_comparison_status", std::string("ok"));
    out.set("beam_transverse_quality_score_log_advantage", advantage);
    out.set("transverse_reference_factor", factor);
    out.set("transverse_reference_scale_score", scale);
    out.set("transverse_gain_score", std::isfinite(factor) ? scale * factor : kNaN);
  } else {
    out.set("transverse_comparison_status", std::string("failed"));
    out.set("beam_transverse_quality_score_log_advantage", kNaN);
    out.set("transverse_reference_factor", kNaN);
    out.set("transverse_reference_scale_score", kNaN);
    out.set("transverse_gain_score", kNaN);
  }
  out.set("transverse_comparison_bucket",
          classify_by(out, "transverse_comparison_status", "transverse_reference_factor"));
  out.set("divergence_improvement_mrad", *out.get("theta_rms_mrad_improvement"));
  out.set("comparison_bucket", classify_pair_row(out));
  return out;
}

Record compare_beamlike_pair_csvs(const fs::path& channel_csv, const fs::path& uniform_csv,
                                  const std::string& channel_case_id, const std::string& uniform_case_id,
                                  RowSelection selection, const BeamlikePairConfig& config) {
  const std::string channel_text = table::python_path_string(channel_csv);
  const std::string uniform_text = table::python_path_string(uniform_csv);
  const Record base{{"status", std::string("failed")},
                    {"failure_reason", std::string()},
                    {"comparison_status", std::string("failed")},
                    {"comparison_bucket", std::string("failed")},
                    {"transverse_comparison_status", std::string("failed")},
                    {"transverse_comparison_bucket", std::string("failed")},
                    {"channel_case_id", channel_case_id},
                    {"uniform_case_id", uniform_case_id},
                    {"channel_csv", channel_text},
                    {"uniform_csv", uniform_text},
                    {"row_selection", std::string(row_selection_name(selection))},
                    {"beamlike_score_source_channel", std::string()},
                    {"beamlike_score_source_uniform", std::string()},
                    {"beamlike_gain_score", kNaN},
                    {"transverse_gain_score", kNaN}};
  const auto failure = [&](const std::string& reason) {
    Record out = base;
    out.set("failure_reason", reason);
    return out;
  };

  Record channel;
  Record uniform;
  try {
    channel = read_particle_summary_row(channel_csv, selection);
    uniform = read_particle_summary_row(uniform_csv, selection);
  } catch (const std::exception& error) {
    return failure(error.what());
  }

  std::vector<std::string> missing;
  for (const auto& [label, row] : {std::pair<const char*, const Record*>{"channel", &channel}, {"uniform", &uniform}}) {
    for (const char* column : {"charge_hot_pC", "n_macroparticles_hot", "E95_hot_MeV", "Emean_hot_MeV", "Emax_hot_MeV"}) {
      if (!row->get(column)) {
        missing.push_back(fmt::format("{}:{}", label, column));
      }
    }
  }
  if (!missing.empty()) {
    return failure(fmt::format("missing_columns: {}", table::python_list_repr(missing)));
  }

  const auto ensure = [](const Record& row) -> std::pair<Record, std::string> {
    const auto score = row.get("beamlike_score");
    std::string text = score ? py_str(*score) : std::string();
    if (std::holds_alternative<std::monostate>(score.value_or(Cell{std::string()}))) {
      text = "None";
    }
    const auto first = text.find_first_not_of(" \t\n\r\v\f");
    if (first != std::string::npos) {
      return {row, "particle_summary"};
    }
    return {physics::add_beamlike_metrics(row), "computed_on_the_fly"};
  };
  auto [channel_row, channel_source] = ensure(channel);
  auto [uniform_row, uniform_source] = ensure(uniform);

  Record row = compare_beamlike_pair_rows(channel_row, uniform_row, channel_case_id, uniform_case_id, channel_text,
                                          uniform_text, selection, config);
  row.set("beamlike_score_source_channel", channel_source);
  row.set("beamlike_score_source_uniform", uniform_source);
  return row;
}

const std::vector<std::string>& pair_output_columns() {
  static const std::vector<std::string> columns = [] {
    std::vector<std::string> names{"status",
                                   "failure_reason",
                                   "comparison_status",
                                   "comparison_bucket",
                                   "transverse_comparison_status",
                                   "transverse_comparison_bucket",
                                   "channel_case_id",
                                   "uniform_case_id",
                                   "channel_csv",
                                   "uniform_csv",
                                   "row_selection",
                                   "beamlike_score_source_channel",
                                   "beamlike_score_source_uniform",
                                   "channel_iteration",
                                   "uniform_iteration",
                                   "eligible_beamlike_channel",
                                   "eligible_beamlike_uniform",
                                   "beamlike_status_channel",
                                   "beamlike_status_uniform",
                                   "beamlike_rejection_reasons_channel",
                                   "beamlike_rejection_reasons_uniform",
                                   "beamlike_score_channel",
                                   "beamlike_score_uniform",
                                   "beamlike_score_delta",
                                   "beamlike_score_ratio",
                                   "beamlike_score_log_advantage",
                                   "beamlike_reference_factor",
                                   "beamlike_reference_scale_score",
                                   "beamlike_gain_score",
                                   "transverse_status_channel",
                                   "transverse_status_uniform",
                                   "beam_transverse_quality_score_channel",
                                   "beam_transverse_quality_score_uniform",
                                   "beam_transverse_quality_score_delta",
                                   "beam_transverse_quality_score_ratio",
                                   "beam_transverse_quality_score_log_advantage",
                                   "transverse_reference_factor",
                                   "transverse_reference_scale_score",
                                   "transverse_gain_score"};
    for (const auto& metric : kBeamPairMetrics) {
      for (const char* suffix : {"channel", "uniform", "delta", "ratio"}) {
        names.push_back(metric + "_" + suffix);
      }
    }
    for (const auto& metric : kTransversePairMetrics) {
      for (const char* suffix : {"channel", "uniform", "delta", "ratio"}) {
        names.push_back(metric + "_" + suffix);
      }
    }
    for (const auto& metric : kTransverseLowerIsBetter) {
      names.push_back(metric + "_improvement");
    }
    names.emplace_back("divergence_improvement_mrad");
    return names;
  }();
  return columns;
}

std::string format_pair_rows_csv(const std::vector<Record>& rows) {
  std::vector<std::string> fieldnames;
  const auto add = [&](const std::string& name) {
    if (std::find(fieldnames.begin(), fieldnames.end(), name) == fieldnames.end()) {
      fieldnames.push_back(name);
    }
  };
  for (const auto& preferred : pair_output_columns()) {
    if (std::any_of(rows.begin(), rows.end(), [&](const Record& row) { return row.get(preferred).has_value(); })) {
      add(preferred);
    }
  }
  for (const auto& row : rows) {
    for (const auto& [key, value] : row.items()) {
      add(key);
    }
  }
  const table::CsvDialect dialect{"\n", false};
  std::string out;
  if (fieldnames.empty()) {
    return "\n";
  }
  table::append_csv_header(out, fieldnames, dialect);
  std::vector<Cell> cells(fieldnames.size());
  for (const auto& row : rows) {
    for (std::size_t i = 0; i < fieldnames.size(); ++i) {
      cells[i] = row.get(fieldnames[i]).value_or(Cell{std::string()});
    }
    table::append_csv_record(out, cells, dialect);
  }
  return out;
}

std::vector<Record> BeamlikePairTables::all() const {
  std::vector<Record> rows;
  for (const auto* part : {&positive, &neutral, &negative, &failed}) {
    rows.insert(rows.end(), part->begin(), part->end());
  }
  return rows;
}

BeamlikePairTables score_beamlike_pairs(const std::vector<campaign::TripletInfo>& triplets,
                                        const fs::path& case_metrics_root, const std::string& particle_outdir_name,
                                        RowSelection selection, const BeamlikePairConfig& config) {
  BeamlikePairTables tables;
  const auto token_cell = [](const campaign::Token& token) { return token ? Cell{*token} : Cell{}; };
  for (const auto& triplet : triplets) {
    if (!triplet.channel) {
      ++tables.skipped_without_channel;
      continue;
    }
    if (!triplet.uniform) {
      ++tables.skipped_without_uniform;
      continue;
    }
    const auto summary_path = [&](const std::string& case_id) {
      return case_metrics_root / case_id / particle_outdir_name / "particle_summary.csv";
    };
    Record row = compare_beamlike_pair_csvs(summary_path(triplet.channel->case_id),
                                            summary_path(triplet.uniform->case_id), triplet.channel->case_id,
                                            triplet.uniform->case_id, selection, config);
    const auto& tokens = triplet.channel->tokens;
    row.set("pair_label", triplet.label());
    row.set("laser_case", token_cell(tokens.laser_case));
    row.set("density", token_cell(tokens.density));
    row.set("plateau", token_cell(tokens.plateau));
    row.set("focus", token_cell(tokens.focus));
    row.set("diameter", token_cell(tokens.diameter));
    row.set("case_source", triplet.channel->source);
    ++tables.attempted;

    const std::string bucket = classify_pair_row(row);
    row.set("comparison_bucket", bucket);
    (bucket == "positive"   ? tables.positive
     : bucket == "neutral"  ? tables.neutral
     : bucket == "negative" ? tables.negative
                            : tables.failed)
        .push_back(std::move(row));
  }

  std::stable_sort(tables.positive.begin(), tables.positive.end(), [](const Record& a, const Record& b) {
    return score_float(a, "beamlike_gain_score") > score_float(b, "beamlike_gain_score");
  });
  std::stable_sort(tables.negative.begin(), tables.negative.end(), [](const Record& a, const Record& b) {
    return score_float(a, "beamlike_gain_score") < score_float(b, "beamlike_gain_score");
  });
  std::stable_sort(tables.neutral.begin(), tables.neutral.end(), [](const Record& a, const Record& b) {
    return std::make_tuple(row_text(a, "plateau"), row_text(a, "diameter"), row_text(a, "focus"),
                           row_text(a, "channel_case_id")) <
           std::make_tuple(row_text(b, "plateau"), row_text(b, "diameter"), row_text(b, "focus"),
                           row_text(b, "channel_case_id"));
  });
  std::stable_sort(tables.failed.begin(), tables.failed.end(), [](const Record& a, const Record& b) {
    return std::make_tuple(row_text(a, "failure_reason"), row_text(a, "channel_case_id"),
                           row_text(a, "uniform_case_id")) <
           std::make_tuple(row_text(b, "failure_reason"), row_text(b, "channel_case_id"),
                           row_text(b, "uniform_case_id"));
  });
  return tables;
}

std::vector<Record> ranked(const std::vector<Record>& rows, std::size_t top) {
  std::vector<Record> out;
  for (std::size_t i = 0; i < std::min(top, rows.size()); ++i) {
    Record row = rows[i];
    row.set("rank", static_cast<std::int64_t>(i + 1));
    out.push_back(std::move(row));
  }
  return out;
}

}  // namespace guiding::products

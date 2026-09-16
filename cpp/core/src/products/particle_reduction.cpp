#include "guiding/products/particle_reduction.hpp"

#include <algorithm>
#include <mutex>
#include <set>
#include <stdexcept>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "guiding/exec/parallel.hpp"
#include "guiding/io/openpmd_series.hpp"
#include "guiding/io/particle_reader.hpp"
#include "guiding/physics/soft50.hpp"
#include "guiding/table/csv.hpp"
#include "guiding/table/py_format.hpp"

namespace guiding::products {
namespace {

namespace fs = std::filesystem;
using table::Cell;
using table::py_float_repr;
using table::python_path_string;
using table::Record;

// Lexical pathlib parent of a normalised path string.
std::string python_parent(const std::string& path) {
  if (path == "/" || path == ".") {
    return path;
  }
  const auto slash = path.rfind('/');
  if (slash == std::string::npos) {
    return ".";
  }
  return slash == 0 ? "/" : path.substr(0, slash);
}

std::string python_name(const std::string& path) {
  if (path == "/" || path == ".") {
    return "";
  }
  const auto slash = path.rfind('/');
  return slash == std::string::npos ? path : path.substr(slash + 1);
}

// str(value) as print() shows it.
struct PyStr {
  std::string operator()(std::monostate) const { return "None"; }
  std::string operator()(bool value) const { return table::py_bool(value); }
  std::string operator()(double value) const { return py_float_repr(value); }
  std::string operator()(std::int64_t value) const { return std::to_string(value); }
  std::string operator()(const std::string& value) const { return value; }
};

std::string py_str(const Cell& cell) { return std::visit(PyStr{}, cell); }

std::string optional_repr(std::optional<double> value) { return value ? py_float_repr(*value) : "None"; }

std::string int_list_repr(std::span<const std::int64_t> values) {
  std::string out = "[";
  for (std::size_t i = 0; i < values.size(); ++i) {
    out += (i == 0 ? "" : ", ") + std::to_string(values[i]);
  }
  return out + "]";
}

std::vector<std::string> split_list(std::string_view text) {
  std::vector<std::string> items;
  std::string current;
  for (char c : text) {
    if (c == ',' || c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' || c == '\f') {
      if (!current.empty()) {
        items.push_back(std::move(current));
        current.clear();
      }
    } else {
      current.push_back(c);
    }
  }
  if (!current.empty()) {
    items.push_back(std::move(current));
  }
  return items;
}

struct IterationRows {
  std::vector<Record> summary;
  std::vector<Record> acceptance;
  std::vector<Record> soft50;
};

IterationRows analyze_iteration(const io::FileSeries& series, const io::ParticleSeriesInfo& info,
                                const ParticleCaseOptions& options, const Record& selection_info,
                                const std::string& case_id, const std::string& case_name, std::int64_t iteration) {
  std::vector<io::ParticleDump> dumps;
  dumps.reserve(options.species.size());
  for (const auto& species : options.species) {
    dumps.push_back(io::read_particle_dump(series, info, species, iteration, options.raw_reads));
  }

  std::vector<std::pair<std::string, const io::ParticleDump*>> scopes;
  io::ParticleDump combined;
  if (dumps.size() > 1) {
    combined = physics::concatenate_particle_dumps(dumps);
    scopes.emplace_back("all_electrons", &combined);
  }
  for (std::size_t i = 0; i < dumps.size(); ++i) {
    scopes.emplace_back(options.species[i], &dumps[i]);
  }

  std::string selection_mode;
  if (const auto mode = selection_info.get("selection_mode")) {
    selection_mode = py_str(*mode);
  }
  std::int64_t selected_iteration = iteration;
  if (const auto selected = selection_info.get("selected_particle_iteration")) {
    selected_iteration = std::get<std::int64_t>(*selected);
  }

  IterationRows rows;
  for (const auto& [scope, dump] : scopes) {
    physics::SummaryOptions summary_options;
    summary_options.hot_energy_mev = options.hot_energy_mev;
    summary_options.longitudinal = options.longitudinal;
    summary_options.exit_window_mm = options.exit_window_mm;
    summary_options.forward_only = options.forward_only;
    summary_options.soft50 = options.soft50;
    summary_options.species_scope = scope;
    Record row = selection_info;
    row.merge(physics::summarize_dump(*dump, summary_options));
    rows.summary.push_back(std::move(row));

    physics::AcceptanceOptions acceptance_options;
    acceptance_options.case_id = case_id;
    acceptance_options.case_name = case_name;
    acceptance_options.species_scope = scope;
    acceptance_options.selection_mode = selection_mode;
    acceptance_options.selected_particle_iteration = selected_iteration;
    acceptance_options.theta_cuts_mrad = options.acceptance_theta_cuts_mrad;
    acceptance_options.energy_min_mev = options.acceptance_energy_cuts_mev;
    acceptance_options.longitudinal = options.longitudinal;
    acceptance_options.forward_only = options.forward_only;
    auto acceptance = physics::summarize_acceptance_curves(*dump, acceptance_options);
    std::move(acceptance.begin(), acceptance.end(), std::back_inserter(rows.acceptance));

    const auto energy = physics::kinetic_energy_mev(*dump);
    const Record metadata{{"case_id", case_id},
                          {"case_name", case_name},
                          {"species_scope", scope},
                          {"selection_mode", selection_mode},
                          {"selected_particle_iteration", selected_iteration}};
    auto soft50 = physics::summarize_soft50_curve(
        *dump, energy, options.soft50_curve_energy_low_mev, options.soft50.energy_target_mev,
        options.soft50.reliability_floor, options.soft50.effective_count_reference, options.longitudinal,
        options.forward_only, options.exit_window_mm, metadata);
    std::move(soft50.begin(), soft50.end(), std::back_inserter(rows.soft50));
  }
  return rows;
}

// csv.DictWriter(fieldnames=columns).writerows(rows) with extrasaction="raise"
std::string format_dict_rows(std::span<const std::string> columns, std::span<const Record> rows, bool check_extras) {
  const auto dialect = table::CsvDialect::python_csv();
  std::string out;
  table::append_csv_header(out, columns, dialect);
  std::vector<Cell> cells(columns.size());
  for (const auto& row : rows) {
    if (check_extras) {
      std::vector<std::string> wrong;
      for (const auto& [key, value] : row.items()) {
        if (std::find(columns.begin(), columns.end(), key) == columns.end()) {
          wrong.push_back(table::py_str_repr(key));
        }
      }
      if (!wrong.empty()) {
        throw std::invalid_argument(
            fmt::format("dict contains fields not in fieldnames: {}", fmt::join(wrong, ", ")));
      }
    }
    for (std::size_t i = 0; i < columns.size(); ++i) {
      cells[i] = row.get(columns[i]).value_or(Cell{std::string()});
    }
    table::append_csv_record(out, cells, dialect);
  }
  return out;
}

}  // namespace

fs::path particle_case_dir(const fs::path& diag, const fs::path& outdir) {
  const std::string diag_text = python_path_string(diag);
  const std::string diag_parent = python_parent(diag_text);
  if (python_name(diag_parent) == "diags") {
    return python_parent(diag_parent);
  }
  return python_parent(python_path_string(outdir));
}

std::string case_id_from_case_dir(const fs::path& case_dir) {
  const std::string name = python_name(python_path_string(case_dir));
  return name.substr(0, name.find('_'));
}

std::vector<std::string> parse_species_list(std::string_view text) {
  auto values = split_list(text);
  if (values.empty()) {
    throw std::invalid_argument("expected at least one species name");
  }
  if (std::set<std::string>(values.begin(), values.end()).size() != values.size()) {
    throw std::invalid_argument("species names must not be repeated");
  }
  return values;
}

std::vector<double> parse_float_list(std::string_view text) {
  std::vector<double> values;
  for (const auto& item : split_list(text)) {
    const auto value = table::parse_py_float(item);
    if (!value) {
      throw std::invalid_argument(fmt::format("could not convert string to float: {}", table::py_str_repr(item)));
    }
    values.push_back(*value);
  }
  if (values.empty()) {
    throw std::invalid_argument("expected at least one numeric value");
  }
  return values;
}

ParticleCaseTables compute_particle_case_tables(const ParticleCaseOptions& options, const LineSink& log) {
  const auto emit = [&](const std::string& line) {
    if (log) {
      log(line);
    }
  };
  const fs::path case_dir = particle_case_dir(options.diag, options.outdir);

  const auto series = io::FileSeries::scan(options.diag);
  const auto info = io::read_particle_series_info(series);
  const std::vector<std::string> no_names;
  emit(fmt::format("[SERIES] {{'iterations': {}, 'avail_fields': {}, 'avail_species': {}}}",
                   int_list_repr(series.iterations()), table::python_list_repr(info.fields.value_or(no_names)),
                   table::python_list_repr(info.species.value_or(no_names))));

  SelectionRequest request;
  request.which = options.which;
  request.stride = options.stride;
  request.diag = options.diag;
  request.case_dir = case_dir;
  request.guiding_metrics = options.guiding_metrics;
  request.exit_kind = options.exit_kind;
  request.target_propagation_mm = options.target_propagation_mm;
  request.downramp_mm = options.downramp_mm;
  request.resolved_parameters = options.resolved_parameters;
  auto selection = resolve_iterations(series.iterations(), request, log);

  if (options.maximum_target_iteration_delta) {
    validate_exit_iteration_alignment(selection.info, *options.maximum_target_iteration_delta);
    selection.info.set("maximum_target_iteration_delta", *options.maximum_target_iteration_delta);
    selection.info.set("target_iteration_alignment_status", std::string("ok"));
  }
  if (selection.iterations.empty()) {
    throw std::runtime_error(fmt::format("No particle iterations selected in {}", python_path_string(options.diag)));
  }

  emit("[SELECTION]");
  for (const auto& [key, value] : selection.info.items()) {
    emit(fmt::format("  {} = {}", key, py_str(value)));
  }
  options.soft50.validate();

  const std::string case_name = python_name(python_path_string(case_dir));
  const std::string case_id = case_id_from_case_dir(case_dir);
  std::vector<IterationRows> per_iteration(selection.iterations.size());
  const unsigned threads = std::max(1U, std::min<unsigned>(options.threads, selection.iterations.size()));
  exec::parallel_for(selection.iterations.size(), threads, [&](std::size_t index) {
    const std::int64_t iteration = selection.iterations[index];
    emit(fmt::format("[READ] iteration {}", iteration));
    per_iteration[index] =
        analyze_iteration(series, info, options, selection.info, case_id, case_name, iteration);
  });

  ParticleCaseTables tables;
  tables.selection_info = std::move(selection.info);
  for (auto& rows : per_iteration) {
    std::move(rows.summary.begin(), rows.summary.end(), std::back_inserter(tables.summary_rows));
    std::move(rows.acceptance.begin(), rows.acceptance.end(), std::back_inserter(tables.acceptance_rows));
    std::move(rows.soft50.begin(), rows.soft50.end(), std::back_inserter(tables.soft50_curve_rows));
  }
  return tables;
}

std::string format_particle_summary_csv(std::span<const Record> rows) {
  if (rows.empty()) {
    throw std::invalid_argument("No particle summary rows to write");
  }
  std::vector<std::string> columns;
  for (const auto& [key, value] : rows.front().items()) {
    columns.push_back(key);
  }
  return format_dict_rows(columns, rows, true);
}

std::string format_particle_acceptance_csv(std::span<const Record> rows) {
  if (rows.empty()) {
    throw std::invalid_argument("No particle acceptance curve rows to write");
  }
  return format_dict_rows(physics::particle_acceptance_columns(), rows, true);
}

std::string format_particle_soft50_curves_csv(std::span<const Record> rows) {
  if (rows.empty()) {
    throw std::invalid_argument("No soft50 curve rows to write");
  }
  return format_dict_rows(physics::soft50_curve_columns(), rows, false);
}

ParticleCaseOutcome run_particle_case(const ParticleCaseOptions& options, const LineSink& log) {
  const auto emit = [&](const std::string& line) {
    if (log) {
      log(line);
    }
  };
  if (options.resolved_parameters && options.which != ParticleWhich::Exit) {
    throw std::invalid_argument("--resolved-parameters is only valid with --which exit");
  }
  if (options.resolved_parameters && options.maximum_target_iteration_delta) {
    throw std::invalid_argument(
        "--maximum-target-iteration-delta cannot be combined with --resolved-parameters; exact exit selection "
        "already requires the resolved target iteration");
  }

  const fs::path summary_csv = options.outdir / "particle_summary.csv";
  const fs::path acceptance_csv = options.outdir / "particle_acceptance_curves.csv";
  const fs::path soft50_csv = options.outdir / "particle_soft50_curves.csv";
  const std::string summary_text = python_path_string(summary_csv);
  const std::string acceptance_text = python_path_string(acceptance_csv);
  const std::string soft50_text = python_path_string(soft50_csv);

  const bool write_summary = !options.plots_only && (options.overwrite || !fs::exists(summary_csv));
  const bool write_acceptance = !options.plots_only && (options.overwrite || !fs::exists(acceptance_csv));
  const bool write_soft50 = !options.plots_only && (options.overwrite || !fs::exists(soft50_csv));

  if (!options.plots_only && options.skip_existing && !options.overwrite && !write_summary && !write_acceptance &&
      !write_soft50) {
    emit(fmt::format("[SKIP] existing {}, {}, and {}", summary_text, acceptance_text, soft50_text));
    return ParticleCaseOutcome::Skipped;
  }
  if (!options.plots_only && !options.skip_existing && !options.overwrite) {
    std::vector<std::string> existing;
    for (const auto& path : {summary_csv, acceptance_csv, soft50_csv}) {
      if (fs::exists(path)) {
        existing.push_back(python_path_string(path));
      }
    }
    if (!existing.empty()) {
      throw std::runtime_error(fmt::format("Output already exists: {}. Use --overwrite or --skip-existing.",
                                           fmt::join(existing, ", ")));
    }
  }

  const fs::path case_dir = particle_case_dir(options.diag, options.outdir);
  emit("=== Particle case analysis ===");
  emit(fmt::format("case_dir          = {}", python_path_string(case_dir)));
  emit(fmt::format("diag              = {}", python_path_string(options.diag)));
  emit(fmt::format("outdir            = {}", python_path_string(options.outdir)));
  emit(fmt::format("species           = {}", table::python_list_repr(options.species)));
  emit(fmt::format("which             = {}", particle_which_name(options.which)));
  emit(fmt::format("exit_kind         = {}", exit_kind_name(options.exit_kind)));
  emit(fmt::format("target_prop_mm    = {}", optional_repr(options.target_propagation_mm)));
  emit(fmt::format("resolved_params   = {}",
                   options.resolved_parameters ? python_path_string(*options.resolved_parameters) : "None"));
  emit(fmt::format("guiding_metrics   = {}", python_path_string(options.guiding_metrics
                                                                    ? *options.guiding_metrics
                                                                    : case_dir / "guiding_metrics.csv")));
  emit(fmt::format("max_target_delta  = {}", options.maximum_target_iteration_delta
                                                 ? std::to_string(*options.maximum_target_iteration_delta)
                                                 : std::string("None")));
  emit(fmt::format("stride            = {}", options.stride));
  emit(fmt::format("hot_energy_mev    = {}", py_float_repr(options.hot_energy_mev)));
  emit(fmt::format("longitudinal      = {}", physics::longitudinal_name(options.longitudinal)));
  emit(fmt::format("forward_cut       = {}", table::py_bool(options.forward_only)));
  emit(fmt::format("exit_window_mm    = {}", optional_repr(options.exit_window_mm)));
  emit(fmt::format("downramp_mm      = {}", optional_repr(options.downramp_mm)));
  emit(fmt::format("spectrum_emin    = {}", py_float_repr(options.spectrum_emin_mev)));
  emit(fmt::format("spectrum_log_y   = {}", table::py_bool(options.spectrum_log_y)));
  emit(fmt::format("plots_only       = {}", table::py_bool(options.plots_only)));
  emit(fmt::format("accept_theta_mrad= {}", table::python_float_list_repr(options.acceptance_theta_cuts_mrad)));
  emit(fmt::format("accept_Emin_MeV  = {}", table::python_float_list_repr(options.acceptance_energy_cuts_mev)));
  emit(fmt::format("soft50_E_low     = {}", py_float_repr(options.soft50.energy_low_mev)));
  emit(fmt::format("soft50_E_target  = {}", py_float_repr(options.soft50.energy_target_mev)));
  emit(fmt::format("soft50_N_ref     = {}", py_float_repr(options.soft50.effective_count_reference)));
  emit(fmt::format("soft50_curve_low = {}", table::python_float_list_repr(options.soft50_curve_energy_low_mev)));
  emit("==============================");

  if (options.plots_only) {
    emit(fmt::format("[PLOTS-ONLY] guiding_cli renders no plots; left unchanged {}, {}, {}", summary_text,
                     acceptance_text, soft50_text));
    return ParticleCaseOutcome::PlotsOnly;
  }

  const auto tables = compute_particle_case_tables(options, log);

  const auto write_or_use = [&](bool write, const fs::path& path, const std::string& shown, const std::string& text) {
    if (write) {
      table::write_file_atomically(path, text);
      emit(fmt::format("[OK] wrote {}", shown));
    } else {
      emit(fmt::format("[USE] existing {}", shown));
    }
  };
  // Format everything first so a formatting error leaves no partial product set.
  const std::string summary = format_particle_summary_csv(tables.summary_rows);
  const std::string acceptance = format_particle_acceptance_csv(tables.acceptance_rows);
  const std::string soft50 = format_particle_soft50_curves_csv(tables.soft50_curve_rows);
  write_or_use(write_summary, summary_csv, summary_text, summary);
  write_or_use(write_acceptance, acceptance_csv, acceptance_text, acceptance);
  write_or_use(write_soft50, soft50_csv, soft50_text, soft50);
  return ParticleCaseOutcome::Written;
}

}  // namespace guiding::products

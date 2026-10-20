#include "guiding/products/particle_exit.hpp"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "guiding/table/csv.hpp"
#include "guiding/table/py_format.hpp"

namespace guiding::products {
namespace {

namespace fs = std::filesystem;
using json = nlohmann::json;
using table::Cell;
using table::python_path_string;
using table::Record;

constexpr const char* kWhitespace = " \t\n\r\v\f";

std::string_view strip(std::string_view text, std::string_view characters = kWhitespace) {
  const auto first = text.find_first_not_of(characters);
  if (first == std::string_view::npos) {
    return {};
  }
  return text.substr(first, text.find_last_not_of(characters) - first + 1);
}

// str.splitlines() for the ASCII line boundaries.
std::vector<std::string_view> split_lines(std::string_view text) {
  std::vector<std::string_view> lines;
  std::size_t start = 0;
  for (std::size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    if (c == '\n' || c == '\r' || c == '\v' || c == '\f' || c == '\x1c' || c == '\x1d' || c == '\x1e') {
      lines.push_back(text.substr(start, i - start));
      if (c == '\r' && i + 1 < text.size() && text[i + 1] == '\n') {
        ++i;
      }
      start = i + 1;
    }
  }
  if (start < text.size()) {
    lines.push_back(text.substr(start));
  }
  return lines;
}

// shlex.split(text) (POSIX mode, no comments); nullopt where shlex raises ValueError.
std::optional<std::vector<std::string>> shlex_split(std::string_view text) {
  const auto is_space = [](char c) { return c == ' ' || c == '\t' || c == '\r' || c == '\n'; };
  const auto is_quote = [](char c) { return c == '\'' || c == '"'; };
  std::vector<std::string> tokens;
  std::string token;
  bool quoted = false;
  char state = ' ';
  char escaped_state = ' ';
  std::size_t i = 0;
  while (true) {
    const bool eof = i >= text.size();
    const char c = eof ? '\0' : text[i++];
    if (state == ' ') {
      if (eof) {
        break;
      }
      if (is_space(c)) {
        continue;
      }
      if (c == '\\') {
        escaped_state = 'a';
        state = '\\';
      } else if (is_quote(c)) {
        state = c;
      } else {
        token.push_back(c);
        state = 'a';
      }
    } else if (is_quote(state)) {
      quoted = true;
      if (eof) {
        return std::nullopt;  // No closing quotation
      }
      if (c == state) {
        state = 'a';
      } else if (c == '\\' && state == '"') {
        escaped_state = state;
        state = '\\';
      } else {
        token.push_back(c);
      }
    } else if (state == '\\') {
      if (eof) {
        return std::nullopt;  // No escaped character
      }
      if (is_quote(escaped_state) && c != '\\' && c != escaped_state) {
        token.push_back('\\');
      }
      token.push_back(c);
      state = escaped_state;
    } else {  // 'a'
      if (eof) {
        break;
      }
      if (is_space(c)) {
        state = ' ';
        tokens.push_back(std::move(token));
        token.clear();
        quoted = false;
      } else if (is_quote(c)) {
        state = c;
      } else if (c == '\\') {
        escaped_state = 'a';
        state = '\\';
      } else {
        token.push_back(c);
      }
    }
  }
  if (!token.empty() || quoted) {
    tokens.push_back(std::move(token));
  }
  return tokens;
}

std::string read_text(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error(fmt::format("cannot open {}", path.string()));
  }
  std::ostringstream buffer;
  buffer << stream.rdbuf();
  return buffer.str();
}

// Path(p).name
std::string python_path_name(const fs::path& path) {
  const std::string text = python_path_string(path);
  if (text == "." || text == "/") {
    return "";
  }
  const auto slash = text.rfind('/');
  return slash == std::string::npos ? text : text.substr(slash + 1);
}

// float(value) for a decoded JSON value; nullopt where Python raises TypeError/ValueError.
std::optional<double> json_to_py_float(const json& value) {
  switch (value.type()) {
    case json::value_t::boolean:
      return value.get<bool>() ? 1.0 : 0.0;
    case json::value_t::number_integer:
      return static_cast<double>(value.get<std::int64_t>());
    case json::value_t::number_unsigned:
      return static_cast<double>(value.get<std::uint64_t>());
    case json::value_t::number_float:
      return value.get<double>();
    case json::value_t::string:
      return table::parse_py_float(value.get_ref<const std::string&>());
    default:
      return std::nullopt;
  }
}

json load_json_file(const fs::path& path) {
  try {
    return json::parse(read_text(path));
  } catch (const std::exception&) {
    throw std::runtime_error(
        fmt::format("Could not read resolved simulation parameters: {}", python_path_string(path)));
  }
}

std::int64_t record_int(const Record& record, std::string_view key) {
  const auto cell = record.get(key);
  if (cell) {
    if (const auto* integer = std::get_if<std::int64_t>(&*cell)) {
      return *integer;
    }
    if (const auto* real = std::get_if<double>(&*cell)) {
      return static_cast<std::int64_t>(*real);
    }
  }
  throw std::invalid_argument(fmt::format("selection metadata field '{}' is not an integer", key));
}

bool double_equals_int(double value, std::int64_t integer) {
  // Python compares float and int exactly.
  if (!std::isfinite(value) || value >= 9223372036854775808.0 || value < -9223372036854775808.0) {
    return false;
  }
  return value == std::trunc(value) && static_cast<std::int64_t>(value) == integer;
}

}  // namespace

ExitKind parse_exit_kind(std::string_view name) {
  if (name == "plateau") {
    return ExitKind::Plateau;
  }
  if (name == "capillary") {
    return ExitKind::Capillary;
  }
  throw std::invalid_argument(fmt::format("Unknown exit_kind: {}", name));
}

const char* exit_kind_name(ExitKind kind) noexcept { return kind == ExitKind::Plateau ? "plateau" : "capillary"; }

ParticleWhich parse_particle_which(std::string_view name) {
  if (name == "last") {
    return ParticleWhich::Last;
  }
  if (name == "all") {
    return ParticleWhich::All;
  }
  if (name == "exit") {
    return ParticleWhich::Exit;
  }
  throw std::invalid_argument(fmt::format("Unknown --which mode: {}", name));
}

const char* particle_which_name(ParticleWhich which) noexcept {
  switch (which) {
    case ParticleWhich::Last:
      return "last";
    case ParticleWhich::All:
      return "all";
    case ParticleWhich::Exit:
      return "exit";
  }
  return "last";
}

CaseEnv parse_case_env(const fs::path& path) {
  CaseEnv env;
  if (!fs::exists(path)) {
    return env;
  }
  const std::string text = read_text(path);
  for (const auto raw : split_lines(text)) {
    std::string_view line = strip(raw);
    if (line.empty() || line.front() == '#') {
      continue;
    }
    if (line.starts_with("export ")) {
      line = strip(line.substr(7));
    }
    const auto equals = line.find('=');
    if (equals == std::string_view::npos) {
      continue;
    }
    const std::string key(strip(line.substr(0, equals)));
    const std::string_view value = strip(line.substr(equals + 1));
    if (const auto parts = shlex_split(value)) {
      env[key] = parts->empty() ? std::string() : parts->front();
    } else {
      env[key] = std::string(strip(value, "'\""));
    }
  }
  return env;
}

std::optional<double> get_float_env(const CaseEnv& env, std::initializer_list<const char*> names) {
  for (const char* name : names) {
    const auto it = env.find(name);
    if (it != env.end() && !it->second.empty()) {
      if (const auto value = table::parse_py_float(it->second)) {
        return value;
      }
    }
  }
  return std::nullopt;
}

std::optional<double> infer_plateau_length_from_case_name(std::string_view name) {
  const auto is_digit = [](char c) { return c >= '0' && c <= '9'; };
  for (std::size_t start = name.find("_L"); start != std::string_view::npos; start = name.find("_L", start + 1)) {
    std::size_t i = start + 2;
    const std::size_t digits_begin = i;
    while (i < name.size() && is_digit(name[i])) {
      ++i;
    }
    if (i == digits_begin) {
      continue;
    }
    // Optional "[p.][0-9]+" fraction.
    if (i + 1 < name.size() && (name[i] == 'p' || name[i] == '.') && is_digit(name[i + 1])) {
      std::size_t j = i + 1;
      while (j < name.size() && is_digit(name[j])) {
        ++j;
      }
      if (name.substr(j).starts_with("mm_")) {
        std::string number(name.substr(digits_begin, j - digits_begin));
        std::replace(number.begin(), number.end(), 'p', '.');
        return table::parse_py_float(number);
      }
    }
    if (name.substr(i).starts_with("mm_")) {
      return table::parse_py_float(name.substr(digits_begin, i - digits_begin));
    }
  }
  return std::nullopt;
}

Record read_resolved_particle_exit_target(const fs::path& path, ExitKind kind) {
  const std::string shown = python_path_string(path);
  if (!fs::is_regular_file(path)) {
    throw std::runtime_error(fmt::format("Missing resolved simulation parameters: {}", shown));
  }
  const json payload = load_json_file(path);
  if (!payload.is_object()) {
    throw std::runtime_error(fmt::format("Resolved simulation parameters must be a JSON object: {}", shown));
  }
  const std::string target_key = kind == ExitKind::Plateau ? "plateau_exit" : "capillary_exit";
  const std::string key_repr = table::py_str_repr(target_key);

  const auto targets = payload.find("particle_diagnostic_targets");
  if (targets == payload.end() || !targets->is_object()) {
    throw std::runtime_error(
        fmt::format("Resolved simulation parameters lack particle_diagnostic_targets in {}", shown));
  }
  const auto target = targets->find(target_key);
  if (target == targets->end() || !target->is_object()) {
    throw std::runtime_error(
        fmt::format("Resolved simulation parameters lack particle target {} in {}", key_repr, shown));
  }
  const auto raw = target->find("iteration");
  if (raw == target->end()) {
    throw std::runtime_error(fmt::format("Particle target {} lacks iteration in {}", key_repr, shown));
  }
  if (raw->is_boolean()) {
    throw std::runtime_error(fmt::format("Particle target {} iteration must be an integer", key_repr));
  }

  const auto not_numeric = [&] {
    return std::runtime_error(fmt::format("Particle target {} iteration is not numeric in {}", key_repr, shown));
  };
  const auto not_integer = [&] {
    return std::runtime_error(fmt::format("Particle target {} iteration must be a non-negative integer", key_repr));
  };
  std::int64_t iteration = 0;
  double numeric = 0.0;
  switch (raw->type()) {
    case json::value_t::number_integer:
      iteration = raw->get<std::int64_t>();
      numeric = static_cast<double>(iteration);
      break;
    case json::value_t::number_unsigned: {
      const auto value = raw->get<std::uint64_t>();
      if (value > static_cast<std::uint64_t>(std::numeric_limits<std::int64_t>::max())) {
        throw not_integer();
      }
      iteration = static_cast<std::int64_t>(value);
      numeric = static_cast<double>(value);
      break;
    }
    case json::value_t::number_float:
      numeric = raw->get<double>();
      if (std::isnan(numeric)) {
        throw not_numeric();
      }
      if (!std::isfinite(numeric) || std::abs(numeric) >= 9223372036854775808.0) {
        throw not_integer();
      }
      iteration = static_cast<std::int64_t>(std::trunc(numeric));
      break;
    case json::value_t::string: {
      const auto integer = table::parse_py_int(raw->get_ref<const std::string&>());
      const auto real = table::parse_py_float(raw->get_ref<const std::string&>());
      if (!integer || !real) {
        throw not_numeric();
      }
      iteration = *integer;
      numeric = *real;
      break;
    }
    default:
      throw not_numeric();
  }
  if (!double_equals_int(numeric, iteration) || iteration < 0) {
    throw not_integer();
  }

  Record result;
  result.set("selection_mode", std::string("exit"));
  result.set("particle_exit_selection_policy", std::string("exact_resolved_v1"));
  result.set("resolved_parameters_path", python_path_string(fs::weakly_canonical(fs::absolute(path))));
  result.set("resolved_particle_target_key", target_key);
  result.set("target_particle_iteration", iteration);
  for (const char* key : {"target_distance_m", "dump_distance_m", "distance_error_m"}) {
    const auto field = target->find(key);
    if (field == target->end()) {
      continue;
    }
    const auto value = json_to_py_float(*field);
    if (!value) {
      throw std::runtime_error(fmt::format("Particle target {} field {} is not numeric in {}", key_repr,
                                           table::py_str_repr(key), shown));
    }
    if (!std::isfinite(*value)) {
      throw std::runtime_error(fmt::format("Particle target {} field {} is not finite in {}", key_repr,
                                           table::py_str_repr(key), shown));
    }
    result.set(key, *value);
  }
  if (const auto distance = result.get("target_distance_m")) {
    result.set("target_propagation_mm", std::get<double>(*distance) * 1.0e3);
  }
  return result;
}

std::optional<double> target_propagation_from_resolved_parameters(const fs::path& case_dir, ExitKind kind) {
  const fs::path path = case_dir / "resolved_parameters.json";
  if (!fs::exists(path)) {
    return std::nullopt;
  }
  const std::string shown = python_path_string(path);
  const json payload = load_json_file(path);
  if (!payload.is_object()) {
    throw std::runtime_error(fmt::format("Resolved simulation parameters must be a JSON object: {}", shown));
  }
  const std::string target_key = kind == ExitKind::Plateau ? "plateau_end_z" : "plasma_end_z";
  std::vector<std::string> missing;
  for (const auto& key : {std::string("plasma_start_z"), target_key}) {
    if (!payload.contains(key)) {
      missing.push_back(key);
    }
  }
  if (!missing.empty()) {
    throw std::runtime_error(fmt::format("Resolved simulation parameters lack {} in {}",
                                         table::python_list_repr(missing), shown));
  }
  const auto start_m = json_to_py_float(payload.at("plasma_start_z"));
  const auto target_m = start_m ? json_to_py_float(payload.at(target_key)) : std::nullopt;
  if (!start_m || !target_m) {
    throw std::runtime_error(fmt::format("Resolved longitudinal boundaries are not numeric in {}", shown));
  }
  if (!std::isfinite(*start_m) || !std::isfinite(*target_m)) {
    throw std::runtime_error(fmt::format("Resolved longitudinal boundaries are not finite in {}", shown));
  }
  if (*target_m <= *start_m) {
    throw std::runtime_error(
        fmt::format("Resolved {} must be downstream of plasma_start_z in {}", target_key, shown));
  }
  return (*target_m - *start_m) * 1.0e3;
}

double target_propagation_from_case(const fs::path& case_dir, ExitKind kind,
                                    std::optional<double> target_propagation_mm,
                                    std::optional<double> downramp_mm, const LineSink& log) {
  if (target_propagation_mm) {
    return *target_propagation_mm;
  }
  if (const auto resolved = target_propagation_from_resolved_parameters(case_dir, kind)) {
    return *resolved;
  }

  const CaseEnv env = parse_case_env(case_dir / "case.env");
  auto plateau_mm = get_float_env(env, {"PLATEAU_LENGTH_MM", "plateau_length_mm", "L_PLATEAU_MM", "LENGTH_MM"});
  if (!plateau_mm) {
    if (const auto plateau_m = get_float_env(env, {"CAP_PLATEAU_LENGTH_M", "PLATEAU_LENGTH_M", "plateau_length_m"})) {
      plateau_mm = *plateau_m * 1.0e3;
    }
  }
  if (!plateau_mm) {
    plateau_mm = infer_plateau_length_from_case_name(python_path_name(case_dir));
  }
  if (!plateau_mm) {
    throw std::runtime_error(fmt::format(
        "Could not infer plateau length from {} or case name {}. Use --target-propagation-mm explicitly.",
        python_path_string(case_dir / "case.env"), table::py_str_repr(python_path_name(case_dir))));
  }

  if (kind == ExitKind::Plateau) {
    return *plateau_mm;
  }
  if (downramp_mm) {
    return *plateau_mm + *downramp_mm;
  }
  if (const auto capillary_mm =
          get_float_env(env, {"CAPILLARY_LENGTH_MM", "capillary_length_mm", "TOTAL_CAPILLARY_LENGTH_MM"})) {
    return *capillary_mm;
  }
  if (const auto env_downramp_mm =
          get_float_env(env, {"DOWNRAMP_LENGTH_MM", "RAMP_DOWN_LENGTH_MM", "downramp_length_mm"})) {
    return *plateau_mm + *env_downramp_mm;
  }
  if (log) {
    log("[WARN] --exit-kind capillary requested, but no explicit capillary or downramp length was found in "
        "case.env. Falling back to plateau exit.");
  }
  return *plateau_mm;
}

Record read_guiding_iteration_at_propagation(const fs::path& metrics_csv, double target_propagation_mm) {
  const auto table = table::read_csv_file(metrics_csv);
  std::vector<std::string> missing;
  for (const char* column : {"iteration", "propagation_mm"}) {
    if (!table.has_column(column)) {
      missing.emplace_back(column);
    }
  }
  if (!missing.empty()) {
    throw std::runtime_error(fmt::format("Missing columns in {}: {}", python_path_string(metrics_csv),
                                         table::python_list_repr(missing)));
  }
  const auto iterations = table.numeric_column("iteration");
  const auto propagation = table.numeric_column("propagation_mm");

  std::optional<std::size_t> best;
  double best_distance = 0.0;
  for (std::size_t i = 0; i < iterations.size(); ++i) {
    if (std::isnan(iterations[i]) || std::isnan(propagation[i])) {
      continue;
    }
    const double distance = std::abs(propagation[i] - target_propagation_mm);
    // idxmin: first minimum, NaN distances skipped
    if (!std::isnan(distance) && (!best || distance < best_distance)) {
      best = i;
      best_distance = distance;
    }
  }
  if (!best) {
    throw std::runtime_error(
        fmt::format("No valid iteration/propagation rows in {}", python_path_string(metrics_csv)));
  }

  Record info;
  info.set("target_propagation_mm", target_propagation_mm);
  info.set("target_guiding_iteration", static_cast<std::int64_t>(iterations[*best]));
  info.set("target_guiding_propagation_mm", propagation[*best]);
  return info;
}

Record require_exact_particle_iteration(std::span<const std::int64_t> iterations, std::int64_t target,
                                        const fs::path& diag) {
  std::vector<std::int64_t> available(iterations.begin(), iterations.end());
  std::sort(available.begin(), available.end());
  available.erase(std::unique(available.begin(), available.end()), available.end());
  if (available.empty()) {
    throw std::runtime_error(fmt::format("No particle iterations found in {}", python_path_string(diag)));
  }
  if (!std::binary_search(available.begin(), available.end(), target)) {
    throw std::runtime_error(fmt::format(
        "Exact particle exit dump is missing: target iteration={}, available min={}, available max={}, count={}. "
        "Refusing nearest/last fallback.",
        target, available.front(), available.back(), available.size()));
  }
  Record info;
  info.set("selected_particle_iteration", target);
  info.set("target_iteration_delta", std::int64_t{0});
  info.set("available_particle_iterations_min", available.front());
  info.set("available_particle_iterations_max", available.back());
  info.set("n_available_particle_iterations", static_cast<std::int64_t>(available.size()));
  return info;
}

Record nearest_particle_iteration(std::span<const std::int64_t> iterations, std::int64_t target,
                                  const fs::path& diag) {
  if (iterations.empty()) {
    throw std::runtime_error(fmt::format("No particle iterations found in {}", python_path_string(diag)));
  }
  std::size_t best = 0;
  for (std::size_t i = 1; i < iterations.size(); ++i) {
    if (std::abs(iterations[i] - target) < std::abs(iterations[best] - target)) {
      best = i;
    }
  }
  const auto [min_it, max_it] = std::minmax_element(iterations.begin(), iterations.end());
  Record info;
  info.set("selected_particle_iteration", iterations[best]);
  info.set("target_iteration_delta", iterations[best] - target);
  info.set("available_particle_iterations_min", *min_it);
  info.set("available_particle_iterations_max", *max_it);
  info.set("n_available_particle_iterations", static_cast<std::int64_t>(iterations.size()));
  return info;
}

void validate_exit_iteration_alignment(const Record& selection_info, std::int64_t maximum_delta) {
  if (maximum_delta < 0) {
    throw std::invalid_argument("maximum target iteration delta must be non-negative");
  }
  const auto mode = selection_info.get("selection_mode");
  const auto* mode_text = mode ? std::get_if<std::string>(&*mode) : nullptr;
  if (mode_text == nullptr || *mode_text != "exit") {
    throw std::invalid_argument("target-iteration alignment can only be required for --which exit");
  }
  std::vector<std::string> missing;
  for (const char* key : {"target_guiding_iteration", "selected_particle_iteration", "target_iteration_delta"}) {
    if (!selection_info.get(key)) {
      missing.emplace_back(key);
    }
  }
  if (!missing.empty()) {
    throw std::invalid_argument(
        fmt::format("exit selection metadata lacks {}", table::python_list_repr(missing)));
  }
  const std::int64_t target = record_int(selection_info, "target_guiding_iteration");
  const std::int64_t selected = record_int(selection_info, "selected_particle_iteration");
  const std::int64_t delta = record_int(selection_info, "target_iteration_delta");
  if (selected - target != delta) {
    throw std::invalid_argument(fmt::format(
        "inconsistent exit selection metadata: selected={}, target={}, delta={}", selected, target, delta));
  }
  if (std::abs(delta) > maximum_delta) {
    throw std::runtime_error(fmt::format(
        "Particle diagnostic is not aligned with the requested exit frame: target guiding iteration={}, selected "
        "particle iteration={}, delta={}, allowed={}. Refusing to analyze a distant snapshot as an exit "
        "measurement.",
        target, selected, delta, maximum_delta));
  }
}

IterationSelection resolve_iterations(std::span<const std::int64_t> series_iterations,
                                      const SelectionRequest& request, const LineSink& log) {
  IterationSelection selection;
  switch (request.which) {
    case ParticleWhich::Last: {
      if (series_iterations.empty()) {
        throw std::runtime_error(fmt::format("No iterations found in particle diagnostic: {}",
                                             python_path_string(request.diag)));
      }
      selection.iterations = {series_iterations.back()};
      selection.info.set("selection_mode", std::string("last"));
      selection.info.set("selected_particle_iteration", series_iterations.back());
      return selection;
    }
    case ParticleWhich::All: {
      if (request.stride < 1) {
        throw std::invalid_argument("stride must be >= 1");
      }
      for (std::size_t i = 0; i < series_iterations.size(); i += static_cast<std::size_t>(request.stride)) {
        selection.iterations.push_back(series_iterations[i]);
      }
      selection.info.set("selection_mode", std::string("all"));
      selection.info.set("analysis_stride", static_cast<std::int64_t>(request.stride));
      return selection;
    }
    case ParticleWhich::Exit:
      break;
  }

  const fs::path metrics_csv =
      request.guiding_metrics ? *request.guiding_metrics : request.case_dir / "guiding_metrics.csv";

  if (request.resolved_parameters) {
    if (request.target_propagation_mm) {
      throw std::invalid_argument(
          "--target-propagation-mm cannot be combined with authoritative --resolved-parameters exit selection");
    }
    const Record target_info = read_resolved_particle_exit_target(*request.resolved_parameters, request.exit_kind);
    const std::int64_t target_iteration = std::get<std::int64_t>(*target_info.get("target_particle_iteration"));
    const Record particle_info = require_exact_particle_iteration(series_iterations, target_iteration, request.diag);

    Record& info = selection.info;
    info.set("selection_mode", std::string("exit_exact_resolved"));
    info.set("exit_kind", std::string(exit_kind_name(request.exit_kind)));
    info.merge(target_info);
    info.merge(particle_info);

    const auto target_mm = target_info.get("target_propagation_mm");
    if (fs::exists(metrics_csv) && target_mm) {
      const Record guiding_info = read_guiding_iteration_at_propagation(metrics_csv, std::get<double>(*target_mm));
      info.set("guiding_context_available", true);
      info.set("guiding_metrics_csv", python_path_string(metrics_csv));
      info.merge(guiding_info);
      info.set("particle_vs_guiding_iteration_delta",
               record_int(particle_info, "selected_particle_iteration") -
                   record_int(guiding_info, "target_guiding_iteration"));
    } else {
      info.set("guiding_context_available", false);
      info.set("guiding_metrics_csv", python_path_string(metrics_csv));
    }
    selection.iterations = {record_int(particle_info, "selected_particle_iteration")};
    return selection;
  }

  if (!fs::exists(metrics_csv)) {
    throw std::runtime_error(
        fmt::format("Missing guiding metrics for --which exit: {}", python_path_string(metrics_csv)));
  }
  const double target_mm = target_propagation_from_case(request.case_dir, request.exit_kind,
                                                        request.target_propagation_mm, request.downramp_mm, log);
  const Record guiding_info = read_guiding_iteration_at_propagation(metrics_csv, target_mm);
  const Record particle_info = nearest_particle_iteration(
      series_iterations, record_int(guiding_info, "target_guiding_iteration"), request.diag);

  Record& info = selection.info;
  info.set("selection_mode", std::string("exit"));
  info.set("exit_kind", std::string(exit_kind_name(request.exit_kind)));
  info.set("guiding_metrics_csv", python_path_string(metrics_csv));
  info.merge(guiding_info);
  info.merge(particle_info);
  selection.iterations = {record_int(particle_info, "selected_particle_iteration")};
  return selection;
}

}  // namespace guiding::products

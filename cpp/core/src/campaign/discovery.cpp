#include "guiding/campaign/discovery.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <set>
#include <stdexcept>
#include <utility>

#include <fmt/format.h>

#ifndef _WIN32
#include <sys/stat.h>
#include <time.h>
#else
#include <chrono>
#endif

#include "guiding/table/csv.hpp"
#include "guiding/table/py_format.hpp"

namespace guiding::campaign {
namespace {

namespace fs = std::filesystem;
using table::python_path_string;

char ascii_lower(char c) noexcept { return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c; }

bool is_digit(char c) noexcept { return c >= '0' && c <= '9'; }

bool is_space(char c) noexcept { return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\f' || c == '\v'; }

std::string to_lower(std::string_view text) {
  std::string out(text);
  for (char& c : out) {
    c = ascii_lower(c);
  }
  return out;
}

bool iequals(std::string_view a, std::string_view b) {
  return a.size() == b.size() &&
         std::equal(a.begin(), a.end(), b.begin(), [](char x, char y) { return ascii_lower(x) == ascii_lower(y); });
}

bool istarts_with(std::string_view text, std::string_view prefix) {
  return text.size() >= prefix.size() && iequals(text.substr(0, prefix.size()), prefix);
}

bool iends_with(std::string_view text, std::string_view suffix) {
  return text.size() >= suffix.size() && iequals(text.substr(text.size() - suffix.size()), suffix);
}

bool all_digits(std::string_view text) { return !text.empty() && std::all_of(text.begin(), text.end(), is_digit); }

// \d+(?:[p.]\d+)? over the whole text (IGNORECASE)
bool is_decimal_number(std::string_view text) {
  std::size_t i = 0;
  while (i < text.size() && is_digit(text[i])) {
    ++i;
  }
  if (i == 0) {
    return false;
  }
  if (i == text.size()) {
    return true;
  }
  const char separator = ascii_lower(text[i]);
  if (separator != 'p' && separator != '.') {
    return false;
  }
  std::size_t j = i + 1;
  while (j < text.size() && is_digit(text[j])) {
    ++j;
  }
  return j > i + 1 && j == text.size();
}

// campaign.py:_norm_token
std::string normalize_token(std::string_view value) {
  std::string out(value);
  for (char& c : out) {
    c = c == '.' ? 'p' : ascii_lower(c);
  }
  return out;
}

std::vector<std::string_view> split_keep_empty(std::string_view text, char delimiter) {
  std::vector<std::string_view> parts;
  std::size_t start = 0;
  for (;;) {
    const std::size_t end = text.find(delimiter, start);
    parts.push_back(text.substr(start, end == std::string_view::npos ? std::string_view::npos : end - start));
    if (end == std::string_view::npos) {
      return parts;
    }
    start = end + 1;
  }
}

// The campaign regexes are all anchored at "_" boundaries and never span an
// underscore, so re.search is equivalent to the first matching token.
template <typename Match>
Token first_token(std::string_view case_id, Match&& match) {
  for (std::string_view token : split_keep_empty(case_id, '_')) {
    if (Token value = match(token)) {
      return value;
    }
  }
  return std::nullopt;
}

std::optional<double> file_mtime_seconds(const fs::path& path) {
#ifndef _WIN32
  struct stat info {};
  if (::stat(path.c_str(), &info) != 0) {
    return std::nullopt;
  }
  // CPython: st_mtime = sec + nsec * 1e-9
  return static_cast<double>(info.st_mtim.tv_sec) + static_cast<double>(info.st_mtim.tv_nsec) * 1e-9;
#else
  std::error_code error;
  const auto time = fs::last_write_time(path, error);
  if (error) {
    return std::nullopt;
  }
  const auto system = std::chrono::clock_cast<std::chrono::system_clock>(time);
  return std::chrono::duration<double>(system.time_since_epoch()).count();
#endif
}

struct ParsedCasesTable {
  std::vector<std::string> columns;
  std::vector<std::vector<std::string>> rows;
};

std::string read_text_file(const fs::path& path) {
  std::ifstream stream(path, std::ios::binary);
  if (!stream) {
    throw std::runtime_error(fmt::format("cannot open {}", path.string()));
  }
  return {std::istreambuf_iterator<char>(stream), std::istreambuf_iterator<char>()};
}

std::string_view trim(std::string_view text) {
  std::size_t begin = 0;
  std::size_t end = text.size();
  while (begin < end && is_space(text[begin])) {
    ++begin;
  }
  while (end > begin && is_space(text[end - 1])) {
    --end;
  }
  return text.substr(begin, end - begin);
}

// Records of pandas.read_csv(sep=delimiter, comment="#"): quotes honoured at
// field starts, "#" outside quotes ends the line, blank lines skipped.
std::vector<std::vector<std::string>> tokenize_delimited(std::string_view text, char delimiter) {
  std::vector<std::vector<std::string>> records;
  std::vector<std::string> record;
  std::string field;
  bool in_quotes = false;
  bool field_started = false;
  bool in_comment = false;

  auto end_record = [&] {
    record.push_back(std::move(field));
    field.clear();
    const bool blank = record.size() == 1 && record.front().empty() && !field_started;
    if (!blank) {
      records.push_back(std::move(record));
    }
    record.clear();
    field_started = false;
    in_comment = false;
  };

  for (std::size_t i = 0; i < text.size(); ++i) {
    const char c = text[i];
    if (in_comment) {
      if (c == '\n') {
        end_record();
      }
      continue;
    }
    if (in_quotes) {
      if (c == '"') {
        if (i + 1 < text.size() && text[i + 1] == '"') {
          field.push_back('"');
          ++i;
        } else {
          in_quotes = false;
        }
      } else {
        field.push_back(c);
      }
      continue;
    }
    if (c == '"' && field.empty()) {
      in_quotes = true;
      field_started = true;
    } else if (c == '#') {
      in_comment = true;
    } else if (c == delimiter) {
      record.push_back(std::move(field));
      field.clear();
      field_started = true;
    } else if (c == '\r') {
      if (i + 1 < text.size() && text[i + 1] == '\n') {
        ++i;
      }
      end_record();
    } else if (c == '\n') {
      end_record();
    } else {
      field.push_back(c);
      field_started = true;
    }
  }
  if (field_started || !field.empty() || !record.empty()) {
    end_record();
  }
  return records;
}

// Records of pandas.read_csv(sep=r"\s+", engine="python", comment="#").
std::vector<std::vector<std::string>> tokenize_whitespace(std::string_view text) {
  std::vector<std::vector<std::string>> records;
  for (std::string_view line : split_keep_empty(text, '\n')) {
    if (const auto hash = line.find('#'); hash != std::string_view::npos) {
      line = line.substr(0, hash);
    }
    line = trim(line);
    if (line.empty()) {
      continue;
    }
    std::vector<std::string> record;
    std::size_t i = 0;
    while (i < line.size()) {
      while (i < line.size() && is_space(line[i])) {
        ++i;
      }
      const std::size_t start = i;
      while (i < line.size() && !is_space(line[i])) {
        ++i;
      }
      if (i > start) {
        record.emplace_back(line.substr(start, i - start));
      }
    }
    records.push_back(std::move(record));
  }
  return records;
}

// campaign.py:_read_cases_full_dataframe for one separator attempt: dtype=str,
// NA fields -> "", column names and cells stripped, at least two columns.
std::optional<ParsedCasesTable> parse_cases_attempt(std::string_view text, char delimiter, std::string& error) {
  if (text.size() >= 3 && text.substr(0, 3) == "\xEF\xBB\xBF") {
    text.remove_prefix(3);
  }
  auto records = delimiter == ' ' ? tokenize_whitespace(text) : tokenize_delimited(text, delimiter);
  if (records.empty()) {
    error = "No columns to parse from file";
    return std::nullopt;
  }
  ParsedCasesTable table;
  for (const auto& name : records.front()) {
    std::string_view column = trim(name);
    while (column.substr(0, 3) == "\xEF\xBB\xBF") {
      column.remove_prefix(3);
    }
    table.columns.emplace_back(column);
  }
  for (std::size_t line = 1; line < records.size(); ++line) {
    auto& record = records[line];
    if (record.size() > table.columns.size()) {
      error = fmt::format("Error tokenizing data. Expected {} fields in line {}, saw {}", table.columns.size(),
                          line + 1, record.size());
      return std::nullopt;
    }
    record.resize(table.columns.size());
    std::vector<std::string> cells;
    for (auto& cell : record) {
      cells.emplace_back(table::is_pandas_na(cell) ? std::string_view() : trim(cell));
    }
    table.rows.push_back(std::move(cells));
  }
  if (table.columns.size() < 2) {
    error = fmt::format("parsed only one column: {}", table::python_list_repr(table.columns));
    return std::nullopt;
  }
  return table;
}

// campaign.py:_pick_column ({c.lower(): c} keeps the last duplicate).
std::optional<std::size_t> pick_column(const std::vector<std::string>& columns,
                                       std::initializer_list<std::string_view> candidates) {
  for (std::string_view candidate : candidates) {
    std::optional<std::size_t> found;
    for (std::size_t i = 0; i < columns.size(); ++i) {
      if (to_lower(columns[i]) == to_lower(candidate)) {
        found = i;
      }
    }
    if (found) {
      return found;
    }
  }
  return std::nullopt;
}

std::string age_cell(const std::optional<double>& age) { return age ? fmt::format("{:.3f}", *age) : std::string(); }

table::Cell token_cell(const Token& token) { return token ? table::Cell{*token} : table::Cell{}; }

}  // namespace

const char* case_type_name(CaseType type) noexcept {
  switch (type) {
    case CaseType::Channel: return "channel";
    case CaseType::Uniform: return "uniform";
    case CaseType::Vacuum: return "vacuum";
  }
  return "channel";
}

std::string TripletInfo::label() const {
  const auto& [fnum, plateau, focus, density] = key;
  const Token diameter = channel ? channel->tokens.diameter : std::nullopt;
  std::vector<std::string> parts;
  if (fnum) {
    parts.push_back("f" + *fnum);
  }
  if (density) {
    parts.push_back("n" + *density);
  }
  if (plateau) {
    parts.push_back("L" + *plateau + "mm");
  }
  if (diameter) {
    parts.push_back("d" + *diameter + "um");
  }
  if (focus) {
    parts.push_back("foc" + *focus);
  }
  std::string label;
  for (const auto& part : parts) {
    label += (label.empty() ? "" : "_") + part;
  }
  return label;
}

std::optional<CaseType> infer_case_type(std::string_view case_id) {
  const std::string name = to_lower(case_id);
  // re.split(r"[_\-/\\]+", name)
  std::vector<std::string> tokens;
  std::string current;
  for (char c : name) {
    if (c == '_' || c == '-' || c == '/' || c == '\\') {
      tokens.push_back(std::move(current));
      current.clear();
    } else {
      current.push_back(c);
    }
  }
  tokens.push_back(std::move(current));
  auto has = [&](std::string_view word) { return std::find(tokens.begin(), tokens.end(), word) != tokens.end(); };
  if (has("chan") || has("channel")) {
    return CaseType::Channel;
  }
  if (has("uni") || has("uniform")) {
    return CaseType::Uniform;
  }
  if (has("vac") || has("vacuum")) {
    return CaseType::Vacuum;
  }
  return std::nullopt;
}

CaseTokens parse_case_tokens(std::string_view case_id) {
  CaseTokens tokens;
  tokens.laser_case = first_token(case_id, [](std::string_view t) -> Token {
    if (t.size() >= 2 && ascii_lower(t[0]) == 'f' && all_digits(t.substr(1))) {
      return normalize_token(t.substr(1));
    }
    return std::nullopt;
  });
  tokens.density = first_token(case_id, [](std::string_view t) -> Token {
    if (t.size() >= 2 && ascii_lower(t[0]) == 'n') {
      return normalize_token(t.substr(1));
    }
    return std::nullopt;
  });
  tokens.ref_density = first_token(case_id, [](std::string_view t) -> Token {
    if (t.size() >= 5 && istarts_with(t, "refn")) {
      return normalize_token(t.substr(4));
    }
    return std::nullopt;
  });
  tokens.plateau = first_token(case_id, [](std::string_view t) -> Token {
    if (t.size() >= 4 && ascii_lower(t[0]) == 'l' && iends_with(t, "mm") && is_decimal_number(t.substr(1, t.size() - 3))) {
      return normalize_token(t.substr(1, t.size() - 3));
    }
    return std::nullopt;
  });
  tokens.focus = first_token(case_id, [](std::string_view t) -> Token {
    if (t.size() < 6 || !istarts_with(t, "foc")) {
      return std::nullopt;
    }
    const std::string_view unit = t.substr(t.size() - 2);
    if (!iequals(unit, "um") && !iequals(unit, "mm")) {
      return std::nullopt;
    }
    const std::string_view body = t.substr(3, t.size() - 5);
    const bool prefixed = !body.empty() && (ascii_lower(body[0]) == 'm' || ascii_lower(body[0]) == 'p') &&
                          is_decimal_number(body.substr(1));
    if (!prefixed && !is_decimal_number(body)) {
      return std::nullopt;
    }
    return normalize_token(body) + normalize_token(unit);
  });
  tokens.diameter = first_token(case_id, [](std::string_view t) -> Token {
    if (t.size() >= 4 && ascii_lower(t[0]) == 'd' && iends_with(t, "um") && is_decimal_number(t.substr(1, t.size() - 3))) {
      return normalize_token(t.substr(1, t.size() - 3));
    }
    return std::nullopt;
  });
  return tokens;
}

fs::path resolve_field_diag_dir(const fs::path& case_dir, bool require_exists) {
  const fs::path diags = case_dir / "diags";
  const std::array<fs::path, 2> candidates{diags / "fields", diags / "diag1"};
  for (const auto& candidate : candidates) {
    std::error_code error;
    if (fs::is_directory(candidate, error)) {
      return candidate;
    }
  }
  if (!require_exists) {
    return candidates[0];
  }
  throw std::runtime_error(fmt::format(
      "No supported WarpX field diagnostic directory found for case {}. Expected one of: {}, {}",
      python_path_string(case_dir), python_path_string(candidates[0]), python_path_string(candidates[1])));
}

fs::path resolve_particle_diag_dir(const fs::path& case_dir, const std::string& species_name,
                                   const std::string& particle_diag_name) {
  const fs::path diags = case_dir / "diags";
  std::error_code error;
  if (!particle_diag_name.empty() && particle_diag_name != "auto") {
    const fs::path candidate = diags / particle_diag_name;
    if (fs::is_directory(candidate, error)) {
      return candidate;
    }
    throw std::runtime_error(fmt::format("Requested particle diagnostic directory does not exist: {}",
                                         python_path_string(candidate)));
  }

  std::vector<std::string> names;
  if (species_name == "electrons") {
    names = {"plasma_electrons",    "plasma_electrons/openpmd", "electron_particles/openpmd",
             "electron_particles", "electrons/openpmd",        "electrons"};
  } else if (species_name == "plasma_electrons") {
    names = {"plasma_electrons", "plasma_electrons/openpmd"};
  } else if (species_name == "ionized_electrons") {
    names = {"ionized_electrons", "ionized_electrons/openpmd"};
  } else {
    names = {species_name, species_name + "/openpmd"};
  }
  std::string shown;
  for (const auto& name : names) {
    const fs::path candidate = diags / name;
    if (fs::is_directory(candidate, error)) {
      return candidate;
    }
    shown += (shown.empty() ? "" : ", ") + python_path_string(candidate);
  }
  throw std::runtime_error(fmt::format(
      "No supported particle diagnostic directory found for species {} in case {}. Expected one of: {}",
      table::py_str_repr(species_name), python_path_string(case_dir), shown));
}

H5Scan scan_h5_files(const fs::path& diag_dir) {
  H5Scan scan;
  std::error_code error;
  if (!fs::is_directory(diag_dir, error)) {
    return scan;
  }
  fs::recursive_directory_iterator it(diag_dir, fs::directory_options::skip_permission_denied, error);
  for (; !error && it != fs::recursive_directory_iterator(); it.increment(error)) {
    const fs::path& path = it->path();
    const std::string name = path.filename().string();
    if (name.size() < 3 || name.compare(name.size() - 3, 3, ".h5") != 0) {
      continue;
    }
    std::error_code file_error;
    if (!fs::is_regular_file(path, file_error)) {
      continue;
    }
    ++scan.count;
    if (const auto mtime = file_mtime_seconds(path)) {
      scan.newest_mtime_s = scan.newest_mtime_s ? std::max(*scan.newest_mtime_s, *mtime) : *mtime;
    }
  }
  return scan;
}

double unix_time_now() {
#ifndef _WIN32
  timespec now{};
  clock_gettime(CLOCK_REALTIME, &now);
  return static_cast<double>(now.tv_sec) + static_cast<double>(now.tv_nsec) * 1e-9;
#else
  return std::chrono::duration<double>(std::chrono::system_clock::now().time_since_epoch()).count();
#endif
}

std::optional<double> newest_h5_age_min(const CaseInfo& info, double now_s) {
  if (!info.newest_h5_mtime_s) {
    return std::nullopt;
  }
  return std::max(0.0, (now_s - *info.newest_h5_mtime_s) / 60.0);
}

std::optional<CaseInfo> infer_case_info_from_dir(const fs::path& case_dir, const std::string& source) {
  const std::string case_id = fs::path(python_path_string(case_dir)).filename().string();
  const auto type = infer_case_type(case_id);
  if (!type) {
    return std::nullopt;
  }
  CaseInfo info;
  info.case_id = case_id;
  info.case_type = *type;
  info.case_dir = case_dir;
  info.diag_dir = resolve_field_diag_dir(case_dir, false);
  const auto scan = scan_h5_files(info.diag_dir);
  info.h5_count = scan.count;
  info.newest_h5_mtime_s = scan.newest_mtime_s;
  info.tokens = parse_case_tokens(case_id);
  info.source = source;
  return info;
}

std::vector<CaseInfo> load_cases_from_cases_full(const fs::path& campaign_root) {
  fs::path cases_path;
  for (const char* name : {"cases_full.tsv", "cases.tsv"}) {
    std::error_code error;
    if (fs::is_regular_file(campaign_root / name, error)) {
      cases_path = campaign_root / name;
      break;
    }
  }
  if (cases_path.empty()) {
    return {};
  }
  const std::string file_name = cases_path.filename().string();
  const bool optional_file = file_name == "cases.tsv";
  const std::string text = read_text_file(cases_path);

  std::optional<ParsedCasesTable> table;
  std::vector<std::string> errors;
  for (const auto& [label, delimiter] : {std::pair{"tab", '\t'}, std::pair{"whitespace", ' '}, std::pair{"comma", ','}}) {
    std::string error;
    table = parse_cases_attempt(text, delimiter, error);
    if (table) {
      break;
    }
    errors.push_back(fmt::format("{}: {}", label, error));
  }
  if (!table) {
    std::string joined;
    for (const auto& error : errors) {
      joined += (joined.empty() ? "" : " | ") + error;
    }
    throw std::runtime_error(
        fmt::format("Could not parse {} as a useful cases table. {}", python_path_string(cases_path), joined));
  }

  const auto& columns = table->columns;
  const auto case_col = pick_column(
      columns, {"case_name", "case_dir_name", "case_folder", "case", "name", "run_name", "run", "run_id", "case_id"});
  const auto path_col = pick_column(columns, {"case_dir", "dir", "directory", "path", "run_dir", "case_path"});
  const auto diag_col = pick_column(columns, {"diag", "diag_dir", "diag_path"});
  const auto type_col = pick_column(columns, {"case_type", "type", "profile", "plasma", "plasma_kind", "kind"});

  if (!case_col && !path_col) {
    if (optional_file) {
      return {};
    }
    throw std::runtime_error(fmt::format(
        "{} exists but does not contain a recognizable case/path column. Parsed columns: {}",
        python_path_string(cases_path), table::python_list_repr(columns)));
  }

  std::vector<CaseInfo> cases;
  for (const auto& row : table->rows) {
    const std::string raw_case = case_col ? row[*case_col] : std::string();
    const std::string raw_path = path_col ? row[*path_col] : std::string();
    const std::string raw_diag = diag_col ? row[*diag_col] : std::string();

    fs::path case_dir;
    if (!raw_path.empty()) {
      case_dir = fs::path(raw_path);
      if (!case_dir.is_absolute()) {
        case_dir = campaign_root / case_dir;
      }
    } else if (!raw_case.empty()) {
      case_dir = campaign_root / raw_case;
    } else {
      continue;
    }

    auto info = infer_case_info_from_dir(case_dir, file_name);
    if (!info) {
      continue;
    }
    if (type_col) {
      if (const auto explicit_type = infer_case_type(row[*type_col])) {
        info->case_type = *explicit_type;
      }
    }
    if (!raw_diag.empty()) {
      fs::path diag_dir(raw_diag);
      if (!diag_dir.is_absolute()) {
        diag_dir = campaign_root / diag_dir;
      }
      info->diag_dir = diag_dir;
      const auto scan = scan_h5_files(diag_dir);
      info->h5_count = scan.count;
      info->newest_h5_mtime_s = scan.newest_mtime_s;
    }
    cases.push_back(std::move(*info));
  }

  if (cases.empty()) {
    if (optional_file) {
      return {};
    }
    throw std::runtime_error(fmt::format(
        "{} was parsed, but no recognizable campaign cases were found. Check case names, case_dir paths, and "
        "type/profile columns.",
        python_path_string(cases_path)));
  }
  return cases;
}

std::vector<CaseInfo> discover_cases_by_name(const fs::path& campaign_root) {
  std::vector<fs::path> children;
  for (const auto& entry : fs::directory_iterator(campaign_root)) {
    children.push_back(entry.path());
  }
  std::sort(children.begin(), children.end());
  std::vector<CaseInfo> cases;
  for (const auto& child : children) {
    std::error_code error;
    if (!fs::is_directory(child, error)) {
      continue;
    }
    if (auto info = infer_case_info_from_dir(child, "name")) {
      cases.push_back(std::move(*info));
    }
  }
  return cases;
}

std::vector<CaseInfo> discover_cases(const fs::path& campaign_root) {
  auto cases = load_cases_from_cases_full(campaign_root);
  if (!cases.empty()) {
    return cases;
  }
  return discover_cases_by_name(campaign_root);
}

std::vector<TripletInfo> build_triplets(const std::vector<CaseInfo>& cases) {
  std::map<CaseInfo::FullKey, const CaseInfo*> uniform_by_key;
  std::map<CaseInfo::FullKey, const CaseInfo*> vacuum_by_full_key;
  std::map<CaseInfo::BaseKey, const CaseInfo*> vacuum_by_base_key;
  for (const auto& info : cases) {
    if (info.case_type == CaseType::Uniform) {
      uniform_by_key[info.full_key()] = &info;
    } else if (info.case_type == CaseType::Vacuum) {
      if (info.tokens.ref_density) {
        vacuum_by_full_key[info.full_key()] = &info;
      }
      vacuum_by_base_key.emplace(info.base_key(), &info);  // setdefault: first vacuum wins
    }
  }

  auto lookup = [](const auto& map, const auto& key) -> std::optional<CaseInfo> {
    const auto it = map.find(key);
    return it == map.end() ? std::nullopt : std::optional<CaseInfo>(*it->second);
  };
  auto find_vacuum = [&](const CaseInfo& info) {
    auto vacuum = lookup(vacuum_by_full_key, info.full_key());
    return vacuum ? vacuum : lookup(vacuum_by_base_key, info.base_key());
  };

  std::vector<TripletInfo> triplets;
  std::set<CaseInfo::FullKey> used_keys;
  for (const auto& info : cases) {
    if (info.case_type != CaseType::Channel) {
      continue;
    }
    const auto key = info.full_key();
    triplets.push_back(TripletInfo{key, info, lookup(uniform_by_key, key), find_vacuum(info)});
    used_keys.insert(key);
  }
  for (const auto& info : cases) {
    if (info.case_type != CaseType::Uniform) {
      continue;
    }
    const auto key = info.full_key();
    if (used_keys.contains(key)) {
      continue;
    }
    triplets.push_back(TripletInfo{key, std::nullopt, info, find_vacuum(info)});
    used_keys.insert(key);
  }
  std::stable_sort(triplets.begin(), triplets.end(),
                   [](const TripletInfo& a, const TripletInfo& b) { return a.label() < b.label(); });
  return triplets;
}

bool case_has_min_h5(const CaseInfo& info, std::int64_t min_h5) { return info.h5_count >= min_h5; }

bool case_has_stable_h5(const CaseInfo& info, double min_last_h5_age_min, double now_s) {
  if (min_last_h5_age_min <= 0.0) {
    return true;
  }
  const auto age = newest_h5_age_min(info, now_s);
  return age && *age >= min_last_h5_age_min;
}

bool case_is_ready(const CaseInfo& info, std::int64_t min_h5, double min_last_h5_age_min, double now_s) {
  return case_has_min_h5(info, min_h5) && case_has_stable_h5(info, min_last_h5_age_min, now_s);
}

bool triplet_ready_min_h5(const TripletInfo& triplet, std::int64_t min_h5) {
  return triplet.complete() && case_has_min_h5(*triplet.channel, min_h5) && case_has_min_h5(*triplet.uniform, min_h5) &&
         case_has_min_h5(*triplet.vacuum, min_h5);
}

bool triplet_is_ready(const TripletInfo& triplet, std::int64_t min_h5, double min_last_h5_age_min, double now_s) {
  return triplet.complete() && case_is_ready(*triplet.channel, min_h5, min_last_h5_age_min, now_s) &&
         case_is_ready(*triplet.uniform, min_h5, min_last_h5_age_min, now_s) &&
         case_is_ready(*triplet.vacuum, min_h5, min_last_h5_age_min, now_s);
}

CampaignReportPaths write_campaign_report(const std::vector<CaseInfo>& cases, const std::vector<TripletInfo>& triplets,
                                          const fs::path& outdir, std::int64_t min_h5, double min_last_h5_age_min,
                                          double now_s) {
  fs::create_directories(outdir);
  const CampaignReportPaths paths{outdir / "campaign_cases.csv", outdir / "campaign_triplets.csv",
                                  outdir / "campaign_insufficient_h5.csv", outdir / "campaign_unstable_h5.csv"};
  const auto dialect = table::CsvDialect::python_csv();
  using table::Cell;

  {
    std::string out;
    const std::vector<std::string> header{"case_id",     "case_type", "case_dir",       "diag_dir",
                                          "h5_count",    "newest_h5_age_min", "ready_min_h5", "stable_min_age",
                                          "ready_for_analysis", "laser_case", "density", "ref_density",
                                          "plateau",     "focus",     "diameter",       "source"};
    table::append_csv_header(out, header, dialect);
    for (const auto& info : cases) {
      const bool ready_min = case_has_min_h5(info, min_h5);
      const bool stable = case_has_stable_h5(info, min_last_h5_age_min, now_s);
      const std::vector<Cell> row{info.case_id,
                                  std::string(case_type_name(info.case_type)),
                                  python_path_string(info.case_dir),
                                  python_path_string(info.diag_dir),
                                  info.h5_count,
                                  age_cell(newest_h5_age_min(info, now_s)),
                                  ready_min,
                                  stable,
                                  ready_min && stable,
                                  token_cell(info.tokens.laser_case),
                                  token_cell(info.tokens.density),
                                  token_cell(info.tokens.ref_density),
                                  token_cell(info.tokens.plateau),
                                  token_cell(info.tokens.focus),
                                  token_cell(info.tokens.diameter),
                                  info.source};
      table::append_csv_record(out, row, dialect);
    }
    table::write_file_atomically(paths.cases, out);
  }

  {
    std::string out;
    const std::vector<std::string> header{"label",       "complete",   "ready_min_h5", "stable_min_age",
                                          "ready_for_analysis", "channel", "uniform",     "vacuum",
                                          "channel_h5",  "uniform_h5", "vacuum_h5",    "channel_newest_h5_age_min",
                                          "uniform_newest_h5_age_min", "vacuum_newest_h5_age_min",
                                          "min_last_h5_age_min"};
    table::append_csv_header(out, header, dialect);
    auto id = [](const std::optional<CaseInfo>& info) { return info ? Cell{info->case_id} : Cell{std::string()}; };
    auto h5 = [](const std::optional<CaseInfo>& info) { return info ? Cell{info->h5_count} : Cell{std::string()}; };
    auto age = [&](const std::optional<CaseInfo>& info) {
      return Cell{info ? age_cell(newest_h5_age_min(*info, now_s)) : std::string()};
    };
    for (const auto& triplet : triplets) {
      const bool ready_min = triplet_ready_min_h5(triplet, min_h5);
      const bool stable = triplet.complete() && case_has_stable_h5(*triplet.channel, min_last_h5_age_min, now_s) &&
                          case_has_stable_h5(*triplet.uniform, min_last_h5_age_min, now_s) &&
                          case_has_stable_h5(*triplet.vacuum, min_last_h5_age_min, now_s);
      const std::vector<Cell> row{triplet.label(), triplet.complete(), ready_min, stable, ready_min && stable,
                                  id(triplet.channel), id(triplet.uniform), id(triplet.vacuum),
                                  h5(triplet.channel), h5(triplet.uniform), h5(triplet.vacuum),
                                  age(triplet.channel), age(triplet.uniform), age(triplet.vacuum),
                                  min_last_h5_age_min};
      table::append_csv_record(out, row, dialect);
    }
    table::write_file_atomically(paths.triplets, out);
  }

  {
    std::string out;
    const std::vector<std::string> header{"case_id", "case_type", "diag_dir", "h5_count", "min_h5"};
    table::append_csv_header(out, header, dialect);
    for (const auto& info : cases) {
      if (info.h5_count >= min_h5) {
        continue;
      }
      const std::vector<Cell> row{info.case_id, std::string(case_type_name(info.case_type)),
                                  python_path_string(info.diag_dir), info.h5_count, min_h5};
      table::append_csv_record(out, row, dialect);
    }
    table::write_file_atomically(paths.insufficient_h5, out);
  }

  {
    std::string out;
    const std::vector<std::string> header{"case_id",  "case_type",         "diag_dir",
                                          "h5_count", "newest_h5_age_min", "min_last_h5_age_min"};
    table::append_csv_header(out, header, dialect);
    for (const auto& info : cases) {
      if (!case_has_min_h5(info, min_h5) || case_has_stable_h5(info, min_last_h5_age_min, now_s)) {
        continue;
      }
      const std::vector<Cell> row{info.case_id, std::string(case_type_name(info.case_type)),
                                  python_path_string(info.diag_dir), info.h5_count,
                                  age_cell(newest_h5_age_min(info, now_s)), min_last_h5_age_min};
      table::append_csv_record(out, row, dialect);
    }
    table::write_file_atomically(paths.unstable_h5, out);
  }
  return paths;
}

}  // namespace guiding::campaign

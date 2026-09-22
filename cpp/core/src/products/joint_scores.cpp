#include "guiding/products/joint_scores.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <limits>
#include <functional>
#include <map>
#include <numeric>
#include <optional>
#include <stdexcept>

#include <fmt/format.h>

#include "guiding/numeric/npcompat.hpp"
#include "guiding/table/csv.hpp"
#include "guiding/table/py_format.hpp"

namespace guiding::products {
namespace {

namespace fs = std::filesystem;
using table::Frame;

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
const std::vector<std::string> kPairKeys{"channel_case_id", "uniform_case_id"};

using RowRef = std::optional<std::size_t>;

std::string key_text(const Frame& frame, const std::string& column, std::size_t row) {
  const auto& values = frame.column(column);
  if (const auto* texts = std::get_if<Frame::Strings>(&values)) {
    return (*texts)[row];
  }
  if (const auto* integers = std::get_if<Frame::Integers>(&values)) {
    return std::to_string((*integers)[row]);
  }
  return table::py_float_repr(std::get<Frame::Doubles>(values)[row]);
}

std::vector<std::string> row_key(const Frame& frame, const std::vector<std::string>& keys, std::size_t row) {
  std::vector<std::string> key;
  for (const auto& column : keys) {
    key.push_back(key_text(frame, column, row));
  }
  return key;
}

// Column values at the given rows; missing rows become NaN (ints turn float) or empty text.
Frame::Column take_with_missing(const Frame::Column& column, const std::vector<RowRef>& rows) {
  const bool any_missing = std::any_of(rows.begin(), rows.end(), [](const RowRef& r) { return !r; });
  if (const auto* integers = std::get_if<Frame::Integers>(&column)) {
    if (!any_missing) {
      Frame::Integers out;
      for (const auto& r : rows) {
        out.push_back((*integers)[*r]);
      }
      return out;
    }
    Frame::Doubles out;
    for (const auto& r : rows) {
      out.push_back(r ? static_cast<double>((*integers)[*r]) : kNaN);
    }
    return out;
  }
  if (const auto* reals = std::get_if<Frame::Doubles>(&column)) {
    Frame::Doubles out;
    for (const auto& r : rows) {
      out.push_back(r ? (*reals)[*r] : kNaN);
    }
    return out;
  }
  const auto& texts = std::get<Frame::Strings>(column);
  Frame::Strings out;
  for (const auto& r : rows) {
    out.push_back(r ? texts[*r] : std::string());
  }
  return out;
}

// Value of a joined row as the reference sees it through Series.get.
struct CellView {
  enum class Kind { Missing, Number, Text } kind = Kind::Missing;
  double number = kNaN;
  std::string text;

  // str(value).strip().lower()
  [[nodiscard]] std::string lowered() const {
    std::string out = kind == Kind::Text ? text : kind == Kind::Number ? (std::isnan(number) ? "nan" : table::py_float_repr(number)) : "";
    const auto first = out.find_first_not_of(" \t\n\r\v\f");
    out = first == std::string::npos ? std::string() : out.substr(first, out.find_last_not_of(" \t\n\r\v\f") - first + 1);
    std::transform(out.begin(), out.end(), out.begin(), [](unsigned char c) { return std::tolower(c); });
    return out;
  }
  // _finite_float(value, default)
  [[nodiscard]] double finite(double fallback = kNaN) const {
    double value = kNaN;
    if (kind == Kind::Number) {
      value = number;
    } else if (kind == Kind::Text) {
      value = table::parse_py_float(text).value_or(kNaN);
      if (text.empty()) {
        return fallback;  // NaN read back as an empty cell
      }
    } else {
      return fallback;
    }
    return std::isfinite(value) ? value : fallback;
  }
};

CellView cell(const Frame& frame, const std::string& column, std::size_t row) {
  CellView view;
  if (!frame.has(column)) {
    return view;
  }
  const auto& values = frame.column(column);
  if (const auto* texts = std::get_if<Frame::Strings>(&values)) {
    view.kind = CellView::Kind::Text;
    view.text = (*texts)[row];
  } else if (const auto* integers = std::get_if<Frame::Integers>(&values)) {
    view.kind = CellView::Kind::Number;
    view.number = static_cast<double>((*integers)[row]);
  } else {
    view.kind = CellView::Kind::Number;
    view.number = std::get<Frame::Doubles>(values)[row];
  }
  return view;
}

const std::vector<std::string> kValidBuckets{"positive", "neutral", "negative", "failed"};

std::string bucket_from_factor(const CellView& status, const CellView& factor) {
  if (status.lowered() != "ok") {
    return "failed";
  }
  const double x = factor.finite();
  if (!std::isfinite(x)) {
    return "failed";
  }
  return x > 0.0 ? "positive" : x < 0.0 ? "negative" : "neutral";
}

std::string existing_or_factor_bucket(const Frame& joined, std::size_t row, const std::string& status_column,
                                      const std::string& bucket_column, const std::string& factor_column) {
  const CellView status = cell(joined, status_column, row);
  const std::string existing = cell(joined, bucket_column, row).lowered();
  if (std::find(kValidBuckets.begin(), kValidBuckets.end(), existing) != kValidBuckets.end()) {
    if (status.lowered() == "ok" || existing == "failed") {
      return existing;
    }
  }
  return bucket_from_factor(status, cell(joined, factor_column, row));
}

double nonnegative_finite(const CellView& value) {
  const double x = value.finite(0.0);
  if (!std::isfinite(x)) {
    return 0.0;
  }
  return 0.0 > x ? 0.0 : x;
}

std::vector<double> average_ranks(std::span<const double> values) {
  std::vector<std::size_t> order(values.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return values[a] < values[b]; });
  std::vector<double> ranks(values.size());
  std::size_t i = 0;
  while (i < order.size()) {
    std::size_t j = i;
    while (j + 1 < order.size() && values[order[j + 1]] == values[order[i]]) {
      ++j;
    }
    // scipy rankdata(method="average"): 0.5 * (first + last) of the 1-based ranks
    const double rank = 0.5 * (static_cast<double>(i + 1) + static_cast<double>(j + 1));
    for (std::size_t k = i; k <= j; ++k) {
      ranks[order[k]] = rank;
    }
    i = j + 1;
  }
  return ranks;
}

std::string frame_csv(const Frame& frame) { return frame.names().empty() ? std::string("\n") : frame.to_pandas_csv(); }

}  // namespace

JoinHow parse_join_how(std::string_view name) {
  if (name == "inner") {
    return JoinHow::Inner;
  }
  if (name == "left") {
    return JoinHow::Left;
  }
  if (name == "right") {
    return JoinHow::Right;
  }
  if (name == "outer") {
    return JoinHow::Outer;
  }
  throw std::invalid_argument(fmt::format("Unknown join: {}", name));
}

Frame merge_frames(const Frame& left, const Frame& right, const std::vector<std::string>& keys, JoinHow how) {
  std::map<std::vector<std::string>, std::vector<std::size_t>> left_rows;
  std::map<std::vector<std::string>, std::vector<std::size_t>> right_rows;
  std::vector<std::vector<std::string>> left_keys(left.row_count());
  std::vector<std::vector<std::string>> right_keys(right.row_count());
  for (std::size_t i = 0; i < left.row_count(); ++i) {
    left_keys[i] = row_key(left, keys, i);
    left_rows[left_keys[i]].push_back(i);
  }
  for (std::size_t i = 0; i < right.row_count(); ++i) {
    right_keys[i] = row_key(right, keys, i);
    right_rows[right_keys[i]].push_back(i);
  }

  std::vector<RowRef> from_left;
  std::vector<RowRef> from_right;
  const auto emit = [&](RowRef l, RowRef r) {
    from_left.push_back(l);
    from_right.push_back(r);
  };
  switch (how) {
    case JoinHow::Inner:
    case JoinHow::Left:
      for (std::size_t l = 0; l < left.row_count(); ++l) {
        const auto match = right_rows.find(left_keys[l]);
        if (match != right_rows.end()) {
          for (std::size_t r : match->second) {
            emit(l, r);
          }
        } else if (how == JoinHow::Left) {
          emit(l, std::nullopt);
        }
      }
      break;
    case JoinHow::Right:
      for (std::size_t r = 0; r < right.row_count(); ++r) {
        const auto match = left_rows.find(right_keys[r]);
        if (match != left_rows.end()) {
          for (std::size_t l : match->second) {
            emit(l, r);
          }
        } else {
          emit(std::nullopt, r);
        }
      }
      break;
    case JoinHow::Outer: {
      std::vector<std::vector<std::string>> all_keys;
      for (const auto& [key, rows] : left_rows) {
        all_keys.push_back(key);
      }
      for (const auto& [key, rows] : right_rows) {
        if (!left_rows.contains(key)) {
          all_keys.push_back(key);
        }
      }
      std::sort(all_keys.begin(), all_keys.end());
      for (const auto& key : all_keys) {
        const auto l = left_rows.find(key);
        const auto r = right_rows.find(key);
        if (l != left_rows.end() && r != right_rows.end()) {
          for (std::size_t li : l->second) {
            for (std::size_t ri : r->second) {
              emit(li, ri);
            }
          }
        } else if (l != left_rows.end()) {
          for (std::size_t li : l->second) {
            emit(li, std::nullopt);
          }
        } else {
          for (std::size_t ri : r->second) {
            emit(std::nullopt, ri);
          }
        }
      }
      break;
    }
  }

  Frame out;
  for (const auto& name : left.names()) {
    const bool is_key = std::find(keys.begin(), keys.end(), name) != keys.end();
    if (!is_key) {
      out.set(name, take_with_missing(left.column(name), from_left));
      continue;
    }
    // Key values come from whichever side has the row.
    Frame::Strings values;
    for (std::size_t i = 0; i < from_left.size(); ++i) {
      values.push_back(from_left[i] ? key_text(left, name, *from_left[i]) : key_text(right, name, *from_right[i]));
    }
    out.set(name, std::move(values));
  }
  for (const auto& name : right.names()) {
    if (std::find(keys.begin(), keys.end(), name) == keys.end()) {
      out.set(name, take_with_missing(right.column(name), from_right));
    }
  }
  return out;
}

Frame load_and_join_guiding_beamlike(const fs::path& triplet_scores_csv, const fs::path& beamlike_pair_scores_csv,
                                     JoinHow how) {
  Frame guiding = table::frame_from_csv(table::read_csv_file(triplet_scores_csv));
  Frame beam = table::frame_from_csv(table::read_csv_file(beamlike_pair_scores_csv));
  for (const auto& [label, frame] : {std::pair<const char*, const Frame*>{"guiding", &guiding}, {"beamlike", &beam}}) {
    std::vector<std::string> missing;
    for (const auto& key : kPairKeys) {
      if (!frame->has(key)) {
        missing.push_back(key);
      }
    }
    if (!missing.empty()) {
      throw std::invalid_argument(fmt::format("{} CSV missing join keys: {}", label, table::python_list_repr(missing)));
    }
  }
  for (auto* frame : {&guiding, &beam}) {
    const std::string prefix = frame == &guiding ? "guiding_" : "beam_";
    const auto names = frame->names();
    for (const auto& name : names) {
      if (std::find(kPairKeys.begin(), kPairKeys.end(), name) == kPairKeys.end()) {
        frame->rename(name, prefix + name);
      }
    }
  }

  Frame joined = merge_frames(guiding, beam, kPairKeys, how);
  const std::size_t n = joined.row_count();
  if (n == 0) {
    return joined;
  }

  Frame::Strings guiding_buckets, beam_buckets, transverse_buckets, joint_buckets, triple_buckets, joint_statuses,
      triple_statuses;
  Frame::Doubles alignment, triple_alignment, joint_positive, triple_positive;
  for (std::size_t row = 0; row < n; ++row) {
    const std::string guiding_bucket =
        bucket_from_factor(cell(joined, "guiding_status", row), cell(joined, "guiding_reference_factor", row));
    const std::string beam_bucket = existing_or_factor_bucket(joined, row, "beam_status", "beam_comparison_bucket",
                                                              "beam_beamlike_reference_factor");
    const std::string transverse_bucket =
        existing_or_factor_bucket(joined, row, "beam_transverse_comparison_status",
                                  "beam_transverse_comparison_bucket", "beam_transverse_reference_factor");
    const std::string joint_bucket = guiding_bucket == "failed" || beam_bucket == "failed"
                                         ? "failed"
                                         : fmt::format("guiding_{}__beam_{}", guiding_bucket, beam_bucket);
    const std::string triple_bucket =
        guiding_bucket == "failed" || beam_bucket == "failed" || transverse_bucket == "failed"
            ? "failed"
            : fmt::format("guiding_{}__beam_{}__transverse_{}", guiding_bucket, beam_bucket, transverse_bucket);

    const double guiding_factor = cell(joined, "guiding_reference_factor", row).finite();
    const double beam_factor = cell(joined, "beam_beamlike_reference_factor", row).finite();
    const double transverse_factor = cell(joined, "beam_transverse_reference_factor", row).finite();
    const bool finite_pair = std::isfinite(guiding_factor) && std::isfinite(beam_factor);

    const double guiding_final = nonnegative_finite(cell(joined, "guiding_final_score", row));
    const double beam_gain = nonnegative_finite(cell(joined, "beam_beamlike_gain_score", row));
    const double transverse_gain = nonnegative_finite(cell(joined, "beam_transverse_gain_score", row));
    const double triple_product = guiding_final * beam_gain * transverse_gain;

    guiding_buckets.push_back(guiding_bucket);
    beam_buckets.push_back(beam_bucket);
    transverse_buckets.push_back(transverse_bucket);
    joint_buckets.push_back(joint_bucket);
    triple_buckets.push_back(triple_bucket);
    joint_statuses.emplace_back(joint_bucket != "failed" ? "ok" : "failed");
    triple_statuses.emplace_back(triple_bucket != "failed" ? "ok" : "failed");
    alignment.push_back(finite_pair ? guiding_factor * beam_factor : kNaN);
    triple_alignment.push_back(finite_pair && std::isfinite(transverse_factor)
                                   ? guiding_factor * beam_factor * transverse_factor
                                   : kNaN);
    joint_positive.push_back(std::sqrt(guiding_final * beam_gain));
    triple_positive.push_back(triple_product > 0.0 ? np::c_pow(triple_product, 1.0 / 3.0) : 0.0);
  }
  joined.set("guiding_bucket", std::move(guiding_buckets));
  joined.set("beam_bucket", std::move(beam_buckets));
  joined.set("transverse_bucket", std::move(transverse_buckets));
  joined.set("joint_bucket", std::move(joint_buckets));
  joined.set("triple_bucket", std::move(triple_buckets));
  joined.set("joint_status", std::move(joint_statuses));
  joined.set("triple_status", std::move(triple_statuses));
  joined.set("guiding_beam_alignment_factor", std::move(alignment));
  joined.set("guiding_beam_transverse_alignment_factor", std::move(triple_alignment));
  joined.set("joint_positive_score", std::move(joint_positive));
  joined.set("triple_positive_score", std::move(triple_positive));
  return joined;
}

double pearson(std::span<const double> x, std::span<const double> y) {
  const std::size_t n = x.size();
  const double mean_x = np::pairwise_sum(x) / static_cast<double>(n);
  const double mean_y = np::pairwise_sum(y) / static_cast<double>(n);
  double sxx = 0.0;
  double syy = 0.0;
  double sxy = 0.0;
  for (std::size_t i = 0; i < n; ++i) {
    const double dx = x[i] - mean_x;
    const double dy = y[i] - mean_y;
    sxx += dx * dx;
    syy += dy * dy;
    sxy += dx * dy;
  }
  const double fact = 1.0 / static_cast<double>(n - 1);
  const double r = ((sxy * fact) / std::sqrt(sxx * fact)) / std::sqrt(syy * fact);
  return std::isnan(r) ? r : std::clamp(r, -1.0, 1.0);
}

double spearman(std::span<const double> x, std::span<const double> y) {
  const auto constant = [](std::span<const double> v) {
    return std::all_of(v.begin(), v.end(), [&](double value) { return value == v.front(); });
  };
  if (constant(x) || constant(y)) {
    return kNaN;
  }
  const auto rx = average_ranks(x);
  const auto ry = average_ranks(y);
  return pearson(rx, ry);
}

Frame compute_joint_correlations(const Frame& df) {
  struct Pair {
    const char* x;
    const char* y;
    const char* description;
  };
  static const std::vector<Pair> kPairs{
      {"guiding_final_score", "beam_beamlike_gain_score", "guiding final score vs beam gain score"},
      {"guiding_reference_factor", "beam_beamlike_reference_factor",
       "guiding reference factor vs beam reference factor"},
      {"guiding_score_channel", "beam_beamlike_score_channel", "absolute guiding score vs absolute beamlike score"},
      {"guiding_a0_exit_channel", "beam_E95_hot_MeV_channel", "a0 exit vs hot E95"},
      {"guiding_a0_exit_channel", "beam_charge_hot_pC_channel", "a0 exit vs hot charge"},
      {"guiding_waist_growth_channel", "beam_z_span_hot_mm_channel", "waist growth vs hot z span"},
      {"guiding_fraction_a0_beats_reference", "beam_beamlike_reference_factor",
       "fraction a0 beats reference vs beam reference factor"},
      {"guiding_final_score", "beam_transverse_gain_score", "guiding final score vs transverse gain score"},
      {"beam_beamlike_gain_score", "beam_transverse_gain_score", "beam gain score vs transverse gain score"},
      {"beam_beamlike_reference_factor", "beam_transverse_reference_factor",
       "beam reference factor vs transverse reference factor"},
      {"beam_E95_hot_MeV_channel", "beam_theta_rms_mrad_channel", "hot E95 vs transverse RMS divergence"},
      {"beam_E95_hot_MeV_channel", "beam_emit_geom_norm_mm_mrad_channel", "hot E95 vs geometric normalized emittance"},
      {"beam_charge_hot_pC_channel", "beam_theta_r_p95_mrad_channel", "hot charge vs transverse p95 divergence"},
  };
  std::vector<table::Record> rows;
  for (const auto& pair : kPairs) {
    if (!df.has(pair.x) || !df.has(pair.y)) {
      continue;
    }
    const auto x_all = df.doubles(pair.x);
    const auto y_all = df.doubles(pair.y);
    std::vector<double> x;
    std::vector<double> y;
    for (std::size_t i = 0; i < x_all.size(); ++i) {
      if (std::isfinite(x_all[i]) && std::isfinite(y_all[i])) {
        x.push_back(x_all[i]);
        y.push_back(y_all[i]);
      }
    }
    const bool enough = x.size() >= 3;
    rows.push_back(table::Record{{"x", std::string(pair.x)},
                                 {"y", std::string(pair.y)},
                                 {"description", std::string(pair.description)},
                                 {"n", static_cast<std::int64_t>(x.size())},
                                 {"pearson", enough ? pearson(x, y) : kNaN},
                                 {"spearman", enough ? spearman(x, y) : kNaN}});
  }
  return table::frame_from_records(rows);
}

Frame bucket_counts(const Frame& joined, const std::string& column) {
  Frame out;
  if (joined.row_count() == 0 || !joined.has(column)) {
    out.set(column, Frame::Strings{});
    out.set("count", Frame::Integers{});
    return out;
  }
  std::vector<std::string> values;
  std::vector<std::int64_t> counts;
  for (const auto& value : joined.strings(column)) {
    const auto it = std::find(values.begin(), values.end(), value);
    if (it == values.end()) {
      values.push_back(value);
      counts.push_back(1);
    } else {
      ++counts[static_cast<std::size_t>(it - values.begin())];
    }
  }
  std::vector<std::size_t> order(values.size());
  std::iota(order.begin(), order.end(), 0);
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) { return counts[a] > counts[b]; });
  Frame::Strings sorted_values;
  Frame::Integers sorted_counts;
  for (std::size_t i : order) {
    sorted_values.push_back(values[i]);
    sorted_counts.push_back(counts[i]);
  }
  out.set(column, std::move(sorted_values));
  out.set("count", std::move(sorted_counts));
  return out;
}

Frame sorted_subset(const Frame& joined, const std::vector<std::size_t>& rows, const std::string& column,
                    bool ascending, std::size_t top) {
  std::vector<std::size_t> order = rows;
  const auto values = joined.doubles(column);
  std::stable_sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
    if (std::isnan(values[a]) || std::isnan(values[b])) {
      return !std::isnan(values[a]) && std::isnan(values[b]);
    }
    return ascending ? values[a] < values[b] : values[a] > values[b];
  });
  if (order.size() > top) {
    order.resize(top);
  }
  return joined.take(order);
}

std::vector<JointOutput> write_joint_outputs(const fs::path& outdir, const Frame& joined, std::size_t top) {
  fs::create_directories(outdir);
  std::vector<JointOutput> paths{
      {"joined", outdir / "guiding_beamlike_joined.csv"},
      {"joined_triple", outdir / "guiding_beam_transverse_joined.csv"},
      {"bucket_counts", outdir / "guiding_beamlike_bucket_counts.csv"},
      {"triple_bucket_counts", outdir / "guiding_beam_transverse_bucket_counts.csv"},
      {"correlations", outdir / "guiding_beamlike_correlations.csv"},
      {"both_positive", outdir / "both_positive_guiding_beamlike.csv"},
      {"guiding_positive_beam_negative", outdir / "guiding_positive_beam_negative.csv"},
      {"guiding_positive_beam_neutral", outdir / "guiding_positive_beam_neutral.csv"},
      {"beam_positive_guiding_nonpositive", outdir / "beam_positive_guiding_nonpositive.csv"},
      {"both_negative", outdir / "both_negative_guiding_beamlike.csv"},
      {"triple_positive", outdir / "guiding_positive_beam_positive_transverse_positive.csv"},
      {"guiding_positive_beam_positive_transverse_negative",
       outdir / "guiding_positive_beam_positive_transverse_negative.csv"},
      {"guiding_negative_beam_positive_transverse_positive",
       outdir / "guiding_negative_beam_positive_transverse_positive.csv"},
      {"beam_positive_transverse_negative", outdir / "beam_positive_transverse_negative.csv"},
  };
  const auto path_of = [&](const std::string& key) {
    return std::find_if(paths.begin(), paths.end(), [&](const JointOutput& o) { return o.key == key; })->path;
  };
  table::write_file_atomically(path_of("joined"), frame_csv(joined));
  table::write_file_atomically(path_of("joined_triple"), frame_csv(joined));
  table::write_file_atomically(path_of("bucket_counts"), frame_csv(bucket_counts(joined, "joint_bucket")));
  table::write_file_atomically(path_of("triple_bucket_counts"), frame_csv(bucket_counts(joined, "triple_bucket")));
  table::write_file_atomically(path_of("correlations"), frame_csv(compute_joint_correlations(joined)));

  const auto write = [&](const std::string& key, const Frame& frame) {
    table::write_file_atomically(path_of(key), frame_csv(frame));
  };
  if (joined.row_count() == 0) {
    // joined.head(0): every subset keeps the header of an empty join.
    const Frame header_only = joined.take(std::vector<std::size_t>{});
    for (const auto& output : paths) {
      if (output.key != "joined" && output.key != "joined_triple" && output.key != "bucket_counts" &&
          output.key != "triple_bucket_counts" && output.key != "correlations") {
        table::write_file_atomically(output.path, frame_csv(header_only));
      }
    }
    return paths;
  }
  const auto select = [&](const std::function<bool(std::size_t)>& keep) {
    std::vector<std::size_t> rows;
    for (std::size_t i = 0; i < joined.row_count(); ++i) {
      if (keep(i)) {
        rows.push_back(i);
      }
    }
    return rows;
  };
  const auto& joint = joined.strings("joint_bucket");
  const auto& triple = joined.strings("triple_bucket");
  const auto& beam = joined.strings("beam_bucket");
  const auto& guiding = joined.strings("guiding_bucket");
  const auto& transverse = joined.strings("transverse_bucket");
  const auto joint_is = [&](const char* bucket) { return select([&](std::size_t i) { return joint[i] == bucket; }); };
  const auto triple_is = [&](const char* bucket) { return select([&](std::size_t i) { return triple[i] == bucket; }); };

  write("both_positive", sorted_subset(joined, joint_is("guiding_positive__beam_positive"), "joint_positive_score", false, top));
  write("guiding_positive_beam_negative",
        sorted_subset(joined, joint_is("guiding_positive__beam_negative"), "guiding_final_score", false, top));
  write("guiding_positive_beam_neutral",
        sorted_subset(joined, joint_is("guiding_positive__beam_neutral"), "guiding_final_score", false, top));
  write("beam_positive_guiding_nonpositive",
        sorted_subset(joined, select([&](std::size_t i) { return beam[i] == "positive" && guiding[i] != "positive"; }),
                      "beam_beamlike_gain_score", false, top));
  write("both_negative",
        sorted_subset(joined, joint_is("guiding_negative__beam_negative"), "guiding_beam_alignment_factor", false, top));
  write("triple_positive", sorted_subset(joined, triple_is("guiding_positive__beam_positive__transverse_positive"),
                                         "triple_positive_score", false, top));
  write("guiding_positive_beam_positive_transverse_negative",
        sorted_subset(joined, triple_is("guiding_positive__beam_positive__transverse_negative"), "joint_positive_score",
                      false, top));
  write("guiding_negative_beam_positive_transverse_positive",
        sorted_subset(joined, triple_is("guiding_negative__beam_positive__transverse_positive"),
                      "triple_positive_score", false, top));
  write("beam_positive_transverse_negative",
        sorted_subset(joined, select([&](std::size_t i) { return beam[i] == "positive" && transverse[i] == "negative"; }),
                      "beam_beamlike_gain_score", false, top));
  return paths;
}

}  // namespace guiding::products

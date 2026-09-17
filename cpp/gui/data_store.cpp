#include "data_store.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <limits>
#include <stdexcept>

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "guiding/campaign/case_metadata.hpp"
#include "guiding/exec/parallel.hpp"
#include "guiding/numeric/npcompat.hpp"
#include "guiding/products/case_reduction.hpp"
#include "guiding/products/singlecase_score.hpp"
#include "guiding/table/py_format.hpp"
#include "png_writer.hpp"

namespace guiding::gui {
namespace {

namespace fs = std::filesystem;
constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr std::size_t kMaxLogEntries = 5000;
constexpr std::size_t kMaxCachedProducts = 48;

double seconds_since(std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
}

// analyze_campaign.py:_case_metrics_is_valid
bool metrics_csv_is_valid(const fs::path& path) {
  try {
    const auto table = table::read_csv_file(path);
    if (table.rows.empty()) {
      return false;
    }
    const auto& required = products::triplet_required_columns();
    return std::all_of(required.begin(), required.end(),
                       [&](const std::string& column) { return table.has_column(column); });
  } catch (const std::exception&) {
    return false;
  }
}

void read_singlecase_sidecar(CaseRecord& record) {
  const fs::path path = record.metrics_csv.parent_path() / "guiding_singlecase_score.csv";
  record.singlecase_stamp = stamp_of(path);
  record.singlecase_score.reset();
  record.singlecase_status.clear();
  record.singlecase_reason.clear();
  if (record.singlecase_stamp.size <= 0) {
    return;
  }
  try {
    const auto table = table::read_csv_file(path);
    if (table.rows.empty()) {
      return;
    }
    if (table.has_column("metric_guiding_singlecase_score_v1")) {
      const double score = table.numeric_column("metric_guiding_singlecase_score_v1").front();
      if (std::isfinite(score)) {
        record.singlecase_score = score;
      }
    }
    if (table.has_column("metric_guiding_singlecase_status")) {
      record.singlecase_status = table.string_column("metric_guiding_singlecase_status").front();
    }
    if (table.has_column("metric_guiding_singlecase_failure_reason")) {
      record.singlecase_reason = table.string_column("metric_guiding_singlecase_failure_reason").front();
    }
  } catch (const std::exception& error) {
    record.singlecase_status = "unreadable";
    record.singlecase_reason = error.what();
  }
}

std::shared_ptr<CaseMetrics> load_case_metrics(const std::string& case_id, const fs::path& csv_path,
                                               FileStamp stamp) {
  auto metrics = std::make_shared<CaseMetrics>();
  metrics->case_id = case_id;
  metrics->csv_path = csv_path;
  metrics->stamp = stamp;
  try {
    const auto table = table::read_csv_file(csv_path);
    const std::size_t n = table.rows.size();
    metrics->rows = n;
    const auto column = [&](const char* name) {
      return table.has_column(name) ? table.numeric_column(name) : std::vector<double>(n, kNaN);
    };
    metrics->iteration = column("iteration");
    metrics->propagation_mm = column("propagation_mm");
    metrics->z_peak_um = column("z_peak_um");
    metrics->front_margin_um = column("front_margin_um");
    metrics->waist_um = column("waist_um");
    metrics->z_Ez_rel_um = column("z_Ez_absmax_rel_um");
    metrics->has_a0 = table.has_column("a0_peak");
    metrics->a0_peak = column("a0_peak");
    const auto peak_I = column("peak_I_proxy");
    const auto energy = column("energy_proxy");
    const auto Ez_abs = column("Ez_wake_absmax");

    metrics->Ez_absmax_GVm.resize(n);
    for (std::size_t i = 0; i < n; ++i) {
      metrics->Ez_absmax_GVm[i] = Ez_abs[i] / 1.0e9;
    }

    // metrics.first_valid_laser_index and the normalisations of plots.save_case_plots
    std::optional<std::size_t> ref;
    for (std::size_t i = 0; i < n && !ref; ++i) {
      if (std::isfinite(metrics->z_peak_um[i]) && std::isfinite(metrics->waist_um[i]) && std::isfinite(peak_I[i]) &&
          std::isfinite(energy[i]) && peak_I[i] > 0.0 && energy[i] > 0.0) {
        ref = i;
      }
    }
    metrics->peak_I_norm.assign(n, kNaN);
    metrics->energy_norm.assign(n, kNaN);
    metrics->a0_norm.assign(n, kNaN);
    if (!ref) {
      metrics->error = "No valid laser dumps found: peak_I/energy are zero or NaN everywhere.";
    } else {
      metrics->ref_iteration = static_cast<std::int64_t>(metrics->iteration[*ref]);
      const bool a0_reference = std::isfinite(metrics->a0_peak[*ref]) && metrics->a0_peak[*ref] > 0.0;
      for (std::size_t i = 0; i < n; ++i) {
        const bool valid = std::isfinite(peak_I[i]) && std::isfinite(energy[i]) && std::isfinite(metrics->waist_um[i]) &&
                           peak_I[i] > 0.0 && energy[i] > 0.0;
        if (valid) {
          metrics->peak_I_norm[i] = peak_I[i] / peak_I[*ref];
          metrics->energy_norm[i] = energy[i] / energy[*ref];
          if (a0_reference) {
            metrics->a0_norm[i] = metrics->a0_peak[i] / metrics->a0_peak[*ref];
          }
        }
      }

      // plots._detect_breakdown
      std::vector<double> plateau_Ez;
      for (std::size_t i = 0; i < n; ++i) {
        if (std::isfinite(Ez_abs[i]) && metrics->propagation_mm[i] >= 0.5 && metrics->propagation_mm[i] <= 3.0) {
          plateau_Ez.push_back(Ez_abs[i]);
        }
      }
      if (plateau_Ez.size() >= 3) {
        const double Ez_plateau = np::nanmedian(plateau_Ez);
        const double waist_ref = metrics->waist_um[*ref];
        for (std::size_t i = 0; i < n; ++i) {
          if (metrics->propagation_mm[i] > 1.0 && std::isfinite(Ez_abs[i]) && std::isfinite(metrics->waist_um[i]) &&
              Ez_abs[i] < 0.70 * Ez_plateau && metrics->waist_um[i] > 1.10 * waist_ref) {
            metrics->breakdown_mm = metrics->propagation_mm[i];
            metrics->breakdown_iteration = static_cast<std::int64_t>(metrics->iteration[i]);
            break;
          }
        }
      }
    }
    metrics->plateau_mm =
        campaign::infer_plateau_window_mm_from_text(table::python_path_string(csv_path.parent_path()));

    const fs::path sidecar = csv_path.parent_path() / "guiding_singlecase_score.csv";
    std::error_code error;
    if (fs::is_regular_file(sidecar, error)) {
      metrics->singlecase = table::read_csv_file(sidecar);
    }
  } catch (const std::exception& error) {
    metrics->error = error.what();
  }
  return metrics;
}

std::shared_ptr<TripletData> load_triplet(const std::string& label, const std::array<fs::path, 3>& csvs,
                                          std::vector<FileStamp> stamps, double late_fraction) {
  auto data = std::make_shared<TripletData>();
  data->label = label;
  data->stamps = std::move(stamps);
  try {
    data->tables = products::build_triplet_tables(csvs[0], csvs[1], csvs[2], label, late_fraction);
    const auto& wide = data->tables.wide;
    if (wide.has("plateau_start_mm") && wide.has("plateau_end_mm") && wide.row_count() > 0) {
      const double start = wide.doubles("plateau_start_mm").front();
      const double end = wide.doubles("plateau_end_mm").front();
      if (std::isfinite(start) && std::isfinite(end) && end > start) {
        data->plateau_mm = std::make_pair(start, end);
      }
    }
    const auto& late = data->tables.late_summary;
    if (late.has("late_start_mm") && late.has("late_end_mm") && late.row_count() > 0) {
      data->late_window_mm = std::make_pair(late.doubles("late_start_mm").front(), late.doubles("late_end_mm").front());
    }
  } catch (const std::exception& error) {
    data->error = error.what();
  }
  return data;
}

}  // namespace

FileStamp stamp_of(const fs::path& path) {
  std::error_code error;
  const auto size = fs::file_size(path, error);
  if (error) {
    return {};
  }
  const auto time = fs::last_write_time(path, error);
  if (error) {
    return {};
  }
  return {static_cast<std::int64_t>(size),
          std::chrono::duration_cast<std::chrono::nanoseconds>(time.time_since_epoch()).count()};
}

void LogBuffer::add(LogLevel level, std::string text) {
  const double now = campaign::unix_time_now();
  std::lock_guard lock(mutex_);
  entries_.push_back({now, level, std::move(text)});
  while (entries_.size() > kMaxLogEntries) {
    entries_.pop_front();
  }
  ++version_;
}

std::vector<LogEntry> LogBuffer::entries() const {
  std::lock_guard lock(mutex_);
  return {entries_.begin(), entries_.end()};
}

void LogBuffer::clear() {
  std::lock_guard lock(mutex_);
  entries_.clear();
  ++version_;
}

const CaseRecord* CampaignSnapshot::find_case(std::string_view case_id) const {
  const auto it = std::find_if(cases.begin(), cases.end(), [&](const CaseRecord& r) { return r.info.case_id == case_id; });
  return it == cases.end() ? nullptr : &*it;
}

const campaign::TripletInfo* CampaignSnapshot::find_triplet(std::string_view label) const {
  const auto it =
      std::find_if(triplets.begin(), triplets.end(), [&](const campaign::TripletInfo& t) { return t.label() == label; });
  return it == triplets.end() ? nullptr : &*it;
}

std::vector<double> TripletData::column(std::string_view name) const {
  return tables.wide.has(name) ? tables.wide.doubles(name) : std::vector<double>{};
}

DataStore::DataStore(unsigned io_workers) {
  for (unsigned i = 0; i < std::max(1U, io_workers); ++i) {
    workers_.emplace_back([this](std::stop_token stop) { worker_loop(stop, io_queue_); });
  }
  workers_.emplace_back([this](std::stop_token stop) { worker_loop(stop, heavy_queue_); });
}

DataStore::~DataStore() {
  for (auto& worker : workers_) {
    worker.request_stop();
  }
  queue_cv_.notify_all();
  workers_.clear();
}

void DataStore::set_wake_callback(std::function<void()> wake) {
  std::lock_guard lock(state_mutex_);
  wake_ = std::move(wake);
}

void DataStore::wake() {
  std::function<void()> wake;
  {
    std::lock_guard lock(state_mutex_);
    wake = wake_;
  }
  if (wake) {
    wake();
  }
}

void DataStore::post_io(Task task) {
  ++active_tasks_;
  {
    std::lock_guard lock(queue_mutex_);
    io_queue_.push_back(std::move(task));
  }
  queue_cv_.notify_all();
}

void DataStore::post_heavy(Task task) {
  ++active_tasks_;
  {
    std::lock_guard lock(queue_mutex_);
    heavy_queue_.push_back(std::move(task));
  }
  queue_cv_.notify_all();
}

void DataStore::worker_loop(std::stop_token stop, std::deque<Task>& queue) {
  while (true) {
    Task task;
    {
      std::unique_lock lock(queue_mutex_);
      if (!queue_cv_.wait(lock, stop, [&] { return !queue.empty(); })) {
        return;
      }
      task = std::move(queue.front());
      queue.pop_front();
    }
    try {
      task();
    } catch (const std::exception& error) {
      log_.add(LogLevel::Error, fmt::format("background task failed: {}", error.what()));
    }
    --active_tasks_;
    wake();
  }
}

void DataStore::request_scan(const CampaignSettings& settings) {
  {
    std::lock_guard lock(state_mutex_);
    if (scan_running_.load()) {
      pending_scan_ = settings;
      return;
    }
    scan_running_ = true;
  }
  start_scan(settings);
}

void DataStore::start_scan(CampaignSettings settings) {
  post_io([this, settings = std::move(settings)] {
    const auto started = std::chrono::steady_clock::now();
    auto snapshot = std::make_shared<CampaignSnapshot>();
    snapshot->settings = settings;
    const auto previous = campaign();
    try {
      std::error_code error;
      if (!fs::is_directory(settings.root, error)) {
        throw std::runtime_error(
            fmt::format("campaign root is not a directory: {}", table::python_path_string(settings.root)));
      }
      const double now = campaign::unix_time_now();
      snapshot->scanned_at_s = now;
      auto cases = campaign::discover_cases(settings.root);
      snapshot->triplets = campaign::build_triplets(cases);
      snapshot->cases.reserve(cases.size());
      for (auto& info : cases) {
        CaseRecord record;
        record.metrics_csv = settings.case_metrics_root / info.case_id / "guiding_metrics.csv";
        record.metrics_stamp = stamp_of(record.metrics_csv);
        record.has_min_h5 = campaign::case_has_min_h5(info, settings.min_h5);
        record.stable_h5 = campaign::case_has_stable_h5(info, settings.min_last_h5_age_min, now);
        record.age_min = campaign::newest_h5_age_min(info, now);

        // Unchanged CSVs keep the validity and score read by the previous scan.
        const CaseRecord* old = previous && previous->settings.case_metrics_root == settings.case_metrics_root
                                    ? previous->find_case(info.case_id)
                                    : nullptr;
        const FileStamp singlecase_stamp = stamp_of(record.metrics_csv.parent_path() / "guiding_singlecase_score.csv");
        if (old != nullptr && old->metrics_stamp == record.metrics_stamp) {
          record.reduced_ready = old->reduced_ready;
        } else {
          record.reduced_ready = record.metrics_stamp.size > 0 && metrics_csv_is_valid(record.metrics_csv);
        }
        if (old != nullptr && old->singlecase_stamp == singlecase_stamp) {
          record.singlecase_stamp = old->singlecase_stamp;
          record.singlecase_score = old->singlecase_score;
          record.singlecase_status = old->singlecase_status;
          record.singlecase_reason = old->singlecase_reason;
        } else {
          read_singlecase_sidecar(record);
        }
        record.info = std::move(info);
        snapshot->cases.push_back(std::move(record));
      }
    } catch (const std::exception& error) {
      snapshot->error = error.what();
    }
    snapshot->scan_seconds = seconds_since(started);

    std::optional<CampaignSettings> next;
    bool changed = false;
    {
      std::lock_guard lock(state_mutex_);
      snapshot->generation = next_generation_++;
      changed = !campaign_ || campaign_->settings != snapshot->settings || campaign_->error != snapshot->error ||
                campaign_->cases.size() != snapshot->cases.size();
      campaign_ = snapshot;
      next = std::exchange(pending_scan_, std::nullopt);
      if (!next) {
        scan_running_ = false;
      }
    }
    if (!snapshot->error.empty()) {
      log_.add(LogLevel::Error, fmt::format("[SCAN] {}", snapshot->error));
    } else if (changed) {
      log_.add(LogLevel::Info, fmt::format("[SCAN] {} cases, {} triplets in {:.2f} s", snapshot->cases.size(),
                                           snapshot->triplets.size(), snapshot->scan_seconds));
    }
    if (next) {
      start_scan(*next);
    }
  });
}

std::shared_ptr<const CampaignSnapshot> DataStore::campaign() const {
  std::lock_guard lock(state_mutex_);
  return campaign_;
}

std::shared_ptr<const CaseMetrics> DataStore::case_metrics(const CaseRecord& record) {
  if (record.metrics_stamp.size <= 0) {
    return nullptr;
  }
  const std::string& id = record.info.case_id;
  std::lock_guard lock(state_mutex_);
  const auto cached = case_cache_.find(id);
  std::shared_ptr<const CaseMetrics> current = cached == case_cache_.end() ? nullptr : cached->second;
  if (current && current->stamp == record.metrics_stamp && current->csv_path == record.metrics_csv) {
    return current;
  }
  const auto loading = case_loading_.find(id);
  if (loading != case_loading_.end() && loading->second == record.metrics_stamp) {
    return current;
  }
  case_loading_[id] = record.metrics_stamp;
  post_io([this, id, csv = record.metrics_csv, stamp = record.metrics_stamp] {
    auto metrics = load_case_metrics(id, csv, stamp);
    if (!metrics->error.empty()) {
      log_.add(LogLevel::Warning, fmt::format("[LOAD] {}: {}", id, metrics->error));
    }
    std::lock_guard publish(state_mutex_);
    case_cache_[id] = std::move(metrics);
    case_loading_.erase(id);
  });
  return current;
}

std::shared_ptr<const TripletData> DataStore::triplet(const CampaignSnapshot& snapshot,
                                                      const campaign::TripletInfo& info) {
  if (!info.complete()) {
    return nullptr;
  }
  std::array<const CaseRecord*, 3> members{snapshot.find_case(info.channel->case_id),
                                           snapshot.find_case(info.uniform->case_id),
                                           snapshot.find_case(info.vacuum->case_id)};
  if (std::any_of(members.begin(), members.end(), [](const CaseRecord* r) { return r == nullptr || !r->reduced_ready; })) {
    return nullptr;
  }
  std::vector<FileStamp> stamps;
  std::array<fs::path, 3> csvs;
  for (std::size_t i = 0; i < 3; ++i) {
    stamps.push_back(members[i]->metrics_stamp);
    csvs[i] = members[i]->metrics_csv;
  }
  stamps.push_back({static_cast<std::int64_t>(snapshot.settings.late_fraction * 1e9), 0});

  const std::string label = info.label();
  std::lock_guard lock(state_mutex_);
  const auto cached = triplet_cache_.find(label);
  std::shared_ptr<const TripletData> current = cached == triplet_cache_.end() ? nullptr : cached->second;
  if (current && current->stamps == stamps) {
    return current;
  }
  const auto loading = triplet_loading_.find(label);
  if (loading != triplet_loading_.end() && loading->second == stamps) {
    return current;
  }
  triplet_loading_[label] = stamps;
  post_io([this, label, csvs, stamps, late_fraction = snapshot.settings.late_fraction] {
    auto data = load_triplet(label, csvs, stamps, late_fraction);
    if (!data->error.empty()) {
      log_.add(LogLevel::Warning, fmt::format("[TRIPLET] {}: {}", label, data->error));
    }
    std::lock_guard publish(state_mutex_);
    triplet_cache_[label] = std::move(data);
    triplet_loading_.erase(label);
  });
  return current;
}

void DataStore::write_png_async(fs::path path, int width, int height, std::vector<std::uint8_t> rgba) {
  post_io([this, path = std::move(path), width, height, rgba = std::move(rgba)] {
    try {
      if (path.has_parent_path()) {
        fs::create_directories(path.parent_path());
      }
      write_png_rgba(path, width, height, rgba);
      log_.add(LogLevel::Info, fmt::format("[EXPORT] {}", path.string()));
    } catch (const std::exception& error) {
      log_.add(LogLevel::Error, fmt::format("[EXPORT] {}: {}", path.string(), error.what()));
    }
  });
}

std::shared_ptr<const void> DataStore::cached(const std::string& key,
                                              std::function<std::shared_ptr<const void>()> loader) {
  std::lock_guard lock(state_mutex_);
  if (const auto it = product_cache_.find(key); it != product_cache_.end()) {
    const auto position = std::find(product_order_.begin(), product_order_.end(), key);
    if (position != product_order_.end()) {
      product_order_.erase(position);
    }
    product_order_.push_back(key);
    return it->second;
  }
  if (product_loading_.contains(key)) {
    return nullptr;
  }
  product_loading_.insert(key);
  post_io([this, key, loader = std::move(loader)] {
    auto value = loader();
    std::lock_guard publish(state_mutex_);
    product_cache_[key] = std::move(value);
    product_order_.push_back(key);
    product_loading_.erase(key);
    while (product_order_.size() > kMaxCachedProducts) {
      product_cache_.erase(product_order_.front());
      product_order_.pop_front();
    }
  });
  return nullptr;
}

std::shared_ptr<const SeriesListing> DataStore::field_series(const CaseRecord& record) {
  const std::string key = fmt::format("field-series|{}|{}|{}", record.info.diag_dir.string(), record.info.h5_count,
                                      record.info.newest_h5_mtime_s.value_or(0.0));
  return std::static_pointer_cast<const SeriesListing>(cached(key, [diag = record.info.diag_dir] {
    auto listing = std::make_shared<SeriesListing>();
    listing->diag = diag;
    try {
      auto series = std::make_shared<io::FileSeries>(io::FileSeries::scan(diag));
      listing->iterations = series->iterations();
      listing->series = std::move(series);
    } catch (const std::exception& error) {
      listing->error = error.what();
    }
    return std::shared_ptr<const void>(std::move(listing));
  }));
}

std::shared_ptr<const FieldMapResult> DataStore::field_map(const SeriesListing& listing, std::int64_t iteration) {
  if (listing.series == nullptr ||
      !std::binary_search(listing.iterations.begin(), listing.iterations.end(), iteration)) {
    return nullptr;
  }
  const std::string key = fmt::format("field-map|{}|{}|{}", listing.diag.string(), iteration,
                                      listing.series->file(iteration).string());
  return std::static_pointer_cast<const FieldMapResult>(cached(key, [this, series = listing.series, iteration] {
    auto result = std::make_shared<FieldMapResult>();
    try {
      const auto started = std::chrono::steady_clock::now();
      result->map = products::load_field_map(series->file(iteration), iteration);
      log_.add(LogLevel::Info, fmt::format("[FIELD] iteration {} in {:.2f} s", iteration, seconds_since(started)));
    } catch (const std::exception& error) {
      result->error = error.what();
      log_.add(LogLevel::Warning, fmt::format("[FIELD] iteration {}: {}", iteration, error.what()));
    }
    return std::shared_ptr<const void>(std::move(result));
  }));
}

std::shared_ptr<const SeriesListing> DataStore::particle_series(const CaseRecord& record,
                                                                const std::vector<std::string>& species,
                                                                const std::string& diag_name) {
  // The rescan period bounds how long a new particle dump stays unnoticed.
  const auto snapshot = campaign();
  const std::string key =
      fmt::format("particle-series|{}|{}|{}|{}", record.info.case_dir.string(), fmt::join(species, ","), diag_name,
                  snapshot == nullptr ? 0 : snapshot->generation);
  const std::string first_species = species.empty() ? std::string("electrons") : species.front();
  return std::static_pointer_cast<const SeriesListing>(
      cached(key, [case_dir = record.info.case_dir, first_species, diag_name] {
        auto listing = std::make_shared<SeriesListing>();
        try {
          listing->diag = campaign::resolve_particle_diag_dir(case_dir, first_species, diag_name);
          auto series = std::make_shared<io::FileSeries>(io::FileSeries::scan(listing->diag));
          auto info = std::make_shared<io::ParticleSeriesInfo>(io::read_particle_series_info(*series));
          listing->iterations = series->iterations();
          listing->species = info->species.value_or(std::vector<std::string>{});
          listing->series = std::move(series);
          listing->particle_info = std::move(info);
        } catch (const std::exception& failure) {
          listing->error = failure.what();
        }
        return std::shared_ptr<const void>(std::move(listing));
      }));
}

std::shared_ptr<const ParticleViewResult> DataStore::particle_views(const SeriesListing& listing,
                                                                    const std::vector<std::string>& species,
                                                                    std::int64_t iteration,
                                                                    const products::ParticleViewOptions& options) {
  if (listing.series == nullptr || listing.particle_info == nullptr) {
    return nullptr;
  }
  const std::string key = fmt::format(
      "particle-view|{}|{}|{}|{}|{}|{}|{}|{}|{}", listing.diag.string(), iteration, fmt::join(species, ","),
      options.hot_energy_mev, physics::longitudinal_name(options.longitudinal), options.forward_only,
      options.exit_window_mm.value_or(-1.0), options.spectrum_bins, options.phase_bins);
  return std::static_pointer_cast<const ParticleViewResult>(cached(
      key, [this, series = listing.series, info = listing.particle_info, species, iteration, options] {
        auto result = std::make_shared<ParticleViewResult>();
        try {
          const auto started = std::chrono::steady_clock::now();
          std::vector<io::ParticleDump> dumps;
          for (const auto& name : species) {
            dumps.push_back(io::read_particle_dump(*series, *info, name, iteration));
          }
          if (dumps.size() > 1) {
            result->scopes.push_back(
                products::build_particle_view(physics::concatenate_particle_dumps(dumps), "all_electrons", options));
          }
          for (std::size_t i = 0; i < dumps.size(); ++i) {
            result->scopes.push_back(products::build_particle_view(dumps[i], species[i], options));
          }
          log_.add(LogLevel::Info, fmt::format("[PARTICLES] iteration {} ({}) in {:.2f} s", iteration,
                                               fmt::join(species, ","), seconds_since(started)));
        } catch (const std::exception& error) {
          result->error = error.what();
          log_.add(LogLevel::Warning, fmt::format("[PARTICLES] iteration {}: {}", iteration, error.what()));
        }
        return std::shared_ptr<const void>(std::move(result));
      }));
}

void DataStore::write_text_async(fs::path path, std::string content) {
  post_io([this, path = std::move(path), content = std::move(content)] {
    try {
      table::write_file_atomically(path, content);
      log_.add(LogLevel::Info, fmt::format("[EXPORT] {}", path.string()));
    } catch (const std::exception& error) {
      log_.add(LogLevel::Error, fmt::format("[EXPORT] {}: {}", path.string(), error.what()));
    }
  });
}

void DataStore::reduce_cases(const std::vector<campaign::CaseInfo>& cases, const CampaignSettings& settings,
                             bool overwrite) {
  if (cases.empty()) {
    return;
  }
  std::uint64_t job_id = 0;
  {
    std::lock_guard lock(state_mutex_);
    job_id = next_job_id_++;
    const std::string name = cases.size() == 1 ? fmt::format("reduce {}", cases.front().case_id)
                                               : fmt::format("reduce {} cases", cases.size());
    jobs_[job_id] = {name, false};
  }
  post_heavy([this, job_id, cases, settings, overwrite] {
    {
      std::lock_guard lock(state_mutex_);
      jobs_[job_id].running = true;
    }
    wake();
    std::size_t ok = 0;
    for (const auto& info : cases) {
      const fs::path csv = settings.case_metrics_root / info.case_id / "guiding_metrics.csv";
      std::error_code error;
      if (!overwrite && fs::exists(csv, error)) {
        log_.add(LogLevel::Info, fmt::format("[USE] existing case metrics: {}", table::python_path_string(csv)));
        ++ok;
        continue;
      }
      log_.add(LogLevel::Info, fmt::format("[MAKE] case metrics for {}", info.case_id));
      try {
        const auto started = std::chrono::steady_clock::now();
        products::CaseReductionOptions options;
        options.threads = exec::default_thread_count();
        const auto rows = products::compute_case_rows(info.diag_dir, options);
        products::write_guiding_metrics_csv(rows, csv);
        (void)products::ensure_singlecase_guiding_score_csv(csv, info.case_id, true);
        log_.add(LogLevel::Info, fmt::format("[OK] {} ({} dumps, {:.2f} s)", table::python_path_string(csv),
                                             rows.size(), seconds_since(started)));
        ++ok;
      } catch (const std::exception& failure) {
        log_.add(LogLevel::Error, fmt::format("[FAIL] case {}: {}", info.case_id, failure.what()));
      }
    }
    log_.add(ok == cases.size() ? LogLevel::Info : LogLevel::Warning,
             fmt::format("[DONE] reduced {}/{} cases", ok, cases.size()));
    {
      std::lock_guard lock(state_mutex_);
      jobs_.erase(job_id);
    }
    request_scan(settings);
  });
}

std::vector<JobStatus> DataStore::jobs() const {
  std::lock_guard lock(state_mutex_);
  std::vector<JobStatus> out;
  for (const auto& [id, job] : jobs_) {
    out.push_back(job);
  }
  return out;
}

bool DataStore::busy() const { return active_tasks_.load() > 0 || scan_running_.load(); }

}  // namespace guiding::gui

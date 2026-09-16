#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#include "guiding/campaign/discovery.hpp"
#include "guiding/products/triplet_tables.hpp"
#include "guiding/table/csv.hpp"

// Everything the dashboard shows is produced here, off the UI thread. The UI
// reads immutable snapshots (shared_ptr<const T>) that background jobs publish
// by pointer swap; it never touches the file system itself.
namespace guiding::gui {

// Size and modification time; a change means "reload". Missing files have size -1.
struct FileStamp {
  std::int64_t size = -1;
  std::int64_t mtime_ns = 0;
  bool operator==(const FileStamp&) const = default;
};
[[nodiscard]] FileStamp stamp_of(const std::filesystem::path& path);

enum class LogLevel { Info, Warning, Error };

struct LogEntry {
  double time_s = 0.0;
  LogLevel level = LogLevel::Info;
  std::string text;
};

class LogBuffer {
 public:
  void add(LogLevel level, std::string text);
  [[nodiscard]] std::vector<LogEntry> entries() const;
  [[nodiscard]] std::uint64_t version() const noexcept { return version_.load(); }
  void clear();

 private:
  mutable std::mutex mutex_;
  std::deque<LogEntry> entries_;
  std::atomic<std::uint64_t> version_{0};
};

struct CampaignSettings {
  std::filesystem::path root;
  std::filesystem::path case_metrics_root;
  std::int64_t min_h5 = 2;
  double min_last_h5_age_min = 0.0;
  double late_fraction = 1.0 / 3.0;
  bool operator==(const CampaignSettings&) const = default;
};

struct CaseRecord {
  campaign::CaseInfo info;
  std::filesystem::path metrics_csv;
  FileStamp metrics_stamp;
  bool has_min_h5 = false;
  bool stable_h5 = false;
  bool reduced_ready = false;  // guiding_metrics.csv with rows and the triplet columns
  std::optional<double> age_min;
  FileStamp singlecase_stamp;
  std::optional<double> singlecase_score;
  std::string singlecase_status;
  std::string singlecase_reason;

  [[nodiscard]] bool raw_ready() const { return has_min_h5 && stable_h5; }
  [[nodiscard]] bool usable() const { return raw_ready() || reduced_ready; }
};

struct CampaignSnapshot {
  std::uint64_t generation = 0;
  CampaignSettings settings;
  double scanned_at_s = 0.0;
  double scan_seconds = 0.0;
  std::vector<CaseRecord> cases;
  std::vector<campaign::TripletInfo> triplets;
  std::string error;

  [[nodiscard]] const CaseRecord* find_case(std::string_view case_id) const;
  [[nodiscard]] const campaign::TripletInfo* find_triplet(std::string_view label) const;
};

// guiding_metrics.csv of one case plus the quantities plots.py derives from it.
struct CaseMetrics {
  std::string case_id;
  std::filesystem::path csv_path;
  FileStamp stamp;
  std::size_t rows = 0;
  std::vector<double> iteration;
  std::vector<double> propagation_mm;
  std::vector<double> z_peak_um;
  std::vector<double> front_margin_um;
  std::vector<double> waist_um;
  std::vector<double> a0_peak;
  std::vector<double> a0_norm;
  std::vector<double> peak_I_norm;
  std::vector<double> energy_norm;
  std::vector<double> Ez_absmax_GVm;
  std::vector<double> z_Ez_rel_um;
  bool has_a0 = false;
  std::int64_t ref_iteration = 0;
  std::optional<std::pair<double, double>> plateau_mm;
  std::optional<double> breakdown_mm;
  std::optional<std::int64_t> breakdown_iteration;
  table::CsvTable singlecase;  // guiding_singlecase_score.csv when present
  std::string error;
};

struct TripletData {
  std::string label;
  std::vector<FileStamp> stamps;  // channel, uniform, vacuum CSVs
  products::TripletTables tables;
  std::optional<std::pair<double, double>> plateau_mm;
  std::optional<std::pair<double, double>> late_window_mm;
  std::string error;

  [[nodiscard]] std::vector<double> column(std::string_view name) const;
};

struct JobStatus {
  std::string name;
  bool running = false;
};

class DataStore {
 public:
  explicit DataStore(unsigned io_workers = 3);
  ~DataStore();
  DataStore(const DataStore&) = delete;
  DataStore& operator=(const DataStore&) = delete;

  // Called from worker threads after results are published (e.g. glfwPostEmptyEvent).
  void set_wake_callback(std::function<void()> wake);

  // Rescans in the background. While a scan runs, the latest request is kept
  // and started when the current one finishes.
  void request_scan(const CampaignSettings& settings);
  [[nodiscard]] std::shared_ptr<const CampaignSnapshot> campaign() const;
  [[nodiscard]] bool scanning() const noexcept { return scan_running_.load(); }

  // Cached products; a missing or outdated entry is scheduled for loading and
  // the previous value (possibly nullptr) is returned meanwhile.
  [[nodiscard]] std::shared_ptr<const CaseMetrics> case_metrics(const CaseRecord& record);
  [[nodiscard]] std::shared_ptr<const TripletData> triplet(const CampaignSnapshot& snapshot,
                                                           const campaign::TripletInfo& triplet);

  // Field reduction of the given cases into case_metrics_root (one case at a
  // time, each using all worker threads), then a rescan.
  void reduce_cases(const std::vector<campaign::CaseInfo>& cases, const CampaignSettings& settings, bool overwrite);

  [[nodiscard]] std::vector<JobStatus> jobs() const;
  [[nodiscard]] bool busy() const;
  [[nodiscard]] LogBuffer& log() noexcept { return log_; }

 private:
  using Task = std::function<void()>;
  void post_io(Task task);
  void post_heavy(Task task);
  void worker_loop(std::stop_token stop, std::deque<Task>& queue);
  void start_scan(CampaignSettings settings);
  void wake();

  LogBuffer log_;
  std::function<void()> wake_;

  mutable std::mutex queue_mutex_;
  std::condition_variable_any queue_cv_;
  std::deque<Task> io_queue_;
  std::deque<Task> heavy_queue_;
  std::atomic<int> active_tasks_{0};

  mutable std::mutex state_mutex_;
  std::shared_ptr<const CampaignSnapshot> campaign_;
  std::optional<CampaignSettings> pending_scan_;
  std::atomic<bool> scan_running_{false};
  std::uint64_t next_generation_ = 1;

  std::map<std::string, std::shared_ptr<const CaseMetrics>> case_cache_;
  std::map<std::string, FileStamp> case_loading_;
  std::map<std::string, std::shared_ptr<const TripletData>> triplet_cache_;
  std::map<std::string, std::vector<FileStamp>> triplet_loading_;
  std::map<std::uint64_t, JobStatus> jobs_;
  std::uint64_t next_job_id_ = 1;

  std::vector<std::jthread> workers_;  // last member: joined before the queues die
};

}  // namespace guiding::gui

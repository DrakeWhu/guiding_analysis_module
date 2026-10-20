#include <cstdio>
#include <mutex>

#include <fmt/format.h>

#include "commands.hpp"
#include "guiding/exec/parallel.hpp"
#include "guiding/products/singlecase_score.hpp"
#include "guiding/table/py_format.hpp"

namespace guiding::cli {

void print_line(const std::string& line) {
  static std::mutex mutex;
  std::lock_guard lock(mutex);
  fmt::print("{}\n", line);
  std::fflush(stdout);
}

void add_field_options(CLI::App& command, FieldOptions& options) {
  command.add_option("--stride", options.stride)->capture_default_str();
  command.add_option("--smooth-um", options.smooth_um)->capture_default_str();
  command.add_option("--wake-behind-um", options.wake_behind_um)->capture_default_str();
  command.add_option("--wake-gap-um", options.wake_gap_um)->capture_default_str();
  command.add_option("--lambda0-m", options.lambda0_m, "Laser wavelength [m] used to convert peak transverse E field to a0.")
      ->capture_default_str();
  command.add_option("--threads", options.threads,
                     "Worker threads (default: GUIDING_THREADS, SLURM_CPUS_PER_TASK or all available cores)");
  command.add_flag("--no-raw-reads", options.no_raw_reads,
                   "Read every dataset through HDF5 (disables the pread fast path)");
}

products::CaseReductionOptions reduction_options(const FieldOptions& options) {
  products::CaseReductionOptions reduction;
  reduction.params = {options.stride, options.smooth_um, options.wake_behind_um, options.wake_gap_um, options.lambda0_m};
  reduction.threads = options.threads == 0 ? exec::default_thread_count() : options.threads;
  reduction.raw_reads = !options.no_raw_reads;
  reduction.on_iteration = [](std::int64_t iteration) { print_line(fmt::format("[READ] iteration {}", iteration)); };
  return reduction;
}

void ensure_singlecase_sidecar(const std::filesystem::path& csv_path, const std::string& case_id, bool overwrite) {
  std::filesystem::path score_path;
  const bool written = products::ensure_singlecase_guiding_score_csv(csv_path, case_id, overwrite, &score_path);
  if (written) {
    print_line(fmt::format("[OK] wrote {}", table::python_path_string(score_path)));
  } else {
    print_line(fmt::format("[USE] existing single-case guiding score: {}", table::python_path_string(score_path)));
  }
}

}  // namespace guiding::cli

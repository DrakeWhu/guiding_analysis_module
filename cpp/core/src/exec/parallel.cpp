#include "guiding/exec/parallel.hpp"

#include <algorithm>
#include <atomic>
#include <charconv>
#include <cstdlib>
#include <exception>
#include <limits>
#include <mutex>
#include <string_view>
#include <thread>
#include <vector>

#ifdef __linux__
#include <sched.h>
#endif

namespace guiding::exec {
namespace {

unsigned parse_positive(const char* text) {
  if (text == nullptr) {
    return 0;
  }
  std::string_view view(text);
  unsigned value = 0;
  const auto result = std::from_chars(view.data(), view.data() + view.size(), value);
  return (result.ec == std::errc() && result.ptr == view.data() + view.size()) ? value : 0;
}

}  // namespace

unsigned default_thread_count() {
  if (unsigned value = parse_positive(std::getenv("GUIDING_THREADS")); value > 0) {
    return value;
  }
  if (unsigned value = parse_positive(std::getenv("SLURM_CPUS_PER_TASK")); value > 0) {
    return value;
  }
#ifdef __linux__
  cpu_set_t set;
  CPU_ZERO(&set);
  if (sched_getaffinity(0, sizeof(set), &set) == 0) {
    if (int count = CPU_COUNT(&set); count > 0) {
      return static_cast<unsigned>(count);
    }
  }
#endif
  return std::max(1U, std::thread::hardware_concurrency());
}

void parallel_for(std::size_t n, unsigned threads, const std::function<void(std::size_t)>& fn) {
  if (n == 0) {
    return;
  }
  const std::size_t workers = std::clamp<std::size_t>(threads == 0 ? 1 : threads, 1, n);

  std::atomic<std::size_t> next{0};
  std::atomic<std::size_t> first_failure{std::numeric_limits<std::size_t>::max()};
  std::mutex error_mutex;
  std::exception_ptr error;

  auto work = [&] {
    for (;;) {
      const std::size_t index = next.fetch_add(1);
      if (index >= n || index > first_failure.load()) {
        return;
      }
      try {
        fn(index);
      } catch (...) {
        std::lock_guard lock(error_mutex);
        if (index < first_failure.load()) {
          first_failure.store(index);
          error = std::current_exception();
        }
      }
    }
  };

  if (workers == 1) {
    work();
  } else {
    std::vector<std::jthread> pool;
    pool.reserve(workers - 1);
    for (std::size_t i = 1; i < workers; ++i) {
      pool.emplace_back(work);
    }
    work();
  }  // jthreads join here
  if (error) {
    std::rethrow_exception(error);
  }
}

}  // namespace guiding::exec

#pragma once

#include <cstddef>
#include <functional>

namespace guiding::exec {

// Worker count when the user gives none: GUIDING_THREADS, then
// SLURM_CPUS_PER_TASK, then the CPU affinity mask, then hardware_concurrency.
[[nodiscard]] unsigned default_thread_count();

// Runs fn(i) for every i in [0, n) on up to `threads` workers. Once a call
// throws, indices above it are skipped; after all workers finish, the exception
// of the lowest failing index is rethrown (matching a sequential loop).
void parallel_for(std::size_t n, unsigned threads, const std::function<void(std::size_t)>& fn);

}  // namespace guiding::exec

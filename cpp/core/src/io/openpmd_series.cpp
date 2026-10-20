#include "guiding/io/openpmd_series.hpp"

#include <algorithm>
#include <charconv>
#include <stdexcept>

#include <fmt/format.h>

namespace guiding::io {
namespace {

std::int64_t parse_iteration_key(const std::string& key, const std::filesystem::path& file) {
  std::int64_t value = 0;
  const char* begin = key.data();
  const char* end = key.data() + key.size();
  if (begin != end && *begin == '+') {
    ++begin;
  }
  const auto result = std::from_chars(begin, end, value);
  if (result.ec != std::errc() || result.ptr != end) {
    throw std::runtime_error(fmt::format("invalid iteration key '{}' in {}", key, file.string()));
  }
  return value;
}

bool has_hdf5_extension(const std::filesystem::path& path) {
  const std::string name = path.filename().string();
  auto ends_with = [&](std::string_view suffix) {
    return name.size() >= suffix.size() && name.compare(name.size() - suffix.size(), suffix.size(), suffix) == 0;
  };
  return ends_with(".h5") || ends_with(".hdf5");
}

}  // namespace

FileSeries FileSeries::scan(const std::filesystem::path& directory) {
  if (!std::filesystem::exists(directory)) {
    throw std::runtime_error(fmt::format("Diagnostic directory does not exist: {}", directory.string()));
  }
  FileSeries series;
  series.directory_ = directory;

  std::vector<std::filesystem::path> candidates;
  for (const auto& entry : std::filesystem::directory_iterator(directory)) {
    if (has_hdf5_extension(entry.path())) {
      candidates.push_back(std::filesystem::absolute(entry.path()));
    }
  }
  // os.listdir order is arbitrary; sorting makes duplicate iterations resolve
  // deterministically (the last file wins, as in the viewer's dict update).
  std::sort(candidates.begin(), candidates.end());

  for (const auto& path : candidates) {
    const auto file = h5::File::open_read_only(path);
    const std::string version = h5::read_string_attribute(file.id(), "openPMD");
    if (version.rfind("1.", 0) != 0) {
      throw std::runtime_error(
          fmt::format("File {} is not supported: Invalid openPMD version: {})", path.string(), version));
    }
    const auto data = h5::open_object(file.id(), "/data");
    for (const auto& key : h5::child_names(data.id())) {
      series.files_[parse_iteration_key(key, path)] = path;
    }
  }
  series.iterations_.reserve(series.files_.size());
  for (const auto& [iteration, path] : series.files_) {
    series.iterations_.push_back(iteration);
  }
  return series;
}

const std::filesystem::path& FileSeries::file(std::int64_t iteration) const {
  const auto it = files_.find(iteration);
  if (it == files_.end()) {
    throw std::out_of_range(fmt::format("iteration {} not found in {}", iteration, directory_.string()));
  }
  return it->second;
}

std::vector<std::int64_t> apply_stride(std::span<const std::int64_t> iterations, int stride) {
  if (stride < 1) {
    throw std::invalid_argument("stride must be >= 1");
  }
  std::vector<std::int64_t> selected;
  for (std::size_t i = 0; i < iterations.size(); i += static_cast<std::size_t>(stride)) {
    selected.push_back(iterations[i]);
  }
  return selected;
}

std::string join_infile_path(std::initializer_list<std::string_view> parts) {
  std::string path;
  bool first = true;
  for (std::string_view part : parts) {
    if (!first) {
      path.push_back('/');
    }
    first = false;
    path.append(part);
  }
  // Python's str.replace('//', '/') is a single non-overlapping pass.
  std::string collapsed;
  collapsed.reserve(path.size());
  for (std::size_t i = 0; i < path.size(); ++i) {
    if (path[i] == '/' && i + 1 < path.size() && path[i + 1] == '/') {
      collapsed.push_back('/');
      ++i;
    } else {
      collapsed.push_back(path[i]);
    }
  }
  return collapsed;
}

std::string meshes_path(h5::Id file, std::int64_t iteration) {
  const std::string relative = h5::read_string_attribute(file, "meshesPath");
  return join_infile_path({fmt::format("/data/{}", iteration), relative});
}

std::string particles_path(h5::Id file, std::int64_t iteration) {
  const std::string relative = h5::read_string_attribute(file, "particlesPath");
  return join_infile_path({fmt::format("/data/{}", iteration), relative});
}

}  // namespace guiding::io

#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <span>
#include <string>
#include <vector>

#include "guiding/io/h5.hpp"

namespace guiding::io {

// A file-based openPMD series in one directory, discovered like openpmd-viewer
// 1.11 (h5py backend): every *.h5 / *.hdf5 file directly inside the directory
// contributes the integer keys of its /data group as iterations.
class FileSeries {
 public:
  static FileSeries scan(const std::filesystem::path& directory);

  [[nodiscard]] const std::filesystem::path& directory() const noexcept { return directory_; }
  [[nodiscard]] const std::vector<std::int64_t>& iterations() const noexcept { return iterations_; }
  [[nodiscard]] const std::filesystem::path& file(std::int64_t iteration) const;

 private:
  std::filesystem::path directory_;
  std::vector<std::int64_t> iterations_;
  std::map<std::int64_t, std::filesystem::path> files_;
};

// iterations[::stride]
[[nodiscard]] std::vector<std::int64_t> apply_stride(std::span<const std::int64_t> iterations, int stride);

// '/'.join(parts) with '//' collapsed, as openpmd-viewer's join_infile_path.
[[nodiscard]] std::string join_infile_path(std::initializer_list<std::string_view> parts);

// Absolute in-file path of the meshes group of one iteration, e.g. "/data/100/fields/".
[[nodiscard]] std::string meshes_path(h5::Id file, std::int64_t iteration);
[[nodiscard]] std::string particles_path(h5::Id file, std::int64_t iteration);

}  // namespace guiding::io

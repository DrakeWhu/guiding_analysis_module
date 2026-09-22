#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <mutex>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

// Thin RAII layer over the HDF5 C API. hdf5.h stays out of public headers:
// hid_t is a 64-bit integer since HDF5 1.10.
namespace guiding::h5 {

using Id = std::int64_t;

class Error : public std::runtime_error {
 public:
  using std::runtime_error::runtime_error;
};

// HDF5 builds on clusters are rarely thread-safe, so every library call in this
// module runs under this lock. Raw pread() of contiguous datasets does not.
std::recursive_mutex& library_mutex();

enum class ScalarKind { Float32, Float64, Int32, Int64, UInt32, UInt64, Other };

struct DatasetInfo {
  std::vector<std::size_t> shape;
  ScalarKind kind = ScalarKind::Other;
  std::size_t element_size = 0;
  bool little_endian = true;
  // Absolute file offset of the raw values when the dataset is contiguous,
  // fully allocated, unfiltered and stored inside the file.
  std::optional<std::uint64_t> contiguous_offset;

  [[nodiscard]] std::size_t element_count() const noexcept;
};

// Owns any HDF5 identifier (file, group, dataset, attribute, dataspace, type,
// property list) and closes it with the matching H5*close under the lock.
class Handle {
 public:
  Handle() = default;
  explicit Handle(Id id) noexcept : id_(id) {}
  Handle(const Handle&) = delete;
  Handle& operator=(const Handle&) = delete;
  Handle(Handle&& other) noexcept;
  Handle& operator=(Handle&& other) noexcept;
  ~Handle();

  [[nodiscard]] Id id() const noexcept { return id_; }
  [[nodiscard]] bool valid() const noexcept { return id_ >= 0; }

 private:
  void reset() noexcept;
  Id id_ = -1;
};

class File {
 public:
  static File open_read_only(const std::filesystem::path& path);

  File(File&& other) noexcept;
  File& operator=(File&& other) noexcept;
  File(const File&) = delete;
  File& operator=(const File&) = delete;
  ~File();

  [[nodiscard]] Id id() const noexcept { return handle_.id(); }
  [[nodiscard]] const std::filesystem::path& path() const noexcept { return path_; }

  // Reads a contiguous dataset without HDF5 when possible (see DatasetInfo);
  // falls back to H5Dread under the library lock otherwise.
  void read(Id dataset, const DatasetInfo& info, std::span<double> out) const;
  void read(Id dataset, const DatasetInfo& info, std::span<float> out) const;

  // Disables the pread fast path (tests compare both paths).
  void set_raw_reads_enabled(bool enabled) noexcept { raw_reads_enabled_ = enabled; }

 private:
  File(Handle handle, std::filesystem::path path, int fd, bool raw_reads_allowed) noexcept;

  Handle handle_;
  std::filesystem::path path_;
  int fd_ = -1;
  bool raw_reads_allowed_ = false;
  bool raw_reads_enabled_ = true;
};

[[nodiscard]] bool link_exists(Id location, const std::string& path);
[[nodiscard]] Handle open_object(Id location, const std::string& path);
[[nodiscard]] bool is_dataset(Id object);
[[nodiscard]] std::vector<std::string> child_names(Id group);

[[nodiscard]] bool has_attribute(Id object, const std::string& name);
[[nodiscard]] std::string read_string_attribute(Id object, const std::string& name);
[[nodiscard]] std::vector<std::string> read_string_array_attribute(Id object, const std::string& name);
[[nodiscard]] double read_double_attribute(Id object, const std::string& name);
[[nodiscard]] std::vector<double> read_double_array_attribute(Id object, const std::string& name);
[[nodiscard]] std::uint64_t read_uint64_attribute(Id object, const std::string& name);
// Whether numpy keeps float32 when combining a float32 array with this
// attribute's scalar (NumPy 2 promotion): float16/32 and 8/16-bit integers.
[[nodiscard]] bool attribute_keeps_float32(Id object, const std::string& name);

[[nodiscard]] DatasetInfo dataset_info(Id dataset);
// Shape of a constant openPMD record component (attributes "value" and "shape").
[[nodiscard]] std::vector<std::size_t> constant_record_shape(Id group);

}  // namespace guiding::h5

#include "guiding/io/h5.hpp"

#include <hdf5.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <cstring>
#include <limits>
#include <utility>

#include <fmt/format.h>

#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif

namespace guiding::h5 {
namespace {

using Lock = std::lock_guard<std::recursive_mutex>;

[[noreturn]] void fail(const std::string& message) { throw Error(message); }

Id check(Id id, const char* what, const std::string& context) {
  if (id < 0) {
    fail(fmt::format("HDF5 {} failed: {}", what, context));
  }
  return id;
}

void check_status(herr_t status, const char* what, const std::string& context) {
  if (status < 0) {
    fail(fmt::format("HDF5 {} failed: {}", what, context));
  }
}

std::string object_name(Id object) {
  const ssize_t size = H5Iget_name(object, nullptr, 0);
  if (size <= 0) {
    return "<unnamed>";
  }
  std::string name(static_cast<std::size_t>(size), '\0');
  H5Iget_name(object, name.data(), static_cast<std::size_t>(size) + 1);
  return name;
}

std::string attribute_context(Id object, const std::string& name) {
  return fmt::format("attribute '{}' of {}", name, object_name(object));
}

// Splits a fixed-length string buffer into elements, honouring the padding.
std::string trim_fixed(const char* data, std::size_t size, H5T_str_t pad) {
  std::size_t length = size;
  if (pad == H5T_STR_SPACEPAD) {
    while (length > 0 && (data[length - 1] == ' ' || data[length - 1] == '\0')) {
      --length;
    }
  } else {
    const auto* end = std::find(data, data + size, '\0');
    length = static_cast<std::size_t>(end - data);
  }
  return std::string(data, length);
}

std::vector<std::string> read_strings(Id object, const std::string& name) {
  Lock lock(library_mutex());
  const std::string context = attribute_context(object, name);
  Handle attribute(check(H5Aopen(object, name.c_str(), H5P_DEFAULT), "H5Aopen", context));
  Handle type(check(H5Aget_type(attribute.id()), "H5Aget_type", context));
  Handle space(check(H5Aget_space(attribute.id()), "H5Aget_space", context));
  if (H5Tget_class(type.id()) != H5T_STRING) {
    fail(fmt::format("{} is not a string", context));
  }
  const hssize_t points = H5Sget_simple_extent_npoints(space.id());
  if (points < 0) {
    fail(fmt::format("cannot size {}", context));
  }
  const auto count = static_cast<std::size_t>(points);
  std::vector<std::string> values;
  values.reserve(count);

  if (H5Tis_variable_str(type.id()) > 0) {
    Handle memory(check(H5Tcopy(H5T_C_S1), "H5Tcopy", context));
    check_status(H5Tset_size(memory.id(), H5T_VARIABLE), "H5Tset_size", context);
    check_status(H5Tset_cset(memory.id(), H5Tget_cset(type.id())), "H5Tset_cset", context);
    std::vector<char*> pointers(count, nullptr);
    check_status(H5Aread(attribute.id(), memory.id(), pointers.data()), "H5Aread", context);
    for (const char* pointer : pointers) {
      values.emplace_back(pointer != nullptr ? pointer : "");
    }
#if H5_VERSION_GE(1, 12, 0)
    H5Treclaim(memory.id(), space.id(), H5P_DEFAULT, pointers.data());
#else
    H5Dvlen_reclaim(memory.id(), space.id(), H5P_DEFAULT, pointers.data());
#endif
    return values;
  }

  const std::size_t size = H5Tget_size(type.id());
  const H5T_str_t pad = H5Tget_strpad(type.id());
  std::vector<char> buffer(size * count + 1, '\0');
  check_status(H5Aread(attribute.id(), type.id(), buffer.data()), "H5Aread", context);
  for (std::size_t i = 0; i < count; ++i) {
    values.push_back(trim_fixed(buffer.data() + i * size, size, pad));
  }
  return values;
}

template <typename T>
std::vector<T> read_numeric(Id object, const std::string& name, hid_t memory_type) {
  Lock lock(library_mutex());
  const std::string context = attribute_context(object, name);
  Handle attribute(check(H5Aopen(object, name.c_str(), H5P_DEFAULT), "H5Aopen", context));
  Handle space(check(H5Aget_space(attribute.id()), "H5Aget_space", context));
  const hssize_t points = H5Sget_simple_extent_npoints(space.id());
  if (points < 0) {
    fail(fmt::format("cannot size {}", context));
  }
  std::vector<T> values(static_cast<std::size_t>(points));
  check_status(H5Aread(attribute.id(), memory_type, values.data()), "H5Aread", context);
  return values;
}

ScalarKind classify(hid_t type, std::size_t size) {
  const H5T_class_t type_class = H5Tget_class(type);
  if (type_class == H5T_FLOAT) {
    if (H5Tequal(type, H5T_IEEE_F32LE) > 0 || H5Tequal(type, H5T_IEEE_F32BE) > 0) {
      return ScalarKind::Float32;
    }
    if (H5Tequal(type, H5T_IEEE_F64LE) > 0 || H5Tequal(type, H5T_IEEE_F64BE) > 0) {
      return ScalarKind::Float64;
    }
    return ScalarKind::Other;
  }
  if (type_class == H5T_INTEGER) {
    const bool is_signed = H5Tget_sign(type) == H5T_SGN_2;
    if (size == 4) {
      return is_signed ? ScalarKind::Int32 : ScalarKind::UInt32;
    }
    if (size == 8) {
      return is_signed ? ScalarKind::Int64 : ScalarKind::UInt64;
    }
  }
  return ScalarKind::Other;
}

#ifndef _WIN32
void pread_exact(int fd, std::uint64_t offset, std::span<std::byte> out, const std::filesystem::path& path) {
  std::size_t done = 0;
  while (done < out.size()) {
    const ssize_t got = ::pread(fd, out.data() + done, out.size() - done,
                                static_cast<off_t>(offset + done));
    if (got < 0) {
      if (errno == EINTR) {
        continue;
      }
      fail(fmt::format("pread failed on {}: {}", path.string(), std::strerror(errno)));
    }
    if (got == 0) {
      fail(fmt::format("unexpected end of file while reading {}", path.string()));
    }
    done += static_cast<std::size_t>(got);
  }
}
#endif

template <typename Out>
void byteswap_in_place(std::span<Out> values) {
  for (auto& value : values) {
    auto bytes = std::bit_cast<std::array<std::byte, sizeof(Out)>>(value);
    std::reverse(bytes.begin(), bytes.end());
    value = std::bit_cast<Out>(bytes);
  }
}

}  // namespace

std::recursive_mutex& library_mutex() {
  static std::recursive_mutex mutex;
  static const bool silenced = [] {
    // Errors are reported through exceptions; keep HDF5's stack printer quiet.
    H5Eset_auto2(H5E_DEFAULT, nullptr, nullptr);
    return true;
  }();
  (void)silenced;
  return mutex;
}

std::size_t DatasetInfo::element_count() const noexcept {
  std::size_t count = 1;
  for (std::size_t extent : shape) {
    count *= extent;
  }
  return count;
}

Handle::Handle(Handle&& other) noexcept : id_(std::exchange(other.id_, -1)) {}

Handle& Handle::operator=(Handle&& other) noexcept {
  if (this != &other) {
    reset();
    id_ = std::exchange(other.id_, -1);
  }
  return *this;
}

Handle::~Handle() { reset(); }

void Handle::reset() noexcept {
  if (id_ < 0) {
    return;
  }
  Lock lock(library_mutex());
  switch (H5Iget_type(id_)) {
    case H5I_FILE: H5Fclose(id_); break;
    case H5I_GROUP: H5Gclose(id_); break;
    case H5I_DATASET: H5Dclose(id_); break;
    case H5I_ATTR: H5Aclose(id_); break;
    case H5I_DATASPACE: H5Sclose(id_); break;
    case H5I_DATATYPE: H5Tclose(id_); break;
    case H5I_GENPROP_LST: H5Pclose(id_); break;
    default: H5Idec_ref(id_); break;
  }
  id_ = -1;
}

File::File(Handle handle, std::filesystem::path path, int fd, bool raw_reads_allowed) noexcept
    : handle_(std::move(handle)), path_(std::move(path)), fd_(fd), raw_reads_allowed_(raw_reads_allowed) {}

File::File(File&& other) noexcept
    : handle_(std::move(other.handle_)),
      path_(std::move(other.path_)),
      fd_(std::exchange(other.fd_, -1)),
      raw_reads_allowed_(other.raw_reads_allowed_),
      raw_reads_enabled_(other.raw_reads_enabled_) {}

File& File::operator=(File&& other) noexcept {
  if (this != &other) {
#ifndef _WIN32
    if (fd_ >= 0) {
      ::close(fd_);
    }
#endif
    handle_ = std::move(other.handle_);
    path_ = std::move(other.path_);
    fd_ = std::exchange(other.fd_, -1);
    raw_reads_allowed_ = other.raw_reads_allowed_;
    raw_reads_enabled_ = other.raw_reads_enabled_;
  }
  return *this;
}

File::~File() {
#ifndef _WIN32
  if (fd_ >= 0) {
    ::close(fd_);
  }
#endif
}

File File::open_read_only(const std::filesystem::path& path) {
  Handle handle;
  bool raw_reads_allowed = false;
  {
    Lock lock(library_mutex());
    handle = Handle(check(H5Fopen(path.c_str(), H5F_ACC_RDONLY, H5P_DEFAULT), "H5Fopen", path.string()));
    // Dataset addresses are relative to the superblock; skip the raw path if a
    // user block shifts it.
    Handle create_plist(check(H5Fget_create_plist(handle.id()), "H5Fget_create_plist", path.string()));
    hsize_t userblock = 0;
    check_status(H5Pget_userblock(create_plist.id(), &userblock), "H5Pget_userblock", path.string());
    raw_reads_allowed = userblock == 0;
  }
  int fd = -1;
#ifndef _WIN32
  if (raw_reads_allowed) {
    fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    raw_reads_allowed = fd >= 0;
  }
#else
  raw_reads_allowed = false;
#endif
  return File(std::move(handle), path, fd, raw_reads_allowed);
}

namespace {

template <typename Out>
void read_typed(const File& file, int fd, bool raw_allowed, Id dataset, const DatasetInfo& info,
                std::span<Out> out, hid_t memory_type) {
  if (out.size() != info.element_count()) {
    fail(fmt::format("buffer of {} elements for dataset {} with {} elements", out.size(),
                     object_name(dataset), info.element_count()));
  }
  if (out.empty()) {
    return;
  }
  const bool float_source = info.kind == ScalarKind::Float32 || info.kind == ScalarKind::Float64;
#ifndef _WIN32
  if (raw_allowed && fd >= 0 && info.contiguous_offset && float_source) {
    const bool native_order = info.little_endian == (std::endian::native == std::endian::little);
    if (info.element_size == sizeof(Out)) {
      pread_exact(fd, *info.contiguous_offset, std::as_writable_bytes(out), file.path());
      if (!native_order) {
        byteswap_in_place(out);
      }
      return;
    }
    if (info.kind == ScalarKind::Float32) {
      std::vector<float> raw(out.size());
      pread_exact(fd, *info.contiguous_offset, std::as_writable_bytes(std::span<float>(raw)), file.path());
      if (!native_order) {
        byteswap_in_place(std::span<float>(raw));
      }
      std::copy(raw.begin(), raw.end(), out.begin());
      return;
    }
    // float64 source into a float buffer: let HDF5 perform the conversion.
  }
#else
  (void)fd;
  (void)raw_allowed;
#endif
  Lock lock(library_mutex());
  check_status(H5Dread(dataset, memory_type, H5S_ALL, H5S_ALL, H5P_DEFAULT, out.data()), "H5Dread",
               object_name(dataset));
}

}  // namespace

void File::read(Id dataset, const DatasetInfo& info, std::span<double> out) const {
  read_typed(*this, fd_, raw_reads_allowed_ && raw_reads_enabled_, dataset, info, out, H5T_NATIVE_DOUBLE);
}

void File::read(Id dataset, const DatasetInfo& info, std::span<float> out) const {
  read_typed(*this, fd_, raw_reads_allowed_ && raw_reads_enabled_, dataset, info, out, H5T_NATIVE_FLOAT);
}

bool link_exists(Id location, const std::string& path) {
  Lock lock(library_mutex());
  std::string prefix;
  std::size_t start = 0;
  if (!path.empty() && path.front() == '/') {
    prefix = "/";
    start = 1;
  }
  while (start <= path.size()) {
    const std::size_t end = path.find('/', start);
    const std::string part = path.substr(start, end == std::string::npos ? std::string::npos : end - start);
    if (!part.empty()) {
      prefix += (prefix.empty() || prefix.back() == '/') ? part : "/" + part;
      if (H5Lexists(location, prefix.c_str(), H5P_DEFAULT) <= 0) {
        return false;
      }
    }
    if (end == std::string::npos) {
      break;
    }
    start = end + 1;
  }
  return true;
}

Handle open_object(Id location, const std::string& path) {
  Lock lock(library_mutex());
  return Handle(check(H5Oopen(location, path.c_str(), H5P_DEFAULT), "H5Oopen", path));
}

bool is_dataset(Id object) {
  Lock lock(library_mutex());
  return H5Iget_type(object) == H5I_DATASET;
}

std::vector<std::string> child_names(Id group) {
  Lock lock(library_mutex());
  std::vector<std::string> names;
  auto callback = [](hid_t, const char* name, const H5L_info2_t*, void* data) -> herr_t {
    static_cast<std::vector<std::string>*>(data)->emplace_back(name);
    return 0;
  };
  hsize_t index = 0;
  check_status(H5Literate2(group, H5_INDEX_NAME, H5_ITER_INC, &index, callback, &names), "H5Literate2",
               object_name(group));
  return names;
}

bool has_attribute(Id object, const std::string& name) {
  Lock lock(library_mutex());
  return H5Aexists(object, name.c_str()) > 0;
}

std::string read_string_attribute(Id object, const std::string& name) {
  auto values = read_strings(object, name);
  if (values.empty()) {
    fail(fmt::format("empty {}", attribute_context(object, name)));
  }
  return values.front();
}

std::vector<std::string> read_string_array_attribute(Id object, const std::string& name) {
  return read_strings(object, name);
}

double read_double_attribute(Id object, const std::string& name) {
  auto values = read_numeric<double>(object, name, H5T_NATIVE_DOUBLE);
  if (values.size() != 1) {
    fail(fmt::format("{} holds {} values, expected 1", attribute_context(object, name), values.size()));
  }
  return values.front();
}

std::vector<double> read_double_array_attribute(Id object, const std::string& name) {
  return read_numeric<double>(object, name, H5T_NATIVE_DOUBLE);
}

std::uint64_t read_uint64_attribute(Id object, const std::string& name) {
  auto values = read_numeric<std::uint64_t>(object, name, H5T_NATIVE_UINT64);
  if (values.size() != 1) {
    fail(fmt::format("{} holds {} values, expected 1", attribute_context(object, name), values.size()));
  }
  return values.front();
}

DatasetInfo dataset_info(Id dataset) {
  Lock lock(library_mutex());
  const std::string context = object_name(dataset);
  DatasetInfo info;

  Handle space(check(H5Dget_space(dataset), "H5Dget_space", context));
  const int rank = H5Sget_simple_extent_ndims(space.id());
  if (rank < 0) {
    fail(fmt::format("cannot read rank of {}", context));
  }
  std::vector<hsize_t> dims(static_cast<std::size_t>(rank));
  if (rank > 0) {
    H5Sget_simple_extent_dims(space.id(), dims.data(), nullptr);
  }
  info.shape.assign(dims.begin(), dims.end());

  Handle type(check(H5Dget_type(dataset), "H5Dget_type", context));
  info.element_size = H5Tget_size(type.id());
  info.kind = classify(type.id(), info.element_size);
  info.little_endian = H5Tget_order(type.id()) != H5T_ORDER_BE;

  Handle create_plist(check(H5Dget_create_plist(dataset), "H5Dget_create_plist", context));
  const bool contiguous = H5Pget_layout(create_plist.id()) == H5D_CONTIGUOUS;
  const bool unfiltered = H5Pget_nfilters(create_plist.id()) == 0;
  const bool internal = H5Pget_external_count(create_plist.id()) == 0;
  if (contiguous && unfiltered && internal) {
    const haddr_t offset = H5Dget_offset(dataset);
    const hsize_t stored = H5Dget_storage_size(dataset);
    if (offset != HADDR_UNDEF && stored == info.element_count() * info.element_size) {
      info.contiguous_offset = static_cast<std::uint64_t>(offset);
    }
  }
  return info;
}

std::vector<std::size_t> constant_record_shape(Id group) {
  auto shape = read_numeric<std::uint64_t>(group, "shape", H5T_NATIVE_UINT64);
  return std::vector<std::size_t>(shape.begin(), shape.end());
}

}  // namespace guiding::h5

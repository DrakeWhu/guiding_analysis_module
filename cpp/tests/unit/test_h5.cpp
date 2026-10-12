#include <catch2/catch_test_macros.hpp>

#include <hdf5.h>

#include <array>
#include <chrono>
#include <filesystem>
#include <future>
#include <mutex>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include "guiding/io/h5.hpp"

namespace {

namespace fs = std::filesystem;
namespace h5 = guiding::h5;

constexpr std::array<double, 4> kValues{1.25, 2.5, 3.75, 5.0};
constexpr std::array<const char*, 4> kDatasets{
    "contiguous_f32", "contiguous_f64", "chunked_f32", "chunked_f64"};

struct H5Fixture {
  fs::path path;

  explicit H5Fixture(const char* name) : path(fs::temp_directory_path() / name) {
    std::lock_guard lock(h5::library_mutex());
    h5::Handle file(H5Fcreate(path.string().c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT));
    REQUIRE(file.valid());
    const hsize_t extent = kValues.size();
    h5::Handle array_space(H5Screate_simple(1, &extent, nullptr));
    h5::Handle scalar_space(H5Screate(H5S_SCALAR));
    REQUIRE(array_space.valid());
    REQUIRE(scalar_space.valid());

    h5::Handle scalar(H5Acreate2(file.id(), "scalar", H5T_IEEE_F64LE, scalar_space.id(),
                               H5P_DEFAULT, H5P_DEFAULT));
    h5::Handle array(H5Acreate2(file.id(), "array", H5T_IEEE_F64LE, array_space.id(),
                              H5P_DEFAULT, H5P_DEFAULT));
    h5::Handle integer(H5Acreate2(file.id(), "integer", H5T_STD_U64LE, scalar_space.id(),
                                H5P_DEFAULT, H5P_DEFAULT));
    h5::Handle shape(H5Acreate2(file.id(), "shape", H5T_STD_U64LE, array_space.id(),
                              H5P_DEFAULT, H5P_DEFAULT));
    REQUIRE(scalar.valid());
    REQUIRE(array.valid());
    REQUIRE(integer.valid());
    REQUIRE(shape.valid());
    const std::uint64_t integer_value = 7;
    const std::array<std::uint64_t, 4> shape_values{1, 2, 3, 4};
    REQUIRE(H5Awrite(scalar.id(), H5T_NATIVE_DOUBLE, kValues.data()) >= 0);
    REQUIRE(H5Awrite(array.id(), H5T_NATIVE_DOUBLE, kValues.data()) >= 0);
    REQUIRE(H5Awrite(integer.id(), H5T_NATIVE_UINT64, &integer_value) >= 0);
    REQUIRE(H5Awrite(shape.id(), H5T_NATIVE_UINT64, shape_values.data()) >= 0);

    h5::Handle string_type(H5Tcopy(H5T_C_S1));
    REQUIRE(string_type.valid());
    REQUIRE(H5Tset_size(string_type.id(), 8) >= 0);
    h5::Handle string(H5Acreate2(file.id(), "string", string_type.id(), scalar_space.id(),
                               H5P_DEFAULT, H5P_DEFAULT));
    const std::array<char, 8> text{'h', 'e', 'l', 'l', 'o', '\0', '\0', '\0'};
    REQUIRE(string.valid());
    REQUIRE(H5Awrite(string.id(), string_type.id(), text.data()) >= 0);
    h5::Handle empty_space(H5Screate(H5S_NULL));
    REQUIRE(empty_space.valid());
    h5::Handle empty(H5Acreate2(file.id(), "empty", string_type.id(), empty_space.id(),
                              H5P_DEFAULT, H5P_DEFAULT));
    REQUIRE(empty.valid());

    for (std::size_t i = 0; i < kDatasets.size(); ++i) {
      h5::Handle properties(H5Pcreate(H5P_DATASET_CREATE));
      REQUIRE(properties.valid());
      if (i >= 2) {
        REQUIRE(H5Pset_chunk(properties.id(), 1, &extent) >= 0);
      }
      // The f64 datasets also exercise the raw path's byte swap on LE hosts.
      const hid_t type = i % 2 == 0 ? H5T_IEEE_F32LE : H5T_IEEE_F64BE;
      h5::Handle dataset(H5Dcreate2(file.id(), kDatasets[i], type, array_space.id(),
                                  H5P_DEFAULT, properties.id(), H5P_DEFAULT));
      REQUIRE(dataset.valid());
      REQUIRE(H5Dwrite(dataset.id(), H5T_NATIVE_DOUBLE, H5S_ALL, H5S_ALL, H5P_DEFAULT,
                       kValues.data()) >= 0);
    }
  }

  ~H5Fixture() {
    std::error_code error;
    fs::remove(path, error);
  }
};

// Catch assertions run only on the test thread; worker exceptions are rethrown
// by future::get so failures do not terminate the process or race Catch2.
void expect(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

template <typename Action>
void expect_h5_error(Action action, const char* context) {
  try {
    action();
  } catch (const h5::Error& error) {
    expect(std::string(error.what()).find(context) != std::string::npos,
           "HDF5 error lost its object context");
    return;
  }
  throw std::runtime_error("expected an HDF5 validation error");
}

void exercise_reads(const fs::path& path) {
  for (unsigned iteration = 0; iteration < 32; ++iteration) {
    auto file = h5::File::open_read_only(path);
    expect(h5::read_double_attribute(file.id(), "scalar") == kValues[0], "double attribute");
    expect(h5::read_double_array_attribute(file.id(), "array") ==
               std::vector<double>(kValues.begin(), kValues.end()), "double array attribute");
    expect(h5::read_uint64_attribute(file.id(), "integer") == 7, "uint64 attribute");
    expect(h5::constant_record_shape(file.id()) == std::vector<std::size_t>{1, 2, 3, 4},
           "constant-record shape");
    expect(h5::read_string_attribute(file.id(), "string") == "hello", "string attribute");
    expect(h5::read_string_array_attribute(file.id(), "string") ==
               std::vector<std::string>{"hello"}, "string array attribute");
    expect(h5::has_attribute(file.id(), "scalar"), "attribute existence");
    expect(!h5::attribute_keeps_float32(file.id(), "scalar"), "attribute type query");
    expect(h5::child_names(file.id()).size() == kDatasets.size(), "child enumeration");

    // These errors construct diagnostic names after the attribute reader has
    // released its lock, which used to leave H5Iget_name unprotected.
    expect_h5_error([&] { (void)h5::read_double_attribute(file.id(), "array"); }, "array");
    expect_h5_error([&] { (void)h5::read_uint64_attribute(file.id(), "shape"); }, "shape");
    expect_h5_error([&] { (void)h5::read_string_attribute(file.id(), "empty"); }, "empty");

    for (const char* name : kDatasets) {
      expect(h5::link_exists(file.id(), name), "dataset existence");
      auto dataset = h5::open_object(file.id(), name);
      expect(h5::is_dataset(dataset.id()), "dataset type query");
      const auto info = h5::dataset_info(dataset.id());
      for (bool raw : {true, false}) {
        file.set_raw_reads_enabled(raw);
        std::array<double, kValues.size()> doubles{};
        std::array<float, kValues.size()> floats{};
        file.read(dataset.id(), info, std::span<double>(doubles));
        file.read(dataset.id(), info, std::span<float>(floats));
        expect(doubles == kValues, "double dataset read");
        for (std::size_t i = 0; i < floats.size(); ++i) {
          expect(floats[i] == static_cast<float>(kValues[i]), "float dataset read");
        }
      }
      std::array<double, 1> wrong_size{};
      expect_h5_error([&] { file.read(dataset.id(), info, std::span<double>(wrong_size)); }, name);
    }
  }
}

}  // namespace

TEST_CASE("HDF5 wrappers and validation errors are safe under concurrent reads", "[h5][io][thread]") {
  H5Fixture fixture("guiding_tests_h5_concurrent.h5");
  std::vector<std::future<void>> workers;
  for (unsigned i = 0; i < 8; ++i) {
    workers.push_back(std::async(std::launch::async, [&] { exercise_reads(fixture.path); }));
  }
  for (auto& worker : workers) {
    REQUIRE_NOTHROW(worker.get());
  }
}

#ifndef _WIN32
TEST_CASE("raw HDF5 dataset reads do not wait for the library mutex", "[h5][io][thread]") {
  H5Fixture fixture("guiding_tests_h5_raw_unlocked.h5");
  auto file = h5::File::open_read_only(fixture.path);
  auto dataset = h5::open_object(file.id(), "contiguous_f32");
  const auto info = h5::dataset_info(dataset.id());
  REQUIRE(info.contiguous_offset.has_value());

  std::future<std::array<double, kValues.size()>> worker;
  std::future_status status;
  {
    std::lock_guard lock(h5::library_mutex());
    worker = std::async(std::launch::async, [&] {
      std::array<double, kValues.size()> values{};
      file.read(dataset.id(), info, std::span<double>(values));
      return values;
    });
    status = worker.wait_for(std::chrono::seconds(5));
  }
  // Release the mutex before assertions or joining even if a regression makes
  // the read block, so a failure is bounded and cannot deadlock the test suite.
  CHECK(status == std::future_status::ready);
  CHECK(worker.get() == kValues);
}
#endif

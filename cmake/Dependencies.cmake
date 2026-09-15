# Third-party dependencies.
#
# Every package first tries an installed copy (HPC module, vcpkg, distro) via
# FIND_PACKAGE_ARGS and otherwise falls back to a pinned, hash-checked archive.
# Offline builds: set FETCHCONTENT_SOURCE_DIR_<NAME> to unpacked sources, or
# run tools/prefetch_deps.sh on a machine with network access.

include(FetchContent)
set(FETCHCONTENT_QUIET ON)

include(FindOrFetchHDF5)

# GCC 12 (the SUNRISE toolchain) has no std::format.
FetchContent_Declare(fmt
  URL https://github.com/fmtlib/fmt/archive/refs/tags/12.2.0.tar.gz
  URL_HASH SHA256=8b852bb5aa6e7d8564f9e81394055395dd1d1936d38dfd3a17792a02bebd7af0
  FIND_PACKAGE_ARGS)

# libc++ lacks floating-point std::from_chars on older macOS toolchains.
FetchContent_Declare(FastFloat
  URL https://github.com/fastfloat/fast_float/archive/refs/tags/v8.2.10.tar.gz
  URL_HASH SHA256=76f958dd97b1cf4d8862d1f0986a47d4bdfa8845252bae15ef0f40de3b95961f
  FIND_PACKAGE_ARGS)

set(FMT_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(fmt FastFloat)

if(GUIDING_BUILD_CLI OR GUIDING_BUILD_GUI)
  FetchContent_Declare(CLI11
    URL https://github.com/CLIUtils/CLI11/archive/refs/tags/v2.7.2.tar.gz
    URL_HASH SHA256=46eef3101da70852ec7af026e09d485ccee81813331c8c6052d39344443b83da
    FIND_PACKAGE_ARGS)
  FetchContent_MakeAvailable(CLI11)
endif()

if(GUIDING_BUILD_TESTS)
  # Debian 12 ships Catch2 v2; require v3.
  FetchContent_Declare(Catch2
    URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.16.0.tar.gz
    URL_HASH SHA256=0957cae5821b17ce07f0833aaa52b5137643a8382203221f363a8303c109af34
    FIND_PACKAGE_ARGS 3)
  FetchContent_MakeAvailable(Catch2)
  if(DEFINED catch2_SOURCE_DIR)
    list(APPEND CMAKE_MODULE_PATH "${catch2_SOURCE_DIR}/extras")
  endif()
endif()

if(GUIDING_BUILD_GUI)
  find_package(OpenGL REQUIRED)

  FetchContent_Declare(glfw3
    URL https://github.com/glfw/glfw/archive/refs/tags/3.4.tar.gz
    URL_HASH SHA256=c038d34200234d071fae9345bc455e4a8f2f544ab60150765d7704e08f3dac01
    FIND_PACKAGE_ARGS 3.3)
  set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)

  FetchContent_Declare(imgui
    URL https://github.com/ocornut/imgui/archive/refs/tags/v1.92.9b-docking.tar.gz
    URL_HASH SHA256=90ded916bd57db2e0e171b6b098940a47c6f5042725dcdc67fb19940ca8bfdcc)
  FetchContent_Declare(implot
    URL https://github.com/epezent/implot/archive/refs/tags/v1.0.tar.gz
    URL_HASH SHA256=e4a9db64eef7bcc604e2a5ea380af124eb97aa3e8a6a96330079c88add0f7e93)
  FetchContent_MakeAvailable(glfw3 imgui implot)
endif()

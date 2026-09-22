cmake_minimum_required(VERSION 3.24)

project(guiding_analysis
  VERSION 0.1.0
  DESCRIPTION "Reduction and visualisation of WarpX RZ capillary guiding campaigns"
  LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE RelWithDebInfo CACHE STRING "Build type" FORCE)
endif()

option(GUIDING_BUILD_CLI "Build the headless guiding_cli executable" ON)
option(GUIDING_BUILD_GUI "Build the guiding_gui desktop dashboard" ON)
option(GUIDING_BUILD_TESTS "Build the Catch2 test suite" ON)
option(GUIDING_FETCH_HDF5 "Build HDF5 from source when no installed HDF5 is found" OFF)
option(GUIDING_NATIVE_ARCH "Optimise for the build machine (-march=native)" OFF)
option(GUIDING_LTO "Enable link-time optimisation" OFF)
set(GUIDING_SANITIZE "" CACHE STRING "Sanitizers to enable, e.g. address;undefined or thread")

if(GUIDING_BUILD_GUI AND NOT EXISTS "${PROJECT_SOURCE_DIR}/cpp/gui/CMakeLists.txt")
  message(STATUS "cpp/gui not present yet; building without the GUI")
  set(GUIDING_BUILD_GUI OFF)
endif()

list(APPEND CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
include(CompilerOptions)
include(Dependencies)

add_subdirectory(cpp/core)

if(GUIDING_BUILD_CLI)
  add_subdirectory(cpp/cli)
endif()

if(GUIDING_BUILD_GUI)
  add_subdirectory(cpp/gui)
endif()

if(GUIDING_BUILD_TESTS)
  enable_testing()
  add_subdirectory(cpp/tests)
endif()

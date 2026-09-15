# Provides guiding::hdf5 (HDF5 C library).
#
# Order: installed HDF5 (HDF5_ROOT, HPC module, distro, vcpkg), then a source
# build of HDF5 1.14.6 when GUIDING_FETCH_HDF5=ON. A parallel (MPI) HDF5 also
# needs MPI::MPI_C because its public headers include mpi.h; the reader itself
# stays serial.

if(GUIDING_FETCH_HDF5)
  find_package(HDF5 QUIET COMPONENTS C)
else()
  find_package(HDF5 REQUIRED COMPONENTS C)
endif()

add_library(guiding_hdf5 INTERFACE)
add_library(guiding::hdf5 ALIAS guiding_hdf5)

if(HDF5_FOUND)
  if(TARGET HDF5::HDF5)
    target_link_libraries(guiding_hdf5 INTERFACE HDF5::HDF5)
  else()
    target_include_directories(guiding_hdf5 SYSTEM INTERFACE ${HDF5_INCLUDE_DIRS})
    target_link_libraries(guiding_hdf5 INTERFACE ${HDF5_C_LIBRARIES})
    target_compile_definitions(guiding_hdf5 INTERFACE ${HDF5_DEFINITIONS})
  endif()
  if(HDF5_IS_PARALLEL)
    find_package(MPI REQUIRED COMPONENTS C)
    target_link_libraries(guiding_hdf5 INTERFACE MPI::MPI_C)
  endif()
  message(STATUS "HDF5 ${HDF5_VERSION} found (parallel: ${HDF5_IS_PARALLEL})")
else()
  message(STATUS "HDF5 not found: building HDF5 1.14.6 from source")
  FetchContent_Declare(hdf5
    URL https://github.com/HDFGroup/hdf5/archive/refs/tags/hdf5_1.14.6.tar.gz
    URL_HASH SHA256=09ee1c671a87401a5201c06106650f62badeea5a3b3941e9b1e2e1e08317357f)
  set(HDF5_EXTERNALLY_CONFIGURED ON)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(BUILD_STATIC_LIBS ON CACHE BOOL "" FORCE)
  set(HDF5_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(HDF5_BUILD_UTILS OFF CACHE BOOL "" FORCE)
  set(HDF5_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(HDF5_BUILD_CPP_LIB OFF CACHE BOOL "" FORCE)
  set(HDF5_BUILD_HL_LIB OFF CACHE BOOL "" FORCE)
  set(HDF5_BUILD_FORTRAN OFF CACHE BOOL "" FORCE)
  set(HDF5_BUILD_JAVA OFF CACHE BOOL "" FORCE)
  set(HDF5_ENABLE_PARALLEL OFF CACHE BOOL "" FORCE)
  set(HDF5_ENABLE_SZIP_SUPPORT OFF CACHE BOOL "" FORCE)
  find_package(ZLIB QUIET)
  set(HDF5_ENABLE_Z_LIB_SUPPORT ${ZLIB_FOUND} CACHE BOOL "" FORCE)
  set(HDF5_TEST_SERIAL OFF CACHE BOOL "" FORCE)
  set(_guiding_saved_build_testing ${BUILD_TESTING})
  set(BUILD_TESTING OFF)
  FetchContent_MakeAvailable(hdf5)
  set(BUILD_TESTING ${_guiding_saved_build_testing})
  target_link_libraries(guiding_hdf5 INTERFACE hdf5-static)
endif()

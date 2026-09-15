add_library(guiding_core STATIC
  src/campaign/case_metadata.cpp
  src/campaign/discovery.cpp
  src/exec/parallel.cpp
  src/io/h5.cpp
  src/io/openpmd_series.cpp
  src/io/thetamode.cpp
  src/numeric/npcompat.cpp
  src/physics/field_metrics.cpp
  src/products/case_reduction.cpp
  src/products/singlecase_score.cpp
  src/products/triplet_tables.cpp
  src/table/csv.cpp
  src/table/py_format.cpp)
add_library(guiding::core ALIAS guiding_core)

target_include_directories(guiding_core PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_features(guiding_core PUBLIC cxx_std_20)
target_link_libraries(guiding_core
  PUBLIC fmt::fmt Threads::Threads
  PRIVATE guiding::compile_options guiding::hdf5 FastFloat::fast_float)
set_target_properties(guiding_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

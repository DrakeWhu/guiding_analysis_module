add_executable(guiding_tests
  parity/test_field_parity.cpp
  support/table_compare.cpp
  unit/test_csv.cpp
  unit/test_numpy_reference.cpp)
target_include_directories(guiding_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(guiding_tests PRIVATE guiding::core Catch2::Catch2WithMain guiding::compile_options)
target_compile_definitions(guiding_tests PRIVATE GUIDING_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

include(Catch)
catch_discover_tests(guiding_tests WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

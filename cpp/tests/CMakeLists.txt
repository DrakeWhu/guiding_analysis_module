add_executable(guiding_tests
  parity/test_campaign_parity.cpp
  parity/test_field_parity.cpp
  parity/test_particle_parity.cpp
  support/table_compare.cpp
  unit/test_campaign.cpp
  unit/test_csv.cpp
  unit/test_numpy_reference.cpp
  unit/test_particle_exit.cpp)
target_include_directories(guiding_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
if(TARGET guiding_gui_data)
  target_sources(guiding_tests PRIVATE unit/test_gui_data_store.cpp)
  target_link_libraries(guiding_tests PRIVATE guiding_gui_data)
endif()
target_link_libraries(guiding_tests PRIVATE guiding::core Catch2::Catch2WithMain guiding::compile_options nlohmann_json::nlohmann_json)
target_compile_definitions(guiding_tests PRIVATE GUIDING_TEST_DATA_DIR="${CMAKE_CURRENT_SOURCE_DIR}/data")

include(Catch)
catch_discover_tests(guiding_tests WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}")

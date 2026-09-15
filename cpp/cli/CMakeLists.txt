add_executable(guiding_cli
  campaign_command.cpp
  case_command.cpp
  inspect_command.cpp
  main.cpp
  support.cpp
  triplet_command.cpp)
target_link_libraries(guiding_cli PRIVATE guiding::core CLI11::CLI11 guiding::compile_options)

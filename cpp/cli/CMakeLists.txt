add_executable(guiding_cli main.cpp)
target_link_libraries(guiding_cli PRIVATE guiding::core CLI11::CLI11 guiding::compile_options)

# Dear ImGui (docking branch) with the GLFW + OpenGL 3 backends, and ImPlot.
# Neither project ships a CMake build, so their sources are compiled here.
add_library(guiding_imgui STATIC
  ${imgui_SOURCE_DIR}/imgui.cpp
  ${imgui_SOURCE_DIR}/imgui_demo.cpp
  ${imgui_SOURCE_DIR}/imgui_draw.cpp
  ${imgui_SOURCE_DIR}/imgui_tables.cpp
  ${imgui_SOURCE_DIR}/imgui_widgets.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_glfw.cpp
  ${imgui_SOURCE_DIR}/backends/imgui_impl_opengl3.cpp
  ${implot_SOURCE_DIR}/implot.cpp
  ${implot_SOURCE_DIR}/implot_items.cpp
  ${implot_SOURCE_DIR}/implot_demo.cpp)
target_include_directories(guiding_imgui SYSTEM PUBLIC
  ${imgui_SOURCE_DIR}
  ${imgui_SOURCE_DIR}/backends
  ${implot_SOURCE_DIR})
target_link_libraries(guiding_imgui PUBLIC glfw OpenGL::GL)
set_target_properties(guiding_imgui PROPERTIES CXX_STANDARD 17)

# Background loading and caching, free of any windowing code so tests can use it.
add_library(guiding_gui_data STATIC data_store.cpp)
target_include_directories(guiding_gui_data PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(guiding_gui_data PUBLIC guiding::core PRIVATE guiding::compile_options)

add_executable(guiding_gui
  app.cpp
  campaign_panel.cpp
  case_panel.cpp
  log_panel.cpp
  main.cpp
  overview_panel.cpp
  png_writer.cpp
  triplet_panel.cpp
  widgets.cpp)
target_link_libraries(guiding_gui PRIVATE guiding_gui_data guiding_imgui CLI11::CLI11 nlohmann_json::nlohmann_json
  guiding::compile_options)

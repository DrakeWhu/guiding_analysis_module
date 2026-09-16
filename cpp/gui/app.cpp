#include "app.hpp"

#include <GLFW/glfw3.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <stdexcept>

#include <fmt/format.h>
#include <nlohmann/json.hpp>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "imgui_internal.h"
#include "implot.h"
#include "panels.hpp"
#include "png_writer.hpp"

namespace guiding::gui {
namespace {

namespace fs = std::filesystem;

fs::path config_directory() {
  if (const char* xdg = std::getenv("XDG_CONFIG_HOME"); xdg != nullptr && *xdg != '\0') {
    return fs::path(xdg) / "guiding_gui";
  }
  if (const char* home = std::getenv("HOME"); home != nullptr && *home != '\0') {
    return fs::path(home) / ".config" / "guiding_gui";
  }
  return fs::current_path() / ".guiding_gui";
}

// analyze_campaign.py writes OUTDIR/case_metrics with OUTDIR=analysis_outputs/campaign
// relative to where it ran; look next to and above the campaign.
std::string guess_metrics_root(const fs::path& root) {
  const fs::path relative = fs::path("analysis_outputs") / "campaign" / "case_metrics";
  const std::vector<fs::path> candidates{root / relative, root.parent_path() / relative, fs::current_path() / relative};
  for (const auto& candidate : candidates) {
    std::error_code error;
    if (fs::is_directory(candidate, error)) {
      return candidate.string();
    }
  }
  return candidates.front().string();
}

void apply_style(float scale) {
  ImGui::StyleColorsDark();
  ImGuiStyle& style = ImGui::GetStyle();
  style.WindowRounding = 4.0f;
  style.FrameRounding = 3.0f;
  style.GrabRounding = 3.0f;
  style.TabRounding = 3.0f;
  style.WindowPadding = ImVec2(8, 6);
  style.FramePadding = ImVec2(6, 3);
  style.ItemSpacing = ImVec2(8, 5);
  style.ScaleAllSizes(scale);
  style.FontScaleDpi = scale;
  ImPlot::StyleColorsDark();
  ImPlot::GetStyle().FitPadding = ImVec2(0.04f, 0.10f);  // keep edge points and markers visible
}

double percentile(std::vector<double> values, double p) {
  if (values.empty()) {
    return 0.0;
  }
  std::sort(values.begin(), values.end());
  const auto index = static_cast<std::size_t>(p / 100.0 * static_cast<double>(values.size() - 1) + 0.5);
  return values[std::min(index, values.size() - 1)];
}

}  // namespace

App::App(AppOptions options) : options_(std::move(options)), store_(std::make_unique<DataStore>()) {
  config_dir_ = config_directory();
  ui_.min_h5 = options_.min_h5;
  ui_.min_last_h5_age_min = options_.min_last_h5_age_min;
  ui_.late_fraction = options_.late_fraction;
  ui_.poll_seconds = options_.poll_seconds;
  ui_.auto_refresh = options_.auto_refresh;
  if (options_.self_test_frames == 0) {
    load_state();
  }
  if (!options_.campaign_root.empty()) {
    ui_.root_input = options_.campaign_root;
  } else if (!ui_.recent_roots.empty()) {
    ui_.root_input = ui_.recent_roots.front();
  }
  if (!options_.case_metrics_root.empty()) {
    ui_.metrics_root_input = options_.case_metrics_root;
  }
  ui_.selected_case = options_.select_case;
  ui_.selected_triplet = options_.select_triplet;
  ui_.focus_request = options_.focus_window;
}

App::~App() {
  if (options_.self_test_frames == 0) {
    save_state();
  }
  store_.reset();  // joins workers before the window (wake callback) goes away
  shutdown_window();
}

CampaignSettings App::settings() const {
  CampaignSettings settings;
  settings.root = ui_.root_input;
  settings.case_metrics_root = ui_.metrics_root_input;
  settings.min_h5 = ui_.min_h5;
  settings.min_last_h5_age_min = ui_.min_last_h5_age_min;
  settings.late_fraction = ui_.late_fraction;
  return settings;
}

void App::open_campaign() {
  if (ui_.root_input.empty()) {
    return;
  }
  const std::string root = fs::path(ui_.root_input).lexically_normal().string();
  ui_.root_input = root;
  if (ui_.metrics_root_input.empty()) {
    const auto known = ui_.metrics_roots.find(root);
    ui_.metrics_root_input = known != ui_.metrics_roots.end() ? known->second : guess_metrics_root(root);
  }
  ui_.metrics_roots[root] = ui_.metrics_root_input;
  ui_.recent_roots.erase(std::remove(ui_.recent_roots.begin(), ui_.recent_roots.end(), root), ui_.recent_roots.end());
  ui_.recent_roots.insert(ui_.recent_roots.begin(), root);
  if (ui_.recent_roots.size() > 10) {
    ui_.recent_roots.resize(10);
  }
  store_->log().add(LogLevel::Info, fmt::format("[OPEN] {} (case metrics: {})", root, ui_.metrics_root_input));
  rescan();
}

void App::rescan() {
  if (ui_.root_input.empty()) {
    return;
  }
  last_poll_s_ = glfwGetTime();
  store_->request_scan(settings());
}

void App::load_state() {
  std::ifstream stream(config_dir_ / "state.json");
  if (!stream) {
    return;
  }
  try {
    const auto state = nlohmann::json::parse(stream);
    ui_.recent_roots = state.value("recent_roots", std::vector<std::string>{});
    ui_.metrics_roots = state.value("case_metrics_roots", std::map<std::string, std::string>{});
    ui_.poll_seconds = state.value("poll_seconds", ui_.poll_seconds);
    ui_.auto_refresh = state.value("auto_refresh", ui_.auto_refresh);
  } catch (const std::exception& error) {
    store_->log().add(LogLevel::Warning, fmt::format("ignoring unreadable GUI state: {}", error.what()));
  }
}

void App::save_state() const {
  try {
    fs::create_directories(config_dir_);
    const nlohmann::json state{{"recent_roots", ui_.recent_roots},
                               {"case_metrics_roots", ui_.metrics_roots},
                               {"poll_seconds", ui_.poll_seconds},
                               {"auto_refresh", ui_.auto_refresh}};
    std::ofstream(config_dir_ / "state.json") << state.dump(2) << '\n';
  } catch (const std::exception&) {
    // Losing the recent-roots list is not worth failing the exit for.
  }
}

void App::init_window() {
  glfwSetErrorCallback([](int code, const char* description) { fmt::print(stderr, "GLFW error {}: {}\n", code, description); });
  if (glfwInit() == GLFW_FALSE) {
    throw std::runtime_error("glfwInit failed (is a display available? try xvfb-run for headless use)");
  }
  glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
  glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
  glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
  glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GLFW_TRUE);
  const float scale = ImGui_ImplGlfw_GetContentScaleForMonitor(glfwGetPrimaryMonitor());
  window_ = glfwCreateWindow(static_cast<int>(static_cast<float>(options_.width) * scale),
                             static_cast<int>(static_cast<float>(options_.height) * scale),
                             "guiding_gui - capillary guiding campaigns", nullptr, nullptr);
  if (window_ == nullptr) {
    glfwTerminate();
    throw std::runtime_error("could not create an OpenGL 3.3 core window");
  }
  glfwMakeContextCurrent(window_);
  glfwSwapInterval(options_.vsync && options_.self_test_frames == 0 ? 1 : 0);

  IMGUI_CHECKVERSION();
  ImGui::CreateContext();
  ImPlot::CreateContext();
  ImGuiIO& io = ImGui::GetIO();
  io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard | ImGuiConfigFlags_DockingEnable;
  if (options_.self_test_frames > 0) {
    io.IniFilename = nullptr;  // reproducible layout
  } else {
    std::error_code error;
    fs::create_directories(config_dir_, error);
    ini_path_ = (config_dir_ / "imgui.ini").string();
    io.IniFilename = ini_path_.c_str();
  }
  apply_style(scale);
  ImGui_ImplGlfw_InitForOpenGL(window_, true);
  ImGui_ImplOpenGL3_Init("#version 330");
  store_->set_wake_callback([] { glfwPostEmptyEvent(); });
}

void App::shutdown_window() {
  if (window_ == nullptr) {
    return;
  }
  ImGui_ImplOpenGL3_Shutdown();
  ImGui_ImplGlfw_Shutdown();
  ImPlot::DestroyContext();
  ImGui::DestroyContext();
  glfwDestroyWindow(window_);
  glfwTerminate();
  window_ = nullptr;
}

int App::run() {
  init_window();
  if (!ui_.root_input.empty()) {
    open_campaign();
  }

  std::vector<double> frame_ms;
  std::vector<double> build_ms;  // UI construction only (NewFrame .. Render), without GPU work
  int frame = 0;
  int active_frames = 3;  // keep drawing a few frames after any event
  while (glfwWindowShouldClose(window_) == GLFW_FALSE) {
    const auto frame_start = std::chrono::steady_clock::now();
    const bool self_test = options_.self_test_frames > 0;
    if (self_test || active_frames > 0 || store_->busy()) {
      glfwPollEvents();
    } else {
      // Idle: sleep until input, a finished background job, or the next poll.
      const double until_poll = ui_.auto_refresh ? std::max(0.05, last_poll_s_ + ui_.poll_seconds - glfwGetTime()) : 1.0;
      glfwWaitEventsTimeout(std::min(until_poll, 1.0));
      active_frames = 3;
    }
    const ImGuiIO& io = ImGui::GetIO();
    if (io.MouseDelta.x != 0.0f || io.MouseDelta.y != 0.0f || ImGui::IsAnyItemActive() || io.MouseWheel != 0.0f) {
      active_frames = 3;
    }
    active_frames = std::max(0, active_frames - 1);

    if (ui_.auto_refresh && !ui_.root_input.empty() && !store_->scanning() &&
        glfwGetTime() - last_poll_s_ >= ui_.poll_seconds) {
      rescan();
    }
    if (self_test) {
      self_test_step(frame);
    }

    const auto build_start = std::chrono::steady_clock::now();
    ImGui_ImplOpenGL3_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    ImGui::NewFrame();
    draw_frame();
    ImGui::Render();
    build_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - build_start).count());

    int width = 0;
    int height = 0;
    glfwGetFramebufferSize(window_, &width, &height);
    glViewport(0, 0, width, height);
    glClearColor(0.08f, 0.08f, 0.09f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

    const bool last_frame = self_test && frame + 1 >= options_.self_test_frames;
    if (last_frame && !options_.screenshot.empty()) {
      save_screenshot(options_.screenshot);
    }
    glfwSwapBuffers(window_);
    frame_ms.push_back(std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - frame_start).count());
    ++frame;
    if (last_frame) {
      break;
    }
  }

  if (options_.self_test_frames > 0) {
    // Skip the warm-up frames (font atlas, first scan) in the statistics.
    const std::size_t skip = std::min<std::size_t>(frame_ms.size() / 5, 30);
    std::vector<double> steady(frame_ms.begin() + static_cast<std::ptrdiff_t>(skip), frame_ms.end());
    std::vector<double> steady_build(build_ms.begin() + static_cast<std::ptrdiff_t>(skip), build_ms.end());
    const double p50 = percentile(steady, 50.0);
    fmt::print("[SELF-TEST] frames={} frame p50={:.2f} ms p99={:.2f} ms max={:.2f} ms (~{:.0f} FPS at p50)\n", frame,
               p50, percentile(steady, 99.0), steady.empty() ? 0.0 : *std::max_element(steady.begin(), steady.end()),
               p50 > 0.0 ? 1000.0 / p50 : 0.0);
    fmt::print("[SELF-TEST] ui build p50={:.2f} ms p99={:.2f} ms (CPU side, excludes GL rendering and swap)\n",
               percentile(steady_build, 50.0), percentile(steady_build, 99.0));
    const auto snapshot = store_->campaign();
    if (snapshot != nullptr) {
      fmt::print("[SELF-TEST] cases={} triplets={} scan={:.3f} s error='{}'\n", snapshot->cases.size(),
                 snapshot->triplets.size(), snapshot->scan_seconds, snapshot->error);
    }
    std::size_t errors = 0;
    for (const auto& entry : store_->log().entries()) {
      if (entry.level == LogLevel::Error) {
        ++errors;
        fmt::print("[SELF-TEST] log error: {}\n", entry.text);
      }
    }
    return errors == 0 ? 0 : 1;
  }
  return 0;
}

// Walks the selection through cases and triplets so every view is drawn with data.
void App::self_test_step(int frame) {
  const auto snapshot = store_->campaign();
  if (snapshot == nullptr || snapshot->cases.empty() || !options_.select_case.empty() ||
      !options_.select_triplet.empty()) {
    return;
  }
  const int period = 20;
  if (frame % period == 0) {
    const auto step = static_cast<std::size_t>(frame / period);
    ui_.selected_case = snapshot->cases[step % snapshot->cases.size()].info.case_id;
    if (!snapshot->triplets.empty()) {
      ui_.selected_triplet = snapshot->triplets[step % snapshot->triplets.size()].label();
    }
    static const std::array<const char*, 3> kViews{kCaseWindow, kTripletWindow, kOverviewWindow};
    ui_.focus_request = kViews[step % kViews.size()];
  }
}

void App::save_screenshot(const fs::path& path) {
  int width = 0;
  int height = 0;
  glfwGetFramebufferSize(window_, &width, &height);
  std::vector<std::uint8_t> pixels(static_cast<std::size_t>(width) * static_cast<std::size_t>(height) * 4);
  glPixelStorei(GL_PACK_ALIGNMENT, 1);
  glReadBuffer(GL_BACK);
  glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());
  // OpenGL rows start at the bottom.
  std::vector<std::uint8_t> flipped(pixels.size());
  const std::size_t stride = static_cast<std::size_t>(width) * 4;
  for (int y = 0; y < height; ++y) {
    std::copy_n(pixels.data() + static_cast<std::size_t>(height - 1 - y) * stride, stride,
                flipped.data() + static_cast<std::size_t>(y) * stride);
  }
  for (std::size_t i = 3; i < flipped.size(); i += 4) {
    flipped[i] = 255;
  }
  write_png_rgba(path, width, height, flipped);
  fmt::print("[SELF-TEST] wrote {}\n", path.string());
}

void App::build_default_layout(unsigned int dockspace_id) {
  ImGui::DockBuilderRemoveNode(dockspace_id);
  ImGui::DockBuilderAddNode(dockspace_id, ImGuiDockNodeFlags_DockSpace);
  ImGui::DockBuilderSetNodeSize(dockspace_id, ImGui::GetMainViewport()->WorkSize);
  ImGuiID main = dockspace_id;
  const ImGuiID left = ImGui::DockBuilderSplitNode(main, ImGuiDir_Left, 0.30f, nullptr, &main);
  const ImGuiID bottom = ImGui::DockBuilderSplitNode(main, ImGuiDir_Down, 0.22f, nullptr, &main);
  ImGui::DockBuilderDockWindow(kCampaignWindow, left);
  ImGui::DockBuilderDockWindow(kLogWindow, bottom);
  ImGui::DockBuilderDockWindow(kCaseWindow, main);
  ImGui::DockBuilderDockWindow(kTripletWindow, main);
  ImGui::DockBuilderDockWindow(kOverviewWindow, main);
  ImGui::DockBuilderFinish(dockspace_id);
}

void App::draw_menu_bar() {
  if (!ImGui::BeginMainMenuBar()) {
    return;
  }
  if (ImGui::BeginMenu("File")) {
    if (ImGui::MenuItem("Open campaign...", "Ctrl+O")) {
      ui_.open_root_picker = true;
    }
    if (ImGui::BeginMenu("Recent", !ui_.recent_roots.empty())) {
      for (const auto& root : ui_.recent_roots) {
        if (ImGui::MenuItem(root.c_str())) {
          ui_.root_input = root;
          ui_.metrics_root_input.clear();
          open_campaign();
        }
      }
      ImGui::EndMenu();
    }
    if (ImGui::MenuItem("Rescan", "F5", false, !ui_.root_input.empty())) {
      rescan();
    }
    ImGui::Separator();
    if (ImGui::MenuItem("Quit", "Ctrl+Q")) {
      glfwSetWindowShouldClose(window_, GLFW_TRUE);
    }
    ImGui::EndMenu();
  }
  if (ImGui::BeginMenu("View")) {
    for (const char* name : {kCampaignWindow, kCaseWindow, kTripletWindow, kOverviewWindow, kLogWindow}) {
      if (ImGui::MenuItem(name)) {
        ui_.focus_request = name;
      }
    }
    ImGui::Separator();
    if (ImGui::MenuItem("Reset layout")) {
      layout_initialized_ = false;
    }
    ImGui::MenuItem("Auto refresh", nullptr, &ui_.auto_refresh);
    ImGui::Separator();
    ImGui::MenuItem("ImGui demo", nullptr, &ui_.show_imgui_demo);
    ImGui::MenuItem("ImPlot demo", nullptr, &ui_.show_implot_demo);
    ImGui::EndMenu();
  }
  if (ImGui::BeginMenu("Help")) {
    ImGui::MenuItem("About", nullptr, &ui_.show_about);
    ImGui::EndMenu();
  }

  const auto snapshot = store_->campaign();
  const std::string status =
      store_->scanning()      ? "scanning..."
      : snapshot == nullptr   ? "no campaign"
      : !snapshot->error.empty() ? "scan error"
                              : fmt::format("{} cases | {} triplets | scanned {:.0f} s ago", snapshot->cases.size(),
                                            snapshot->triplets.size(), campaign::unix_time_now() - snapshot->scanned_at_s);
  const std::string right = fmt::format("{}   {:.0f} FPS", status, ImGui::GetIO().Framerate);
  ImGui::SameLine(ImGui::GetWindowWidth() - ImGui::CalcTextSize(right.c_str()).x - 16.0f);
  ImGui::TextDisabled("%s", right.c_str());
  ImGui::EndMainMenuBar();
}

void App::draw_frame() {
  const ImGuiIO& io = ImGui::GetIO();
  if (ImGui::IsKeyPressed(ImGuiKey_F5, false)) {
    rescan();
  }
  if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_O, false)) {
    ui_.open_root_picker = true;
  }
  if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Q, false)) {
    glfwSetWindowShouldClose(window_, GLFW_TRUE);
  }

  draw_menu_bar();
  const ImGuiID dockspace = ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());
  if (!layout_initialized_) {
    // Keep a user's saved arrangement; build the default one otherwise.
    const ImGuiDockNode* node = ImGui::DockBuilderGetNode(dockspace);
    if (node == nullptr || node->IsLeafNode() || options_.self_test_frames > 0) {
      build_default_layout(dockspace);
    }
    layout_initialized_ = true;
  }

  draw_campaign_panel(*this, ui_, *store_);
  draw_case_panel(*this, ui_, *store_);
  draw_triplet_panel(ui_, *store_);
  draw_overview_panel(ui_, *store_);
  draw_log_panel(ui_, *store_);

  if (!ui_.focus_request.empty()) {
    ImGui::SetWindowFocus(ui_.focus_request.c_str());
    ui_.focus_request.clear();
  }
  if (ui_.open_root_picker) {
    ImGui::OpenPopup("Open campaign");
    ui_.open_root_picker = false;
  }
  if (draw_directory_picker("Open campaign", ui_.root_input)) {
    ui_.metrics_root_input.clear();
    open_campaign();
  }
  if (ui_.show_imgui_demo) {
    ImGui::ShowDemoWindow(&ui_.show_imgui_demo);
  }
  if (ui_.show_implot_demo) {
    ImPlot::ShowDemoWindow(&ui_.show_implot_demo);
  }
  if (ui_.show_about) {
    ImGui::Begin("About guiding_gui", &ui_.show_about, ImGuiWindowFlags_AlwaysAutoResize);
    ImGui::TextUnformatted("guiding_gui: dashboard for WarpX RZ capillary guiding campaigns");
    ImGui::Text("Dear ImGui %s, ImPlot %s", ImGui::GetVersion(), IMPLOT_VERSION);
    ImGui::TextDisabled("Products are computed by libguiding_core (parity with cap_guiding).");
    ImGui::End();
  }
}

}  // namespace guiding::gui

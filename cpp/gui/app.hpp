#pragma once

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <vector>

#include "data_store.hpp"

struct GLFWwindow;

namespace guiding::gui {

inline constexpr const char* kCampaignWindow = "Campaign";
inline constexpr const char* kCaseWindow = "Case";
inline constexpr const char* kTripletWindow = "Triplet";
inline constexpr const char* kOverviewWindow = "Overview";
inline constexpr const char* kLogWindow = "Log";

struct AppOptions {
  std::string campaign_root;
  std::string case_metrics_root;
  std::int64_t min_h5 = 2;
  double min_last_h5_age_min = 0.0;
  double late_fraction = 1.0 / 3.0;
  double poll_seconds = 10.0;
  bool auto_refresh = true;
  bool vsync = true;
  int width = 1600;
  int height = 1000;
  // Automation for CI / documentation screenshots.
  int self_test_frames = 0;
  std::string screenshot;
  std::string select_case;
  std::string select_triplet;
  std::string focus_window;
};

// Per-session UI state shared by the panels.
struct UiState {
  std::string root_input;
  std::string metrics_root_input;
  std::int64_t min_h5 = 2;
  double min_last_h5_age_min = 0.0;
  double late_fraction = 1.0 / 3.0;
  bool auto_refresh = true;
  double poll_seconds = 10.0;

  std::string selected_case;
  std::set<std::string> marked_cases;  // multi-selection for batch reduction
  std::string selected_triplet;
  std::string case_filter;
  bool show_channel = true;
  bool show_uniform = true;
  bool show_vacuum = true;
  bool usable_only = false;
  int overview_parameter = 0;
  bool log_autoscroll = true;
  bool log_show_info = true;

  std::string focus_request;  // window to bring to front next frame
  bool show_imgui_demo = false;
  bool show_implot_demo = false;
  bool show_about = false;
  bool open_root_picker = false;

  std::vector<std::string> recent_roots;
  std::map<std::string, std::string> metrics_roots;  // campaign root -> case metrics root
};

class App {
 public:
  explicit App(AppOptions options);
  ~App();
  App(const App&) = delete;
  App& operator=(const App&) = delete;

  int run();

  // Applies root_input/metrics_root_input and rescans.
  void open_campaign();
  void rescan();
  [[nodiscard]] CampaignSettings settings() const;

 private:
  void init_window();
  void shutdown_window();
  void draw_frame();
  void draw_menu_bar();
  void build_default_layout(unsigned int dockspace_id);
  void load_state();
  void save_state() const;
  void self_test_step(int frame);
  void save_screenshot(const std::filesystem::path& path);

  AppOptions options_;
  UiState ui_;
  std::unique_ptr<DataStore> store_;
  GLFWwindow* window_ = nullptr;
  std::filesystem::path config_dir_;
  std::string ini_path_;
  double last_poll_s_ = 0.0;
  bool layout_initialized_ = false;
};

}  // namespace guiding::gui

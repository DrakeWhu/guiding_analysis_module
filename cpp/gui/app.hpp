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
inline constexpr const char* kFieldsWindow = "Fields";
inline constexpr const char* kParticlesWindow = "Particles";
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
  std::string select_tab;
  std::string capture;  // with --self-test: export this window as PNG once data has loaded  // tab to open in the focused view, e.g. "Phase space"
  std::string export_dir;  // PNG/CSV exports (default: ./guiding_gui_exports)
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

  // Fields view
  std::int64_t field_iteration = -1;
  bool field_follow_latest = true;
  bool field_log_scale = false;

  // Particles view
  std::string particle_species = "electrons";
  std::string particle_diag_name = "auto";
  std::int64_t particle_iteration = -1;
  bool particle_follow_latest = true;
  std::string particle_scope;
  double particle_hot_energy_mev = 10.0;
  bool particle_forward_only = true;
  int particle_longitudinal = 2;  // x, y, z
  bool particle_use_exit_window = false;
  double particle_exit_window_mm = 0.3;
  bool particle_log_spectrum = true;
  bool particle_log_phase = true;

  std::string focus_request;  // window to bring to front next frame
  std::string tab_request;    // tab to select (kept for a few frames so every panel sees it)
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

  // Saves the given window as PNG after the next rendered frame.
  void request_capture(const std::string& window);
  // <export dir>/<stem>_<local time>.<extension>
  [[nodiscard]] std::filesystem::path export_path(const std::string& stem, const std::string& extension) const;

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
  void capture_pending_window();

  AppOptions options_;
  UiState ui_;
  std::unique_ptr<DataStore> store_;
  GLFWwindow* window_ = nullptr;
  std::filesystem::path config_dir_;
  std::string ini_path_;
  double last_poll_s_ = 0.0;
  bool layout_initialized_ = false;
  std::string capture_window_;
  std::filesystem::path capture_path_;
};

}  // namespace guiding::gui

#pragma once

#include <optional>
#include <span>
#include <string>
#include <utility>

#include "imgui.h"
#include "implot.h"

#include "app.hpp"
#include "data_store.hpp"
#include "guiding/table/frame.hpp"

namespace guiding::gui {

void draw_campaign_panel(App& app, UiState& ui, DataStore& store);
void draw_case_panel(App& app, UiState& ui, DataStore& store);
void draw_triplet_panel(UiState& ui, DataStore& store);
void draw_overview_panel(UiState& ui, DataStore& store);
void draw_log_panel(UiState& ui, DataStore& store);

// Modal directory browser opened with ImGui::OpenPopup(id); true when a
// directory was chosen (written to `path`).
bool draw_directory_picker(const char* id, std::string& path);

namespace widgets {

// Matplotlib's default cycle, so colours match the reference PNGs.
inline constexpr ImVec4 kChannelColor{0.122f, 0.467f, 0.706f, 1.0f};
inline constexpr ImVec4 kUniformColor{1.000f, 0.498f, 0.055f, 1.0f};
inline constexpr ImVec4 kVacuumColor{0.173f, 0.627f, 0.173f, 1.0f};
inline constexpr ImVec4 kGood{0.36f, 0.78f, 0.42f, 1.0f};
inline constexpr ImVec4 kWarn{0.95f, 0.72f, 0.25f, 1.0f};
inline constexpr ImVec4 kBad{0.90f, 0.35f, 0.33f, 1.0f};
inline constexpr ImVec4 kMuted{0.55f, 0.55f, 0.58f, 1.0f};

[[nodiscard]] ImVec4 case_type_color(campaign::CaseType type);

// Small coloured label, e.g. readiness states.
void chip(const char* text, const ImVec4& color);

// Call inside BeginPlot after the setup: shaded plateau window and its edges.
void plateau_band(const std::optional<std::pair<double, double>>& window_mm, bool tags);
// Shaded x range drawn beneath the items (e.g. the late window).
void x_band(double x0, double x1, const ImVec4& color);
void vertical_marker(const char* id, double x, const ImVec4& color, const char* tag);
void horizontal_line(const char* id, double y, const ImVec4& color);
void line_series(const char* label, std::span<const double> x, std::span<const double> y, const ImVec4& color,
                 bool markers = true);

// Finite [min, max] of the values padded by 3 % (unit range when empty), used as
// the initial x range of linked subplots, which do not fit linked axes themselves.
[[nodiscard]] std::pair<double, double> padded_range(std::span<const double> values);

// A read-only, scrollable table of a Frame (numbers shown with %.6g).
void frame_table(const char* id, const table::Frame& frame, float height = 0.0f);

}  // namespace widgets
}  // namespace guiding::gui

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <limits>
#include <vector>

#include <fmt/format.h>

#include "panels.hpp"

namespace guiding::gui {
namespace fs = std::filesystem;

namespace widgets {

ImVec4 case_type_color(campaign::CaseType type) {
  switch (type) {
    case campaign::CaseType::Channel:
      return kChannelColor;
    case campaign::CaseType::Uniform:
      return kUniformColor;
    case campaign::CaseType::Vacuum:
      return kVacuumColor;
  }
  return kMuted;
}

void chip(const char* text, const ImVec4& color) {
  const ImVec2 size = ImGui::CalcTextSize(text);
  const ImVec2 pos = ImGui::GetCursorScreenPos();
  const float pad = 3.0f;
  ImVec4 fill = color;
  fill.w = 0.22f;
  ImGui::GetWindowDrawList()->AddRectFilled(pos, ImVec2(pos.x + size.x + 2 * pad, pos.y + size.y + 1.0f),
                                            ImGui::GetColorU32(fill), 3.0f);
  ImGui::SetCursorScreenPos(ImVec2(pos.x + pad, pos.y));
  ImGui::TextColored(color, "%s", text);
}

void x_band(double x0, double x1, const ImVec4& color) {
  const ImPlotRect limits = ImPlot::GetPlotLimits();
  ImPlot::PushPlotClipRect();
  const ImVec2 a = ImPlot::PlotToPixels(x0, limits.Y.Max);
  const ImVec2 b = ImPlot::PlotToPixels(x1, limits.Y.Min);
  ImPlot::GetPlotDrawList()->AddRectFilled(ImVec2(std::min(a.x, b.x), a.y), ImVec2(std::max(a.x, b.x), b.y),
                                          ImGui::GetColorU32(color));
  ImPlot::PopPlotClipRect();
}

void plateau_band(const std::optional<std::pair<double, double>>& window_mm, bool tags) {
  if (!window_mm) {
    return;
  }
  x_band(window_mm->first, window_mm->second, ImVec4(0.55f, 0.55f, 0.65f, 0.10f));
  const ImVec4 edge(0.75f, 0.75f, 0.85f, 0.65f);
  vertical_marker("##plateau_start", window_mm->first, edge, tags ? "plateau start" : nullptr);
  vertical_marker("##plateau_end", window_mm->second, edge, tags ? "plateau end" : nullptr);
}

void vertical_marker(const char* id, double x, const ImVec4& color, const char* tag) {
  ImPlotSpec spec;
  spec.LineColor = color;
  spec.LineWeight = 1.0f;
  spec.Flags = ImPlotItemFlags_NoLegend | ImPlotItemFlags_NoFit;
  ImPlot::PlotInfLines(id, &x, 1, spec);
  if (tag != nullptr) {
    ImPlot::TagX(x, color, "%s", tag);
  }
}

void horizontal_line(const char* id, double y, const ImVec4& color) {
  ImPlotSpec spec;
  spec.LineColor = color;
  spec.LineWeight = 1.0f;
  spec.Flags = static_cast<int>(ImPlotItemFlags_NoLegend) | static_cast<int>(ImPlotInfLinesFlags_Horizontal);
  ImPlot::PlotInfLines(id, &y, 1, spec);
}

void line_series(const char* label, std::span<const double> x, std::span<const double> y, const ImVec4& color,
                 bool markers) {
  const int count = static_cast<int>(std::min(x.size(), y.size()));
  if (count == 0) {
    return;
  }
  ImPlotSpec spec;
  spec.LineColor = color;
  spec.LineWeight = 1.5f;
  if (markers) {
    spec.Marker = ImPlotMarker_Circle;
    spec.MarkerSize = 2.5f;
    spec.MarkerFillColor = color;
  }
  ImPlot::PlotLine(label, x.data(), y.data(), count, spec);
}

std::pair<double, double> padded_range(std::span<const double> values) {
  double low = std::numeric_limits<double>::infinity();
  double high = -std::numeric_limits<double>::infinity();
  for (double value : values) {
    if (std::isfinite(value)) {
      low = std::min(low, value);
      high = std::max(high, value);
    }
  }
  if (!(low <= high)) {
    return {0.0, 1.0};
  }
  const double pad = high > low ? 0.03 * (high - low) : std::max(0.5, 0.03 * std::abs(low));
  return {low - pad, high + pad};
}

void frame_table(const char* id, const table::Frame& frame, float height) {
  const auto& names = frame.names();
  if (names.empty()) {
    ImGui::TextDisabled("(empty table)");
    return;
  }
  const int columns = static_cast<int>(std::min<std::size_t>(names.size(), 64));
  const ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollX |
                                ImGuiTableFlags_ScrollY | ImGuiTableFlags_SizingFixedFit | ImGuiTableFlags_Resizable;
  if (!ImGui::BeginTable(id, columns, flags, ImVec2(0.0f, height))) {
    return;
  }
  ImGui::TableSetupScrollFreeze(1, 1);
  for (int c = 0; c < columns; ++c) {
    ImGui::TableSetupColumn(names[static_cast<std::size_t>(c)].c_str());
  }
  ImGui::TableHeadersRow();
  ImGuiListClipper clipper;
  clipper.Begin(static_cast<int>(frame.row_count()));
  while (clipper.Step()) {
    for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
      ImGui::TableNextRow();
      for (int c = 0; c < columns; ++c) {
        ImGui::TableSetColumnIndex(c);
        const auto& column = frame.column(names[static_cast<std::size_t>(c)]);
        const auto r = static_cast<std::size_t>(row);
        if (const auto* reals = std::get_if<table::Frame::Doubles>(&column)) {
          const double value = (*reals)[r];
          if (std::isnan(value)) {
            ImGui::TextDisabled("nan");
          } else {
            ImGui::Text("%.6g", value);
          }
        } else if (const auto* integers = std::get_if<table::Frame::Integers>(&column)) {
          ImGui::Text("%lld", static_cast<long long>((*integers)[r]));
        } else {
          ImGui::TextUnformatted(std::get<table::Frame::Strings>(column)[r].c_str());
        }
      }
    }
  }
  ImGui::EndTable();
}

}  // namespace widgets

bool draw_directory_picker(const char* id, std::string& path) {
  static std::string current;
  static std::string listed;
  static std::vector<std::string> entries;
  static char path_buffer[4096] = {};
  bool chosen = false;

  ImGui::SetNextWindowSize(ImVec2(640, 460), ImGuiCond_Appearing);
  if (!ImGui::BeginPopupModal(id, nullptr)) {
    return false;
  }
  std::error_code error;
  if (ImGui::IsWindowAppearing()) {
    current = !path.empty() && fs::is_directory(path, error) ? fs::path(path).lexically_normal().string()
                                                             : fs::current_path(error).string();
    listed.clear();
  }
  if (listed != current) {
    entries.clear();
    for (auto it = fs::directory_iterator(current, fs::directory_options::skip_permission_denied, error);
         !error && it != fs::directory_iterator(); it.increment(error)) {
      std::error_code type_error;
      if (it->is_directory(type_error)) {
        entries.push_back(it->path().filename().string());
      }
    }
    std::sort(entries.begin(), entries.end());
    listed = current;
    const std::size_t length = std::min(current.size(), sizeof(path_buffer) - 1);
    std::copy_n(current.data(), length, path_buffer);
    path_buffer[length] = '\0';
  }

  if (ImGui::Button("Up")) {
    current = fs::path(current).parent_path().string();
  }
  ImGui::SameLine();
  ImGui::SetNextItemWidth(-1.0f);
  if (ImGui::InputText("##path", path_buffer, sizeof(path_buffer), ImGuiInputTextFlags_EnterReturnsTrue) &&
      fs::is_directory(path_buffer, error)) {
    current = path_buffer;
  }

  ImGui::BeginChild("##dirs", ImVec2(0, -ImGui::GetFrameHeightWithSpacing() * 1.4f), ImGuiChildFlags_Borders);
  for (const auto& name : entries) {
    if (ImGui::Selectable(name.c_str(), false, ImGuiSelectableFlags_AllowDoubleClick) &&
        ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) {
      current = (fs::path(current) / name).string();
    }
  }
  ImGui::EndChild();

  if (ImGui::Button("Open this directory")) {
    path = current;
    chosen = true;
    ImGui::CloseCurrentPopup();
  }
  ImGui::SameLine();
  if (ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape)) {
    ImGui::CloseCurrentPopup();
  }
  ImGui::SameLine();
  ImGui::TextDisabled("double-click a directory to enter it");
  ImGui::EndPopup();
  return chosen;
}

}  // namespace guiding::gui

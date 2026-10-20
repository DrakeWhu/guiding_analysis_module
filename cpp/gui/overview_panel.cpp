#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "panels.hpp"

namespace guiding::gui {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();

// Numeric value of a normalised name token ("2p5" -> 2.5, "4e18cm3" -> 4e18,
// focus "m100um" -> -100 um, "p0p5mm" -> 500 um).
double token_value(const std::optional<std::string>& token, bool focus) {
  if (!token || token->empty()) {
    return kNaN;
  }
  std::string text = *token;
  double sign = 1.0;
  double unit = 1.0;
  if (focus) {
    if (text.ends_with("mm")) {
      unit = 1000.0;
    }
    text.resize(text.size() - 2);
    if (!text.empty() && (text.front() == 'm' || text.front() == 'p')) {
      sign = text.front() == 'm' ? -1.0 : 1.0;
      text.erase(text.begin());
    }
  }
  for (char& c : text) {
    if (c == 'p') {
      c = '.';
    }
  }
  char* end = nullptr;
  const double value = std::strtod(text.c_str(), &end);
  return end == text.c_str() ? kNaN : sign * unit * value;
}

struct Parameter {
  const char* label;
  bool log_scale;
};

constexpr std::array<Parameter, 5> kParameters{{{"plateau length [mm]", false},
                                                 {"density [cm^-3]", true},
                                                 {"channel diameter [um]", false},
                                                 {"focus position [um]", false},
                                                 {"f-number", false}}};

double parameter_value(const CaseRecord& record, int parameter) {
  const auto& tokens = record.info.tokens;
  switch (parameter) {
    case 0:
      return token_value(tokens.plateau, false);
    case 1:
      return token_value(record.info.density_key(), false);
    case 2:
      return token_value(tokens.diameter, false);
    case 3:
      return token_value(tokens.focus, true);
    case 4:
      return token_value(tokens.laser_case, false);
    default:
      return kNaN;
  }
}

}  // namespace

void draw_overview_panel(UiState& ui, DataStore& store) {
  if (!ImGui::Begin(kOverviewWindow)) {
    ImGui::End();
    return;
  }
  const auto snapshot = store.campaign();
  if (snapshot == nullptr || snapshot->cases.empty()) {
    ImGui::TextDisabled("No cases loaded.");
    ImGui::End();
    return;
  }

  ImGui::SetNextItemWidth(240.0f);
  if (ImGui::BeginCombo("x axis", kParameters[static_cast<std::size_t>(ui.overview_parameter)].label)) {
    for (int i = 0; i < static_cast<int>(kParameters.size()); ++i) {
      if (ImGui::Selectable(kParameters[static_cast<std::size_t>(i)].label, i == ui.overview_parameter)) {
        ui.overview_parameter = i;
      }
    }
    ImGui::EndCombo();
  }
  ImGui::SameLine();
  ImGui::TextDisabled("single-case guiding score v1 per case; click a point to open the case");

  struct Point {
    double x;
    double y;
    const CaseRecord* record;
  };
  std::array<std::vector<Point>, 3> groups;
  std::size_t without_score = 0;
  for (const auto& record : snapshot->cases) {
    const double x = parameter_value(record, ui.overview_parameter);
    if (!record.singlecase_score || !std::isfinite(x)) {
      ++without_score;
      continue;
    }
    groups[static_cast<std::size_t>(record.info.case_type)].push_back({x, *record.singlecase_score, &record});
  }
  if (without_score > 0) {
    ImGui::TextDisabled("%zu cases without a score or parameter are not shown", without_score);
  }

  const Parameter& parameter = kParameters[static_cast<std::size_t>(ui.overview_parameter)];
  if (ImPlot::BeginPlot(fmt::format("##overview_{}_{}", ui.overview_parameter, snapshot->generation).c_str(),
                        ImVec2(-1, -1))) {
    ImPlot::SetupAxes(parameter.label, "single-case score v1");
    if (parameter.log_scale) {
      ImPlot::SetupAxisScale(ImAxis_X1, ImPlotScale_Log10);
    }
    constexpr std::array<campaign::CaseType, 3> kTypes{campaign::CaseType::Channel, campaign::CaseType::Uniform,
                                                       campaign::CaseType::Vacuum};
    const Point* hovered = nullptr;
    float hovered_distance = 12.0f;
    const ImVec2 mouse = ImGui::GetMousePos();
    for (std::size_t g = 0; g < groups.size(); ++g) {
      const auto& points = groups[g];
      std::vector<double> xs;
      std::vector<double> ys;
      for (const auto& point : points) {
        xs.push_back(point.x);
        ys.push_back(point.y);
      }
      ImPlotSpec spec;
      spec.Marker = ImPlotMarker_Circle;
      spec.MarkerSize = 4.5f;
      spec.MarkerFillColor = widgets::case_type_color(kTypes[g]);
      spec.MarkerLineColor = widgets::case_type_color(kTypes[g]);
      ImPlot::PlotScatter(campaign::case_type_name(kTypes[g]), xs.data(), ys.data(), static_cast<int>(xs.size()), spec);
      if (ImPlot::IsPlotHovered()) {
        for (const auto& point : points) {
          const ImVec2 pixel = ImPlot::PlotToPixels(point.x, point.y);
          const float distance = std::hypot(pixel.x - mouse.x, pixel.y - mouse.y);
          if (distance < hovered_distance) {
            hovered_distance = distance;
            hovered = &point;
          }
        }
      }
    }
    if (const CaseRecord* selected = snapshot->find_case(ui.selected_case)) {
      const double x = parameter_value(*selected, ui.overview_parameter);
      if (selected->singlecase_score && std::isfinite(x)) {
        const ImVec2 pixel = ImPlot::PlotToPixels(x, *selected->singlecase_score);
        ImPlot::GetPlotDrawList()->AddCircle(pixel, 9.0f, IM_COL32(255, 255, 255, 200), 0, 2.0f);
      }
    }
    if (hovered != nullptr) {
      ImGui::SetTooltip("%s\nscore %.2f", hovered->record->info.case_id.c_str(), hovered->y);
      if (ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        ui.selected_case = hovered->record->info.case_id;
      }
    }
    ImPlot::EndPlot();
  }
  ImGui::End();
}

}  // namespace guiding::gui

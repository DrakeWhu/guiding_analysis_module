#include <array>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "panels.hpp"

namespace guiding::gui {
namespace {

constexpr std::array<const char*, 3> kCases{"channel", "uniform", "vacuum"};
constexpr std::array<ImVec4, 3> kCaseColors{widgets::kChannelColor, widgets::kUniformColor, widgets::kVacuumColor};
constexpr std::array<const char*, 3> kRatios{"channel_over_vacuum", "uniform_over_vacuum", "channel_over_uniform"};
constexpr std::array<const char*, 3> kRatioLabels{"channel/vacuum", "uniform/vacuum", "channel/uniform"};
constexpr std::array<ImVec4, 3> kRatioColors{ImVec4(0.58f, 0.40f, 0.74f, 1.0f), ImVec4(0.55f, 0.34f, 0.29f, 1.0f),
                                             ImVec4(0.89f, 0.47f, 0.76f, 1.0f)};

struct PlotDef {
  const char* title;
  const char* ylabel;
  const char* prefix;  // "<prefix><case>" or "<prefix><ratio>"
  bool ratio;
};

void draw_grid(const char* id, const TripletData& data, std::span<const PlotDef> plots, int rows, int cols) {
  const auto x = data.column("propagation_mm");
  if (!ImPlot::BeginSubplots(fmt::format("{}_{}", id, data.label).c_str(), rows, cols, ImVec2(-1, -1),
                             ImPlotSubplotFlags_LinkAllX | ImPlotSubplotFlags_NoTitle)) {
    return;
  }
  const auto [x_min, x_max] = widgets::padded_range(x);
  for (const auto& plot : plots) {
    if (!ImPlot::BeginPlot(plot.title)) {
      continue;
    }
    ImPlot::SetupAxes("propagation distance [mm]", plot.ylabel);
    ImPlot::SetupAxisLimits(ImAxis_X1, x_min, x_max, ImPlotCond_Once);
    ImPlot::SetupLegend(ImPlotLocation_NorthWest);
    ImPlot::SetupFinish();
    if (data.late_window_mm) {
      widgets::x_band(data.late_window_mm->first, data.late_window_mm->second, ImVec4(0.95f, 0.72f, 0.25f, 0.08f));
    }
    widgets::plateau_band(data.plateau_mm, false);
    if (plot.ratio) {
      widgets::horizontal_line("##one", 1.0, ImVec4(0.7f, 0.7f, 0.7f, 0.6f));
      for (std::size_t i = 0; i < kRatios.size(); ++i) {
        const auto y = data.column(std::string(plot.prefix) + kRatios[i]);
        widgets::line_series(kRatioLabels[i], x, y, kRatioColors[i]);
      }
    } else {
      for (std::size_t i = 0; i < kCases.size(); ++i) {
        const auto y = data.column(std::string(plot.prefix) + kCases[i]);
        widgets::line_series(kCases[i], x, y, kCaseColors[i]);
      }
    }
    ImPlot::EndPlot();
  }
  ImPlot::EndSubplots();
}

}  // namespace

void draw_triplet_panel(UiState& ui, DataStore& store) {
  if (!ImGui::Begin(kTripletWindow)) {
    ImGui::End();
    return;
  }
  const auto snapshot = store.campaign();
  if (snapshot == nullptr || snapshot->triplets.empty()) {
    ImGui::TextDisabled("No triplets in the campaign.");
    ImGui::End();
    return;
  }

  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.5f);
  if (ImGui::BeginCombo("Triplet", ui.selected_triplet.empty() ? "(choose)" : ui.selected_triplet.c_str())) {
    for (const auto& triplet : snapshot->triplets) {
      const std::string label = triplet.label();
      if (ImGui::Selectable(label.c_str(), label == ui.selected_triplet)) {
        ui.selected_triplet = label;
      }
    }
    ImGui::EndCombo();
  }
  const campaign::TripletInfo* triplet = snapshot->find_triplet(ui.selected_triplet);
  if (triplet == nullptr) {
    ImGui::TextDisabled("Select a triplet here or in the Campaign panel.");
    ImGui::End();
    return;
  }

  const std::array<const std::optional<campaign::CaseInfo>*, 3> members{&triplet->channel, &triplet->uniform,
                                                                        &triplet->vacuum};
  bool all_reduced = true;
  for (std::size_t i = 0; i < members.size(); ++i) {
    const auto& member = *members[i];
    const CaseRecord* record = member ? snapshot->find_case(member->case_id) : nullptr;
    const bool reduced = record != nullptr && record->reduced_ready;
    all_reduced = all_reduced && reduced;
    ImGui::TextColored(kCaseColors[i], "%-8s", kCases[i]);
    ImGui::SameLine();
    if (!member) {
      ImGui::TextColored(widgets::kBad, "missing");
    } else {
      if (ImGui::SmallButton(fmt::format("{}##member{}", member->case_id, i).c_str())) {
        ui.selected_case = member->case_id;
        ui.focus_request = kCaseWindow;
      }
      ImGui::SameLine();
      widgets::chip(reduced ? "reduced" : "not reduced", reduced ? widgets::kGood : widgets::kWarn);
    }
  }
  if (!triplet->complete() || !all_reduced) {
    ImGui::TextDisabled("Triplet plots need all three cases with a valid guiding_metrics.csv.");
    ImGui::End();
    return;
  }

  const auto data = store.triplet(*snapshot, *triplet);
  if (data == nullptr) {
    ImGui::TextDisabled("building triplet tables...");
    ImGui::End();
    return;
  }
  if (!data->error.empty()) {
    ImGui::TextColored(widgets::kBad, "%s", data->error.c_str());
    ImGui::End();
    return;
  }

  const bool has_a0 = !data->column("a0_peak_channel").empty();
  if (ImGui::BeginTabBar("##triplet_tabs")) {
    if (ImGui::BeginTabItem("Comparison", nullptr, widgets::tab_flags(ui, "Comparison"))) {
      std::vector<PlotDef> plots{{"Optical confinement", "waist RMS [um]", "waist_um_", false},
                                 {"Peak intensity", "peak I / first dump", "peak_I_norm_", false},
                                 {"Laser energy", "energy / first dump", "energy_norm_", false},
                                 {"Wake amplitude", "max |Ez wake| [GV/m]", "Ez_wake_absmax_GVm_", false}};
      if (has_a0) {
        plots.push_back({"Peak a0", "peak a0", "a0_peak_", false});
        plots.push_back({"Front margin", "front margin [um]", "front_margin_um_", false});
      }
      draw_grid("##comparison", *data, plots, has_a0 ? 3 : 2, 2);
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Ratios", nullptr, widgets::tab_flags(ui, "Ratios"))) {
      std::vector<PlotDef> plots{{"Waist ratios (lower = stronger confinement)", "waist ratio", "waist_", true},
                                 {"Peak-intensity ratios", "peak I ratio", "peakI_", true},
                                 {"Energy-proxy ratios", "energy proxy ratio", "energy_", true},
                                 {"Wake ratios", "|Ez| ratio", "Ezabs_", true}};
      if (has_a0) {
        plots.push_back({"Peak a0 ratios", "a0 ratio", "a0_", true});
      }
      draw_grid("##ratios", *data, plots, has_a0 ? 3 : 2, 2);
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Late window", nullptr, widgets::tab_flags(ui, "Late window"))) {
      if (data->late_window_mm) {
        ImGui::TextDisabled("late window %.3f - %.3f mm (late fraction %.3f)", data->late_window_mm->first,
                            data->late_window_mm->second, snapshot->settings.late_fraction);
      }
      ImGui::SeparatorText("late summary");
      widgets::frame_table("##late_summary", data->tables.late_summary, ImGui::GetContentRegionAvail().y * 0.45f);
      ImGui::SeparatorText("late ratios");
      widgets::frame_table("##late_ratios", data->tables.late_ratios);
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Wide table", nullptr, widgets::tab_flags(ui, "Wide table"))) {
      widgets::frame_table("##wide", data->tables.wide);
      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }
  ImGui::End();
}

}  // namespace guiding::gui

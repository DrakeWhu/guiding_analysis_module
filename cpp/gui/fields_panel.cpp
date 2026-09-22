#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "panels.hpp"

namespace guiding::gui {
namespace {

// log10 image for display, recomputed only when the map changes.
const std::vector<double>& display_image(const FieldMapResult& result, bool log_scale, double& scale_min,
                                         double& scale_max) {
  static const FieldMapResult* cached_for = nullptr;
  static bool cached_log = false;
  static std::vector<double> image;
  static double cached_min = 0.0;
  static double cached_max = 1.0;
  const auto& map = result.map;
  if (cached_for != &result || cached_log != log_scale) {
    image = map.intensity;
    if (log_scale && map.intensity_max > 0.0) {
      cached_max = std::log10(map.intensity_max);
      cached_min = cached_max - 4.0;
      for (double& value : image) {
        value = value > 0.0 ? std::max(std::log10(value), cached_min) : cached_min;
      }
    } else {
      cached_min = 0.0;
      cached_max = map.intensity_max > 0.0 ? map.intensity_max : 1.0;
      for (double& value : image) {
        if (!std::isfinite(value)) {
          value = 0.0;
        }
      }
    }
    cached_for = &result;
    cached_log = log_scale;
  }
  scale_min = cached_min;
  scale_max = cached_max;
  return image;
}

void draw_maps(const FieldMapResult& result, const UiState& ui) {
  const auto& map = result.map;
  double scale_min = 0.0;
  double scale_max = 1.0;
  const auto& image = display_image(result, ui.field_log_scale, scale_min, scale_max);
  const std::string suffix = fmt::format("{}", map.iteration);
  const float scale_width = 96.0f;

  const ImVec2 available = ImGui::GetContentRegionAvail();
  const float left_width = available.x * 0.68f;
  ImGui::BeginChild("##field_left", ImVec2(left_width, 0));
  static float ratios[] = {0.5f, 0.25f, 0.25f};
  if (ImPlot::BeginSubplots(fmt::format("##field_column_{}", suffix).c_str(), 3, 1,
                            ImVec2(-scale_width, -1), ImPlotSubplotFlags_LinkAllX | ImPlotSubplotFlags_NoTitle,
                            ratios)) {
    const double z_lo = map.z_min_um;
    const double z_hi = map.z_max_um;
    if (ImPlot::BeginPlot("##intensity", ImVec2(), ImPlotFlags_NoLegend)) {
      ImPlot::SetupAxes(nullptr, "r [um]", ImPlotAxisFlags_NoGridLines, ImPlotAxisFlags_NoGridLines);
      ImPlot::SetupAxisLimits(ImAxis_X1, z_lo, z_hi, ImPlotCond_Once);
      ImPlot::SetupAxisLimits(ImAxis_Y1, map.r_min_um, map.r_max_um, ImPlotCond_Once);
      ImPlot::PushColormap(ImPlotColormap_Viridis);
      ImPlot::PlotHeatmap("|E_perp|^2", image.data(), static_cast<int>(map.rows), static_cast<int>(map.cols),
                          scale_min, scale_max, nullptr, ImPlotPoint(z_lo, map.r_min_um),
                          ImPlotPoint(z_hi, map.r_max_um));
      ImPlot::PopColormap();
      widgets::vertical_marker("##z_peak", map.laser.z_peak_um, ImVec4(1, 1, 1, 0.7f), "z_peak");
      if (std::isfinite(map.laser.waist_um)) {
        widgets::horizontal_line("##waist", map.laser.waist_um, ImVec4(1.0f, 0.55f, 0.35f, 0.8f));
      }
      ImPlot::EndPlot();
    }
    if (ImPlot::BeginPlot("##I_z", ImVec2(), ImPlotFlags_NoLegend)) {
      ImPlot::SetupAxes(nullptr, "I_z (r-weighted)");
      ImPlot::SetupAxisLimits(ImAxis_X1, z_lo, z_hi, ImPlotCond_Once);
      widgets::line_series("I_z", map.z_um, map.I_z, ImVec4(0.6f, 0.6f, 0.65f, 1.0f), false);
      widgets::line_series("I_z smoothed", map.z_um, map.I_z_smooth, widgets::kChannelColor, false);
      widgets::vertical_marker("##z_peak_iz", map.laser.z_peak_um, widgets::kWarn, nullptr);
      ImPlot::EndPlot();
    }
    if (ImPlot::BeginPlot("##Ez_axis", ImVec2(), ImPlotFlags_NoLegend)) {
      ImPlot::SetupAxes("z [um]", "E_z on axis [GV/m]");
      ImPlot::SetupAxisLimits(ImAxis_X1, z_lo, z_hi, ImPlotCond_Once);
      ImPlot::SetupFinish();
      // Wake window of the reduction: [z_peak - wake_behind_um, z_peak - wake_gap_um].
      const physics::FieldParams params;
      widgets::x_band(map.laser.z_peak_um - params.wake_behind_um, map.laser.z_peak_um - params.wake_gap_um,
                      ImVec4(0.95f, 0.72f, 0.25f, 0.08f));
      widgets::horizontal_line("##zero", 0.0, ImVec4(0.6f, 0.6f, 0.6f, 0.4f));
      widgets::line_series("E_z", map.z_um, map.Ez_axis_GVm, widgets::kUniformColor, false);
      if (std::isfinite(map.wake.z_Ez_absmax_um)) {
        widgets::vertical_marker("##ez_absmax", map.wake.z_Ez_absmax_um, widgets::kBad, "|Ez| max");
      }
      ImPlot::EndPlot();
    }
    ImPlot::EndSubplots();
  }
  ImGui::SameLine();
  ImPlot::PushColormap(ImPlotColormap_Viridis);
  ImPlot::ColormapScale(ui.field_log_scale ? "log10 |E_perp|^2" : "|E_perp|^2", scale_min, scale_max,
                        ImVec2(scale_width - 8.0f, ImGui::GetContentRegionAvail().y * 0.5f), "%.2g");
  ImPlot::PopColormap();
  ImGui::EndChild();

  ImGui::SameLine();
  ImGui::BeginChild("##field_right", ImVec2(0, 0));
  if (ImPlot::BeginPlot(fmt::format("Radial profile at z_peak##{}", suffix).c_str(), ImVec2(-1, ImGui::GetContentRegionAvail().y * 0.55f),
                        ImPlotFlags_NoLegend)) {
    ImPlot::SetupAxes("r [um]", "sum I over |z - z_peak| <= 5 um");
    widgets::line_series("P(r)", map.r_um, map.P_r, widgets::kChannelColor);
    if (std::isfinite(map.laser.waist_um)) {
      widgets::vertical_marker("##waist_r", map.laser.waist_um, ImVec4(1.0f, 0.55f, 0.35f, 0.9f), "waist");
    }
    ImPlot::EndPlot();
  }
  if (ImGui::BeginTable("##field_metrics", 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_Borders)) {
    const auto row = [](const char* name, const std::string& value) {
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::TextUnformatted(name);
      ImGui::TableSetColumnIndex(1);
      ImGui::TextUnformatted(value.c_str());
    };
    row("time", fmt::format("{:.1f} fs", map.time_fs));
    row("z_peak", fmt::format("{:.2f} um", map.laser.z_peak_um));
    row("front margin", fmt::format("{:.2f} um", map.laser.front_margin_um));
    row("waist (rms)", fmt::format("{:.3f} um", map.laser.waist_um));
    row("a0 peak", fmt::format("{:.4f}", map.laser.a0_peak));
    row("E_perp peak", fmt::format("{:.4g} V/m", map.laser.Eperp_peak_Vm));
    row("energy proxy", fmt::format("{:.4g}", map.laser.energy_proxy));
    row("|Ez wake| max", fmt::format("{:.3f} GV/m", map.wake.Ez_wake_absmax / 1.0e9));
    row("z(|Ez| max) - z_peak", fmt::format("{:.2f} um", map.wake.z_Ez_absmax_rel_um));
    row("grid (shown)", fmt::format("{} x {} (of {} x {})", map.rows, map.cols, map.r_um.size(), map.z_um.size()));
    ImGui::EndTable();
  }
  ImGui::EndChild();
}

}  // namespace

void draw_fields_panel(App& app, UiState& ui, DataStore& store) {
  if (!ImGui::Begin(kFieldsWindow)) {
    ImGui::End();
    return;
  }
  const auto snapshot = store.campaign();
  const CaseRecord* record = snapshot != nullptr ? snapshot->find_case(ui.selected_case) : nullptr;
  if (record == nullptr) {
    ImGui::TextDisabled("Select a case in the Campaign panel.");
    ImGui::End();
    return;
  }
  ImGui::TextColored(widgets::case_type_color(record->info.case_type), "%s", record->info.case_id.c_str());
  ImGui::SameLine();
  ImGui::TextDisabled("| %s", record->info.diag_dir.string().c_str());

  const auto listing = store.field_series(*record);
  if (listing == nullptr) {
    ImGui::TextDisabled("listing dumps...");
    ImGui::End();
    return;
  }
  if (!listing->error.empty()) {
    ImGui::TextColored(widgets::kBad, "%s", listing->error.c_str());
    ImGui::End();
    return;
  }
  const std::int64_t iteration =
      widgets::iteration_selector("field_iteration", listing->iterations, ui.field_iteration, ui.field_follow_latest);
  ImGui::SameLine();
  ImGui::Checkbox("log scale", &ui.field_log_scale);

  const auto result = iteration >= 0 ? store.field_map(*listing, iteration) : nullptr;
  if (widgets::export_buttons(app, kFieldsWindow, result != nullptr && result->error.empty())) {
    const auto& map = result->map;
    const std::string stem = fmt::format("fields_{}_it{}", record->info.case_id, map.iteration);
    store.write_text_async(app.export_path(stem + "_z", "csv"),
                           widgets::columns_csv({{"z_um", map.z_um},
                                                 {"I_z", map.I_z},
                                                 {"I_z_smooth", map.I_z_smooth},
                                                 {"Ez_axis_GVm", map.Ez_axis_GVm}}));
    store.write_text_async(app.export_path(stem + "_r", "csv"), widgets::columns_csv({{"r_um", map.r_um}, {"P_r", map.P_r}}));
  }
  if (result == nullptr) {
    ImGui::TextDisabled("reading iteration %lld ...", static_cast<long long>(iteration));
  } else if (!result->error.empty()) {
    ImGui::TextColored(widgets::kBad, "%s", result->error.c_str());
  } else {
    draw_maps(*result, ui);
  }
  ImGui::End();
}

}  // namespace guiding::gui

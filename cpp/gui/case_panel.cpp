#include <array>
#include <cmath>
#include <span>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "panels.hpp"

namespace guiding::gui {
namespace {

struct SummaryPanel {
  std::string label;
  const std::vector<double>* values;
};

void draw_summary_plots(const CaseMetrics& metrics, const ImVec4& color) {
  const std::string ref = fmt::format("dump {}", metrics.ref_iteration);
  const std::array<SummaryPanel, 6> panels{{
      {"front margin [um]", &metrics.front_margin_um},
      {"laser waist rms [um]", &metrics.waist_um},
      {metrics.has_a0 ? "a0 / " + ref : "peak I / " + ref, metrics.has_a0 ? &metrics.a0_norm : &metrics.peak_I_norm},
      {"energy / " + ref, &metrics.energy_norm},
      {"max |Ez wake| [GV/m]", &metrics.Ez_absmax_GVm},
      {"z(Ez absmax) - z_peak [um]", &metrics.z_Ez_rel_um},
  }};

  // IDs carry the case so a newly selected case starts with fitted axes, while
  // zoom and pan stay free afterwards.
  if (!ImPlot::BeginSubplots(fmt::format("##case_summary_{}", metrics.case_id).c_str(), 3, 2, ImVec2(-1, -1),
                             ImPlotSubplotFlags_LinkAllX | ImPlotSubplotFlags_NoTitle)) {
    return;
  }
  const auto [x_min, x_max] = widgets::padded_range(metrics.propagation_mm);
  for (std::size_t i = 0; i < panels.size(); ++i) {
    const auto& panel = panels[i];
    if (ImPlot::BeginPlot(fmt::format("##summary{}", i).c_str(), ImVec2(), ImPlotFlags_NoLegend)) {
      const bool bottom = i >= 4;
      ImPlot::SetupAxes(bottom ? "propagation distance [mm]" : nullptr, panel.label.c_str());
      ImPlot::SetupAxisLimits(ImAxis_X1, x_min, x_max, ImPlotCond_Once);
      ImPlot::SetupFinish();
      widgets::plateau_band(metrics.plateau_mm, i == 0);
      if (i == 0) {
        for (double level : {20.0, 30.0, 50.0}) {
          widgets::horizontal_line(fmt::format("##margin{}", level).c_str(), level, ImVec4(0.6f, 0.6f, 0.6f, 0.35f));
        }
      }
      widgets::line_series(panel.label.c_str(), metrics.propagation_mm, *panel.values, color);
      if (metrics.breakdown_mm) {
        widgets::vertical_marker("##breakdown", *metrics.breakdown_mm, widgets::kBad, i == 0 ? "breakdown?" : nullptr);
      }
      ImPlot::EndPlot();
    }
  }
  ImPlot::EndSubplots();
}

void draw_singlecase(const CaseMetrics& metrics) {
  const auto& table = metrics.singlecase;
  if (table.columns.empty() || table.rows.empty()) {
    ImGui::TextDisabled("No guiding_singlecase_score.csv next to the metrics.");
    return;
  }
  const auto& row = table.rows.front();
  const auto find = [&](std::string_view name) -> std::string {
    const auto index = table.column_index(name);
    return index && *index < row.size() ? row[*index] : std::string();
  };
  const std::string status = find("metric_guiding_singlecase_status");
  const std::string score = find("metric_guiding_singlecase_score_v1");
  ImGui::TextColored(status == "ok" ? widgets::kGood : widgets::kBad, "status: %s", status.c_str());
  ImGui::SameLine();
  ImGui::Text("   score v1: %s", score.empty() ? "-" : score.c_str());
  const std::string reason = find("metric_guiding_singlecase_failure_reason");
  if (!reason.empty()) {
    ImGui::TextColored(widgets::kWarn, "reason: %s", reason.c_str());
  }
  if (ImGui::BeginTable("##singlecase", 2, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY)) {
    ImGui::TableSetupScrollFreeze(0, 1);
    ImGui::TableSetupColumn("field", ImGuiTableColumnFlags_WidthStretch, 2.0f);
    ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch, 3.0f);
    ImGui::TableHeadersRow();
    for (std::size_t c = 0; c < table.columns.size(); ++c) {
      ImGui::TableNextRow();
      ImGui::TableSetColumnIndex(0);
      ImGui::TextUnformatted(table.columns[c].c_str());
      ImGui::TableSetColumnIndex(1);
      ImGui::TextUnformatted(c < row.size() ? row[c].c_str() : "");
    }
    ImGui::EndTable();
  }
}

void draw_data_table(const CaseMetrics& metrics) {
  struct Column {
    const char* name;
    const std::vector<double>* values;
  };
  const std::array<Column, 9> columns{{{"iteration", &metrics.iteration},
                                       {"propagation_mm", &metrics.propagation_mm},
                                       {"z_peak_um", &metrics.z_peak_um},
                                       {"front_margin_um", &metrics.front_margin_um},
                                       {"waist_um", &metrics.waist_um},
                                       {"a0_peak", &metrics.a0_peak},
                                       {"energy_norm", &metrics.energy_norm},
                                       {"Ez_absmax_GVm", &metrics.Ez_absmax_GVm},
                                       {"z_Ez_rel_um", &metrics.z_Ez_rel_um}}};
  const ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY |
                                ImGuiTableFlags_ScrollX | ImGuiTableFlags_SizingFixedFit;
  if (!ImGui::BeginTable("##case_data", static_cast<int>(columns.size()), flags)) {
    return;
  }
  ImGui::TableSetupScrollFreeze(1, 1);
  for (const auto& column : columns) {
    ImGui::TableSetupColumn(column.name);
  }
  ImGui::TableHeadersRow();
  ImGuiListClipper clipper;
  clipper.Begin(static_cast<int>(metrics.rows));
  while (clipper.Step()) {
    for (int row = clipper.DisplayStart; row < clipper.DisplayEnd; ++row) {
      ImGui::TableNextRow();
      for (std::size_t c = 0; c < columns.size(); ++c) {
        ImGui::TableSetColumnIndex(static_cast<int>(c));
        const double value = (*columns[c].values)[static_cast<std::size_t>(row)];
        if (std::isnan(value)) {
          ImGui::TextDisabled("nan");
        } else {
          ImGui::Text(c == 0 ? "%.0f" : "%.6g", value);
        }
      }
    }
  }
  ImGui::EndTable();
}

}  // namespace

void draw_case_panel(App& app, UiState& ui, DataStore& store) {
  if (!ImGui::Begin(kCaseWindow)) {
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

  const ImVec4 color = widgets::case_type_color(record->info.case_type);
  ImGui::TextColored(color, "%s", campaign::case_type_name(record->info.case_type));
  ImGui::SameLine();
  ImGui::TextUnformatted(record->info.case_id.c_str());
  ImGui::SameLine();
  ImGui::TextDisabled("| h5 %lld | newest %s", static_cast<long long>(record->info.h5_count),
                      record->age_min ? fmt::format("{:.1f} min ago", *record->age_min).c_str() : "-");
  if (record->singlecase_score) {
    ImGui::SameLine();
    ImGui::TextDisabled("| score %.1f", *record->singlecase_score);
  }
  ImGui::SameLine(ImGui::GetContentRegionMax().x - ImGui::CalcTextSize("Reduce  Re-reduce").x -
                  4 * ImGui::GetStyle().FramePadding.x);
  ImGui::BeginDisabled(!record->raw_ready());
  if (ImGui::Button("Reduce")) {
    store.reduce_cases({record->info}, app.settings(), false);
  }
  ImGui::SameLine();
  if (ImGui::Button("Re-reduce")) {
    store.reduce_cases({record->info}, app.settings(), true);
  }
  ImGui::EndDisabled();

  if (!record->reduced_ready) {
    ImGui::Separator();
    ImGui::TextColored(widgets::kWarn, "No valid guiding_metrics.csv yet:");
    ImGui::TextDisabled("%s", record->metrics_csv.string().c_str());
    ImGui::TextWrapped("%s", record->raw_ready()
                                 ? "The HDF5 diagnostic is ready: press Reduce to compute the metrics here."
                                 : "The diagnostic does not meet the readiness gate (min h5 / newest-file age) yet.");
    ImGui::End();
    return;
  }

  const auto metrics = store.case_metrics(*record);
  if (metrics == nullptr) {
    ImGui::TextDisabled("loading %s ...", record->metrics_csv.filename().string().c_str());
    ImGui::End();
    return;
  }
  if (metrics->stamp != record->metrics_stamp) {
    ImGui::SameLine();
    ImGui::TextColored(widgets::kWarn, "(reloading)");
  }
  if (!metrics->error.empty()) {
    ImGui::TextColored(widgets::kBad, "%s", metrics->error.c_str());
  }
  ImGui::TextDisabled("%zu dumps | plateau %s | %s", metrics->rows,
                      metrics->plateau_mm
                          ? fmt::format("[{:.2f}, {:.2f}] mm", metrics->plateau_mm->first, metrics->plateau_mm->second).c_str()
                          : "unknown",
                      metrics->breakdown_mm ? fmt::format("tentative breakdown at {:.3f} mm (iteration {})",
                                                          *metrics->breakdown_mm, *metrics->breakdown_iteration)
                                                  .c_str()
                                            : "no tentative breakdown");

  if (ImGui::BeginTabBar("##case_tabs")) {
    if (ImGui::BeginTabItem("Summary", nullptr, widgets::tab_flags(ui, "Summary"))) {
      draw_summary_plots(*metrics, color);
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Single-case score", nullptr, widgets::tab_flags(ui, "Single-case score"))) {
      draw_singlecase(*metrics);
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem("Data", nullptr, widgets::tab_flags(ui, "Data"))) {
      draw_data_table(*metrics);
      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }
  ImGui::End();
}

}  // namespace guiding::gui

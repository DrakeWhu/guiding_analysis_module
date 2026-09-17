#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "guiding/products/particle_reduction.hpp"
#include "panels.hpp"

namespace guiding::gui {
namespace {

bool input_text(const char* label, std::string& value, float width) {
  char buffer[512];
  const std::size_t length = std::min(value.size(), sizeof(buffer) - 1);
  std::copy_n(value.data(), length, buffer);
  buffer[length] = '\0';
  ImGui::SetNextItemWidth(width);
  if (ImGui::InputText(label, buffer, sizeof(buffer), ImGuiInputTextFlags_EnterReturnsTrue)) {
    value = buffer;
    return true;
  }
  return false;
}

std::string cell_text(const table::Cell& cell) {
  struct Visitor {
    std::string operator()(std::monostate) const { return ""; }
    std::string operator()(bool v) const { return v ? "True" : "False"; }
    std::string operator()(std::int64_t v) const { return std::to_string(v); }
    std::string operator()(double v) const { return fmt::format("{:.6g}", v); }
    std::string operator()(const std::string& v) const { return v; }
  };
  return std::visit(Visitor{}, cell);
}

double record_number(const table::Record& record, const char* key) {
  const auto cell = record.get(key);
  if (!cell) {
    return std::nan("");
  }
  if (const auto* real = std::get_if<double>(&*cell)) {
    return *real;
  }
  if (const auto* integer = std::get_if<std::int64_t>(&*cell)) {
    return static_cast<double>(*integer);
  }
  return std::nan("");
}

void draw_spectrum_and_acceptance(const products::ParticleView& view, UiState& ui) {
  const float half = ImGui::GetContentRegionAvail().y * 0.5f;
  if (ImPlot::BeginPlot(fmt::format("Charge spectrum##{}{}{}", view.species_scope, view.iteration,
                                    ui.particle_log_spectrum).c_str(),
                        ImVec2(-1, half))) {
    ImPlot::SetupAxes("kinetic energy [MeV]", "dQ/dE [pC/MeV]");
    if (ui.particle_log_spectrum) {
      // Empty bins would drag a log axis fit towards zero: show four decades.
      const double peak = *std::max_element(view.spectrum_all.begin(), view.spectrum_all.end());
      ImPlot::SetupAxisScale(ImAxis_Y1, ImPlotScale_Log10);
      if (peak > 0.0) {
        ImPlot::SetupAxisLimits(ImAxis_Y1, peak * 1.0e-4, peak * 2.0, ImPlotCond_Once);
      }
    }
    const int bins = static_cast<int>(view.spectrum_all.size());
    ImPlotSpec all;
    all.LineColor = widgets::kMuted;
    ImPlot::PlotStairs("all valid", view.energy_edges_mev.data(), view.spectrum_all.data(), bins, all);
    ImPlotSpec hot;
    hot.LineColor = widgets::kChannelColor;
    hot.FillColor = widgets::kChannelColor;
    hot.FillAlpha = 0.25f;
    hot.Flags = ImPlotStairsFlags_Shaded;
    ImPlot::PlotStairs("hot", view.energy_edges_mev.data(), view.spectrum_hot.data(), bins, hot);
    widgets::vertical_marker("##hot_threshold", ui.particle_hot_energy_mev, widgets::kWarn, "hot threshold");
    ImPlot::EndPlot();
  }

  const std::size_t n_theta = view.theta_cuts_mrad.size();
  const std::size_t n_energy = view.energy_cuts_mev.size();
  if (n_theta == 0 || n_energy == 0 || view.accepted_pC.size() != n_theta * n_energy) {
    return;
  }
  const float width = ImGui::GetContentRegionAvail().x * 0.5f;
  if (ImPlot::BeginPlot(fmt::format("Accepted charge [pC]##grid{}{}", view.species_scope, view.iteration).c_str(),
                        ImVec2(width, -1), ImPlotFlags_NoLegend | ImPlotFlags_NoMouseText)) {
    std::vector<std::string> energy_labels;
    std::vector<std::string> theta_labels;
    for (double e : view.energy_cuts_mev) {
      energy_labels.push_back(fmt::format(">={:g}", e));
    }
    for (double t : view.theta_cuts_mrad) {
      theta_labels.push_back(fmt::format("<={:g}", t));
    }
    std::vector<const char*> energy_ptrs;
    std::vector<const char*> theta_ptrs;
    for (const auto& label : energy_labels) {
      energy_ptrs.push_back(label.c_str());
    }
    // Heatmap row 0 is drawn at the top: largest theta cut first.
    std::vector<double> grid(view.accepted_pC.size());
    for (std::size_t t = 0; t < n_theta; ++t) {
      std::copy_n(view.accepted_pC.begin() + static_cast<std::ptrdiff_t>((n_theta - 1 - t) * n_energy), n_energy,
                  grid.begin() + static_cast<std::ptrdiff_t>(t * n_energy));
    }
    for (std::size_t t = 0; t < n_theta; ++t) {
      theta_ptrs.push_back(theta_labels[t].c_str());
    }
    ImPlot::SetupAxes("E_min [MeV]", "theta_r cut [mrad]", ImPlotAxisFlags_NoGridLines, ImPlotAxisFlags_NoGridLines);
    ImPlot::SetupAxesLimits(0.0, static_cast<double>(n_energy), 0.0, static_cast<double>(n_theta), ImPlotCond_Always);
    ImPlot::SetupAxisTicks(ImAxis_X1, 0.5, static_cast<double>(n_energy) - 0.5, static_cast<int>(n_energy),
                           energy_ptrs.data());
    ImPlot::SetupAxisTicks(ImAxis_Y1, 0.5, static_cast<double>(n_theta) - 0.5, static_cast<int>(n_theta),
                           theta_ptrs.data());
    const double max_charge = *std::max_element(grid.begin(), grid.end());
    ImPlot::PushColormap(ImPlotColormap_Viridis);
    ImPlot::PlotHeatmap("accepted", grid.data(), static_cast<int>(n_theta), static_cast<int>(n_energy), 0.0,
                        max_charge > 0.0 ? max_charge : 1.0, "%.3g", ImPlotPoint(0, 0),
                        ImPlotPoint(static_cast<double>(n_energy), static_cast<double>(n_theta)));
    ImPlot::PopColormap();
    ImPlot::EndPlot();
  }
  ImGui::SameLine();
  if (ImPlot::BeginPlot(fmt::format("Q(E >= E_min, theta_r <= cut)##lines{}{}", view.species_scope, view.iteration).c_str(),
                        ImVec2(-1, -1))) {
    ImPlot::SetupAxes("E_min [MeV]", "accepted charge [pC]");
    ImPlot::SetupLegend(ImPlotLocation_NorthEast);
    for (std::size_t t = 0; t < n_theta; ++t) {
      const std::span<const double> values(view.accepted_pC.data() + t * n_energy, n_energy);
      ImPlotSpec spec;
      spec.Marker = ImPlotMarker_Circle;
      spec.MarkerSize = 3.0f;
      ImPlot::PlotLine(fmt::format("<= {:g} mrad", view.theta_cuts_mrad[t]).c_str(), view.energy_cuts_mev.data(),
                       values.data(), static_cast<int>(n_energy), spec);
    }
    ImPlot::EndPlot();
  }
}

void draw_phase_spaces(const products::ParticleView& view, const UiState& ui) {
  if (view.n_hot == 0) {
    ImGui::TextDisabled("No hot electrons in this selection.");
    return;
  }
  const int count = static_cast<int>(view.phase_spaces.size());
  const int cols = count > 2 ? 3 : count;
  const int rows = (count + cols - 1) / cols;
  if (!ImPlot::BeginSubplots(fmt::format("##phase_{}_{}", view.species_scope, view.iteration).c_str(), rows, cols,
                             ImVec2(-1, -1), ImPlotSubplotFlags_NoTitle)) {
    return;
  }
  for (const auto& histogram : view.phase_spaces) {
    if (!ImPlot::BeginPlot(histogram.title.c_str(), ImVec2(), ImPlotFlags_NoLegend)) {
      continue;
    }
    ImPlot::SetupAxes(histogram.x_label.c_str(), histogram.y_label.c_str(), ImPlotAxisFlags_NoGridLines,
                      ImPlotAxisFlags_NoGridLines);
    ImPlot::SetupAxesLimits(histogram.x_min, histogram.x_max, histogram.y_min, histogram.y_max, ImPlotCond_Once);
    std::vector<double> values = histogram.charge_pC;
    double scale_min = 0.0;
    double scale_max = histogram.max_charge_pC > 0.0 ? histogram.max_charge_pC : 1.0;
    if (ui.particle_log_phase && histogram.max_charge_pC > 0.0) {
      scale_max = std::log10(histogram.max_charge_pC);
      scale_min = scale_max - 3.0;
      for (double& value : values) {
        value = value > 0.0 ? std::max(std::log10(value), scale_min) : scale_min;
      }
    }
    ImPlot::PushColormap(ImPlotColormap_Viridis);
    ImPlot::PlotHeatmap("charge", values.data(), static_cast<int>(histogram.ny), static_cast<int>(histogram.nx),
                        scale_min, scale_max, nullptr, ImPlotPoint(histogram.x_min, histogram.y_min),
                        ImPlotPoint(histogram.x_max, histogram.y_max));
    ImPlot::PopColormap();
    ImPlot::EndPlot();
  }
  ImPlot::EndSubplots();
}

void draw_summary(const products::ParticleView& view) {
  const auto& summary = view.summary;
  const auto status = summary.get("beamlike_status");
  ImGui::Text("hot charge %.2f pC | E95 hot %.1f MeV | beamlike score %.1f (%s) | soft50 charge %.2f pC",
              record_number(summary, "charge_hot_pC"), record_number(summary, "E95_hot_MeV"),
              record_number(summary, "beamlike_score"), status ? cell_text(*status).c_str() : "-",
              record_number(summary, "charge_soft50_pC"));
  if (!ImGui::BeginTable("##particle_summary", 2,
                         ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY)) {
    return;
  }
  ImGui::TableSetupScrollFreeze(0, 1);
  ImGui::TableSetupColumn("particle_summary.csv column", ImGuiTableColumnFlags_WidthStretch, 2.0f);
  ImGui::TableSetupColumn("value", ImGuiTableColumnFlags_WidthStretch, 3.0f);
  ImGui::TableHeadersRow();
  for (const auto& [key, value] : summary.items()) {
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    ImGui::TextUnformatted(key.c_str());
    ImGui::TableSetColumnIndex(1);
    ImGui::TextUnformatted(cell_text(value).c_str());
  }
  ImGui::EndTable();
}

}  // namespace

void draw_particles_panel(App& app, UiState& ui, DataStore& store) {
  if (!ImGui::Begin(kParticlesWindow)) {
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

  std::vector<std::string> species;
  try {
    species = products::parse_species_list(ui.particle_species);
  } catch (const std::exception&) {
    species = {"electrons"};
  }

  ImGui::TextColored(widgets::case_type_color(record->info.case_type), "%s", record->info.case_id.c_str());
  ImGui::SameLine();
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("species");
  ImGui::SameLine();
  input_text("##species", ui.particle_species, 220.0f);
  ImGui::SameLine();
  ImGui::TextUnformatted("diag");
  ImGui::SameLine();
  input_text("##diag_name", ui.particle_diag_name, 160.0f);

  const auto listing = store.particle_series(*record, species, ui.particle_diag_name);
  if (listing == nullptr) {
    ImGui::TextDisabled("looking for the particle diagnostic...");
    ImGui::End();
    return;
  }
  if (!listing->error.empty()) {
    ImGui::TextColored(widgets::kWarn, "%s", listing->error.c_str());
    ImGui::End();
    return;
  }

  const std::int64_t iteration = widgets::iteration_selector("particle_iteration", listing->iterations,
                                                             ui.particle_iteration, ui.particle_follow_latest);
  ImGui::SameLine();
  ImGui::SetNextItemWidth(90.0f);
  ImGui::InputDouble("hot E [MeV]", &ui.particle_hot_energy_mev, 0.0, 0.0, "%.1f",
                     ImGuiInputTextFlags_EnterReturnsTrue);
  ImGui::SameLine();
  ImGui::Checkbox("forward", &ui.particle_forward_only);
  ImGui::SameLine();
  ImGui::SetNextItemWidth(50.0f);
  ImGui::Combo("axis", &ui.particle_longitudinal, "x\0y\0z\0");
  ImGui::SameLine();
  ImGui::Checkbox("exit window", &ui.particle_use_exit_window);
  if (ui.particle_use_exit_window) {
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80.0f);
    ImGui::InputDouble("mm##window", &ui.particle_exit_window_mm, 0.0, 0.0, "%.3f", ImGuiInputTextFlags_EnterReturnsTrue);
    ui.particle_exit_window_mm = std::max(1.0e-6, ui.particle_exit_window_mm);
  }

  products::ParticleViewOptions options;
  options.hot_energy_mev = ui.particle_hot_energy_mev;
  options.forward_only = ui.particle_forward_only;
  options.longitudinal = static_cast<physics::Longitudinal>(std::clamp(ui.particle_longitudinal, 0, 2));
  if (ui.particle_use_exit_window) {
    options.exit_window_mm = ui.particle_exit_window_mm;
  }
  const auto result = iteration >= 0 ? store.particle_views(*listing, species, iteration, options) : nullptr;

  const products::ParticleView* view = nullptr;
  if (result != nullptr && result->error.empty() && !result->scopes.empty()) {
    const auto found = std::find_if(result->scopes.begin(), result->scopes.end(),
                                    [&](const products::ParticleView& v) { return v.species_scope == ui.particle_scope; });
    view = found != result->scopes.end() ? &*found : &result->scopes.front();
    ui.particle_scope = view->species_scope;
  }

  ImGui::TextDisabled("%s | species in file: %zu", listing->diag.string().c_str(), listing->species.size());
  if (view != nullptr && result->scopes.size() > 1) {
    ImGui::SameLine();
    ImGui::SetNextItemWidth(180.0f);
    if (ImGui::BeginCombo("scope", ui.particle_scope.c_str())) {
      for (const auto& scope : result->scopes) {
        if (ImGui::Selectable(scope.species_scope.c_str(), scope.species_scope == ui.particle_scope)) {
          ui.particle_scope = scope.species_scope;
        }
      }
      ImGui::EndCombo();
    }
  }
  if (widgets::export_buttons(app, kParticlesWindow, view != nullptr)) {
    const std::string stem = fmt::format("particles_{}_it{}_{}", record->info.case_id, view->iteration, view->species_scope);
    std::vector<double> centers;
    for (std::size_t b = 0; b + 1 < view->energy_edges_mev.size(); ++b) {
      centers.push_back(0.5 * (view->energy_edges_mev[b] + view->energy_edges_mev[b + 1]));
    }
    store.write_text_async(app.export_path(stem + "_spectrum", "csv"),
                           widgets::columns_csv({{"energy_MeV", centers},
                                                 {"dQdE_all_pC_per_MeV", view->spectrum_all},
                                                 {"dQdE_hot_pC_per_MeV", view->spectrum_hot}}));
    std::vector<double> theta;
    std::vector<double> energy;
    for (double t : view->theta_cuts_mrad) {
      for (double e : view->energy_cuts_mev) {
        theta.push_back(t);
        energy.push_back(e);
      }
    }
    store.write_text_async(app.export_path(stem + "_acceptance", "csv"),
                           widgets::columns_csv({{"theta_cut_mrad", theta},
                                                 {"E_min_MeV", energy},
                                                 {"accepted_charge_pC", view->accepted_pC}}));
  }

  if (result == nullptr) {
    ImGui::TextDisabled("reading iteration %lld ...", static_cast<long long>(iteration));
  } else if (!result->error.empty()) {
    ImGui::TextColored(widgets::kBad, "%s", result->error.c_str());
  } else if (view != nullptr) {
    ImGui::Text("%s | t = %.1f fs | %zu macroparticles, %zu hot", view->species_scope.c_str(), view->time_fs,
                view->n_total, view->n_hot);
    if (ImGui::BeginTabBar("##particle_tabs")) {
      if (ImGui::BeginTabItem("Spectrum & acceptance", nullptr, widgets::tab_flags(ui, "Spectrum & acceptance"))) {
        ImGui::Checkbox("log dQ/dE", &ui.particle_log_spectrum);
        draw_spectrum_and_acceptance(*view, ui);
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Phase space", nullptr, widgets::tab_flags(ui, "Phase space"))) {
        ImGui::Checkbox("log charge", &ui.particle_log_phase);
        draw_phase_spaces(*view, ui);
        ImGui::EndTabItem();
      }
      if (ImGui::BeginTabItem("Summary", nullptr, widgets::tab_flags(ui, "Summary"))) {
        draw_summary(*view);
        ImGui::EndTabItem();
      }
      ImGui::EndTabBar();
    }
  }
  ImGui::End();
}

}  // namespace guiding::gui

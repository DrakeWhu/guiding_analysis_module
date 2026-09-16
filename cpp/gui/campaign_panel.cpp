#include <algorithm>
#include <cctype>
#include <cmath>
#include <vector>

#include <fmt/format.h>

#include "panels.hpp"

namespace guiding::gui {
namespace {

bool input_string(const char* label, std::string& value, ImGuiInputTextFlags flags = 0) {
  char buffer[2048];
  const std::size_t length = std::min(value.size(), sizeof(buffer) - 1);
  std::copy_n(value.data(), length, buffer);
  buffer[length] = '\0';
  if (ImGui::InputText(label, buffer, sizeof(buffer), flags)) {
    value = buffer;
    return true;
  }
  return false;
}

bool contains_case_insensitive(std::string_view text, std::string_view needle) {
  if (needle.empty()) {
    return true;
  }
  const auto it = std::search(text.begin(), text.end(), needle.begin(), needle.end(), [](char a, char b) {
    return std::tolower(static_cast<unsigned char>(a)) == std::tolower(static_cast<unsigned char>(b));
  });
  return it != text.end();
}

bool type_visible(const UiState& ui, campaign::CaseType type) {
  switch (type) {
    case campaign::CaseType::Channel:
      return ui.show_channel;
    case campaign::CaseType::Uniform:
      return ui.show_uniform;
    case campaign::CaseType::Vacuum:
      return ui.show_vacuum;
  }
  return true;
}

void case_state_chips(const CaseRecord& record) {
  if (record.reduced_ready) {
    widgets::chip("reduced", widgets::kGood);
  } else if (record.raw_ready()) {
    widgets::chip("raw", widgets::kWarn);
  } else if (!record.has_min_h5) {
    widgets::chip("h5<min", widgets::kBad);
  } else {
    widgets::chip("writing", widgets::kWarn);
  }
}

enum CaseColumn { kColumnCase, kColumnType, kColumnH5, kColumnAge, kColumnState, kColumnScore };

void sort_cases(std::vector<const CaseRecord*>& rows, ImGuiTableSortSpecs* specs) {
  if (specs == nullptr || specs->SpecsCount == 0) {
    return;
  }
  const ImGuiTableColumnSortSpecs& spec = specs->Specs[0];
  const bool ascending = spec.SortDirection == ImGuiSortDirection_Ascending;
  const auto key = [&](const CaseRecord* r) -> double {
    switch (spec.ColumnUserID) {
      case kColumnType:
        return static_cast<double>(r->info.case_type);
      case kColumnH5:
        return static_cast<double>(r->info.h5_count);
      case kColumnAge:
        return r->age_min.value_or(-1.0);
      case kColumnState:
        return r->reduced_ready ? 2.0 : (r->raw_ready() ? 1.0 : 0.0);
      case kColumnScore:
        return r->singlecase_score.value_or(-1.0);
      default:
        return 0.0;
    }
  };
  std::stable_sort(rows.begin(), rows.end(), [&](const CaseRecord* a, const CaseRecord* b) {
    if (spec.ColumnUserID == kColumnCase) {
      return ascending ? a->info.case_id < b->info.case_id : a->info.case_id > b->info.case_id;
    }
    return ascending ? key(a) < key(b) : key(a) > key(b);
  });
}

void draw_cases_table(App& app, UiState& ui, DataStore& store, const CampaignSnapshot& snapshot) {
  std::vector<const CaseRecord*> rows;
  for (const auto& record : snapshot.cases) {
    if (type_visible(ui, record.info.case_type) && contains_case_insensitive(record.info.case_id, ui.case_filter) &&
        (!ui.usable_only || record.usable())) {
      rows.push_back(&record);
    }
  }

  const ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY |
                                ImGuiTableFlags_Sortable | ImGuiTableFlags_Resizable |
                                ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_Hideable;
  if (!ImGui::BeginTable("##cases", 6, flags, ImVec2(0, -ImGui::GetFrameHeightWithSpacing() * 1.2f))) {
    return;
  }
  ImGui::TableSetupScrollFreeze(0, 1);
  ImGui::TableSetupColumn("Case", ImGuiTableColumnFlags_DefaultSort | ImGuiTableColumnFlags_WidthStretch, 4.0f,
                          kColumnCase);
  ImGui::TableSetupColumn("Type", ImGuiTableColumnFlags_WidthFixed, 0.0f, kColumnType);
  ImGui::TableSetupColumn("h5", ImGuiTableColumnFlags_WidthFixed, 0.0f, kColumnH5);
  ImGui::TableSetupColumn("Age [min]", ImGuiTableColumnFlags_WidthFixed, 0.0f, kColumnAge);
  ImGui::TableSetupColumn("State", ImGuiTableColumnFlags_WidthFixed, 0.0f, kColumnState);
  ImGui::TableSetupColumn("Score", ImGuiTableColumnFlags_WidthFixed, 0.0f, kColumnScore);
  ImGui::TableHeadersRow();
  sort_cases(rows, ImGui::TableGetSortSpecs());

  for (const CaseRecord* record : rows) {
    const auto& id = record->info.case_id;
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(kColumnCase);
    const bool selected = ui.selected_case == id || ui.marked_cases.contains(id);
    if (ImGui::Selectable(id.c_str(), selected, ImGuiSelectableFlags_SpanAllColumns)) {
      if (ImGui::GetIO().KeyCtrl) {
        if (!ui.marked_cases.erase(id)) {
          ui.marked_cases.insert(id);
        }
      } else {
        ui.marked_cases.clear();
        ui.selected_case = id;
        ui.focus_request = kCaseWindow;
      }
    }
    if (ImGui::BeginPopupContextItem()) {
      ui.selected_case = id;
      if (ImGui::MenuItem("Reduce (keep existing CSV)")) {
        store.reduce_cases({record->info}, app.settings(), false);
      }
      if (ImGui::MenuItem("Re-reduce (overwrite)")) {
        store.reduce_cases({record->info}, app.settings(), true);
      }
      if (ImGui::MenuItem("Copy case directory")) {
        ImGui::SetClipboardText(record->info.case_dir.string().c_str());
      }
      if (ImGui::MenuItem("Copy guiding_metrics.csv path")) {
        ImGui::SetClipboardText(record->metrics_csv.string().c_str());
      }
      ImGui::EndPopup();
    }
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayNormal)) {
      ImGui::SetTooltip("%s\ndiag: %s\ncsv: %s%s%s", record->info.case_dir.string().c_str(),
                        record->info.diag_dir.string().c_str(), record->metrics_csv.string().c_str(),
                        record->singlecase_reason.empty() ? "" : "\nscore: ", record->singlecase_reason.c_str());
    }

    ImGui::TableSetColumnIndex(kColumnType);
    ImGui::TextColored(widgets::case_type_color(record->info.case_type), "%s",
                       campaign::case_type_name(record->info.case_type));
    ImGui::TableSetColumnIndex(kColumnH5);
    ImGui::Text("%lld", static_cast<long long>(record->info.h5_count));
    ImGui::TableSetColumnIndex(kColumnAge);
    if (record->age_min) {
      ImGui::Text("%.1f", *record->age_min);
    } else {
      ImGui::TextDisabled("-");
    }
    ImGui::TableSetColumnIndex(kColumnState);
    case_state_chips(*record);
    ImGui::TableSetColumnIndex(kColumnScore);
    if (record->singlecase_score) {
      ImGui::Text("%.1f", *record->singlecase_score);
    } else if (!record->singlecase_status.empty()) {
      ImGui::TextDisabled("%s", record->singlecase_status.c_str());
    } else {
      ImGui::TextDisabled("-");
    }
  }
  ImGui::EndTable();

  std::vector<campaign::CaseInfo> marked;
  for (const auto& record : snapshot.cases) {
    if (ui.marked_cases.contains(record.info.case_id)) {
      marked.push_back(record.info);
    }
  }
  ImGui::BeginDisabled(marked.empty());
  if (ImGui::Button(fmt::format("Reduce marked ({})", marked.size()).c_str())) {
    store.reduce_cases(marked, app.settings(), false);
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  std::vector<campaign::CaseInfo> missing;
  for (const auto& record : snapshot.cases) {
    if (!record.reduced_ready && record.raw_ready()) {
      missing.push_back(record.info);
    }
  }
  ImGui::BeginDisabled(missing.empty());
  if (ImGui::Button(fmt::format("Reduce all ready ({})", missing.size()).c_str())) {
    store.reduce_cases(missing, app.settings(), false);
  }
  ImGui::EndDisabled();
  ImGui::SameLine();
  ImGui::TextDisabled("Ctrl+click marks cases");
}

void member_chip(const char* letter, const std::optional<campaign::CaseInfo>& member, const CampaignSnapshot& snapshot) {
  if (!member) {
    widgets::chip(letter, widgets::kMuted);
    return;
  }
  const CaseRecord* record = snapshot.find_case(member->case_id);
  const ImVec4 color = record == nullptr        ? widgets::kMuted
                       : record->reduced_ready  ? widgets::kGood
                       : record->raw_ready()    ? widgets::kWarn
                                                : widgets::kBad;
  widgets::chip(letter, color);
  if (ImGui::IsItemHovered()) {
    ImGui::SetTooltip("%s", member->case_id.c_str());
  }
}

void draw_triplets_table(UiState& ui, const CampaignSnapshot& snapshot) {
  const ImGuiTableFlags flags = ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY |
                                ImGuiTableFlags_Resizable | ImGuiTableFlags_SizingStretchProp;
  if (!ImGui::BeginTable("##triplets", 3, flags)) {
    return;
  }
  ImGui::TableSetupScrollFreeze(0, 1);
  ImGui::TableSetupColumn("Triplet", ImGuiTableColumnFlags_WidthStretch, 4.0f);
  ImGui::TableSetupColumn("C U V", ImGuiTableColumnFlags_WidthFixed);
  ImGui::TableSetupColumn("Channel score", ImGuiTableColumnFlags_WidthFixed);
  ImGui::TableHeadersRow();
  for (const auto& triplet : snapshot.triplets) {
    const std::string label = triplet.label();
    if (!contains_case_insensitive(label, ui.case_filter)) {
      continue;
    }
    ImGui::TableNextRow();
    ImGui::TableSetColumnIndex(0);
    if (ImGui::Selectable(label.c_str(), ui.selected_triplet == label, ImGuiSelectableFlags_SpanAllColumns)) {
      ui.selected_triplet = label;
      ui.focus_request = kTripletWindow;
    }
    ImGui::TableSetColumnIndex(1);
    member_chip("C", triplet.channel, snapshot);
    ImGui::SameLine();
    member_chip("U", triplet.uniform, snapshot);
    ImGui::SameLine();
    member_chip("V", triplet.vacuum, snapshot);
    ImGui::TableSetColumnIndex(2);
    const CaseRecord* channel = triplet.channel ? snapshot.find_case(triplet.channel->case_id) : nullptr;
    if (channel != nullptr && channel->singlecase_score) {
      ImGui::Text("%.1f", *channel->singlecase_score);
    } else {
      ImGui::TextDisabled("-");
    }
  }
  ImGui::EndTable();
}

}  // namespace

void draw_campaign_panel(App& app, UiState& ui, DataStore& store) {
  if (!ImGui::Begin(kCampaignWindow)) {
    ImGui::End();
    return;
  }

  const float label_width = ImGui::CalcTextSize("Case metrics").x + ImGui::GetStyle().ItemSpacing.x;
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Campaign");
  ImGui::SameLine(label_width);
  ImGui::SetNextItemWidth(-ImGui::CalcTextSize("Browse  Open").x - 4 * ImGui::GetStyle().FramePadding.x);
  const bool enter = input_string("##root", ui.root_input, ImGuiInputTextFlags_EnterReturnsTrue);
  ImGui::SameLine();
  if (ImGui::Button("Browse")) {
    ui.open_root_picker = true;
  }
  ImGui::SameLine();
  if (ImGui::Button("Open") || enter) {
    ui.metrics_root_input.clear();
    app.open_campaign();
  }
  ImGui::AlignTextToFramePadding();
  ImGui::TextUnformatted("Case metrics");
  ImGui::SameLine(label_width);
  ImGui::SetNextItemWidth(-1.0f);
  if (input_string("##metrics_root", ui.metrics_root_input, ImGuiInputTextFlags_EnterReturnsTrue)) {
    app.open_campaign();
  }

  if (ImGui::CollapsingHeader("Readiness and refresh")) {
    ImGui::SetNextItemWidth(120.0f);
    int min_h5 = static_cast<int>(ui.min_h5);
    if (ImGui::InputInt("min h5", &min_h5)) {
      ui.min_h5 = std::max(0, min_h5);
      app.rescan();
    }
    ImGui::SetNextItemWidth(120.0f);
    if (ImGui::InputDouble("min newest-file age [min]", &ui.min_last_h5_age_min, 1.0, 10.0, "%.1f")) {
      app.rescan();
    }
    ImGui::SetNextItemWidth(120.0f);
    if (ImGui::InputDouble("late fraction", &ui.late_fraction, 0.05, 0.1, "%.3f")) {
      ui.late_fraction = std::clamp(ui.late_fraction, 0.01, 1.0);
      app.rescan();
    }
    ImGui::Checkbox("Auto refresh every", &ui.auto_refresh);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(90.0f);
    ImGui::InputDouble("s", &ui.poll_seconds, 1.0, 10.0, "%.0f");
    ui.poll_seconds = std::max(1.0, ui.poll_seconds);
  }

  const auto snapshot = store.campaign();
  if (ImGui::Button("Rescan (F5)")) {
    app.rescan();
  }
  ImGui::SameLine();
  if (store.scanning()) {
    ImGui::TextColored(widgets::kWarn, "scanning...");
  } else if (snapshot != nullptr && !snapshot->error.empty()) {
    ImGui::TextColored(widgets::kBad, "%s", snapshot->error.c_str());
  } else if (snapshot != nullptr) {
    std::size_t reduced = 0;
    std::size_t usable = 0;
    for (const auto& record : snapshot->cases) {
      reduced += record.reduced_ready ? 1 : 0;
      usable += record.usable() ? 1 : 0;
    }
    ImGui::TextDisabled("%zu cases (%zu reduced, %zu usable), scan %.2f s", snapshot->cases.size(), reduced, usable,
                        snapshot->scan_seconds);
  } else {
    ImGui::TextDisabled("open a campaign directory");
  }

  ImGui::SetNextItemWidth(ImGui::GetContentRegionAvail().x * 0.45f);
  input_string("##filter", ui.case_filter);
  if (ui.case_filter.empty() && !ImGui::IsItemActive()) {
    const ImVec2 min = ImGui::GetItemRectMin();
    ImGui::GetWindowDrawList()->AddText(ImVec2(min.x + ImGui::GetStyle().FramePadding.x, min.y + ImGui::GetStyle().FramePadding.y),
                                        ImGui::GetColorU32(ImGuiCol_TextDisabled), "filter");
  }
  ImGui::SameLine();
  ImGui::PushStyleColor(ImGuiCol_CheckMark, widgets::kChannelColor);
  ImGui::Checkbox("C", &ui.show_channel);
  ImGui::PopStyleColor();
  ImGui::SameLine();
  ImGui::PushStyleColor(ImGuiCol_CheckMark, widgets::kUniformColor);
  ImGui::Checkbox("U", &ui.show_uniform);
  ImGui::PopStyleColor();
  ImGui::SameLine();
  ImGui::PushStyleColor(ImGuiCol_CheckMark, widgets::kVacuumColor);
  ImGui::Checkbox("V", &ui.show_vacuum);
  ImGui::PopStyleColor();
  ImGui::SameLine();
  ImGui::Checkbox("usable", &ui.usable_only);

  if (snapshot != nullptr && ImGui::BeginTabBar("##campaign_tabs")) {
    if (ImGui::BeginTabItem(fmt::format("Cases ({})###cases", snapshot->cases.size()).c_str())) {
      draw_cases_table(app, ui, store, *snapshot);
      ImGui::EndTabItem();
    }
    if (ImGui::BeginTabItem(fmt::format("Triplets ({})###triplets", snapshot->triplets.size()).c_str())) {
      draw_triplets_table(ui, *snapshot);
      ImGui::EndTabItem();
    }
    ImGui::EndTabBar();
  }
  ImGui::End();
}

}  // namespace guiding::gui

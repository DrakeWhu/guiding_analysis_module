#include <ctime>
#include <string>

#include <fmt/format.h>

#include "panels.hpp"

namespace guiding::gui {

void draw_log_panel(UiState& ui, DataStore& store) {
  if (!ImGui::Begin(kLogWindow)) {
    ImGui::End();
    return;
  }
  const auto jobs = store.jobs();
  for (const auto& job : jobs) {
    const char spinner = "|/-\\"[static_cast<int>(ImGui::GetTime() * 8.0) % 4];
    ImGui::TextColored(job.running ? widgets::kWarn : widgets::kMuted, "%c %s%s", job.running ? spinner : ' ',
                       job.name.c_str(), job.running ? "" : " (queued)");
  }
  if (store.scanning()) {
    ImGui::TextColored(widgets::kMuted, "scanning campaign...");
  }

  ImGui::Checkbox("info", &ui.log_show_info);
  ImGui::SameLine();
  ImGui::Checkbox("autoscroll", &ui.log_autoscroll);
  ImGui::SameLine();
  if (ImGui::Button("Clear")) {
    store.log().clear();
  }
  ImGui::SameLine();
  const bool copy = ImGui::Button("Copy");

  const auto entries = store.log().entries();
  std::string copied;
  ImGui::BeginChild("##log_lines", ImVec2(0, 0), ImGuiChildFlags_Borders, ImGuiWindowFlags_HorizontalScrollbar);
  for (const auto& entry : entries) {
    if (entry.level == LogLevel::Info && !ui.log_show_info) {
      continue;
    }
    const auto seconds = static_cast<std::time_t>(entry.time_s);
    std::tm local{};
    localtime_r(&seconds, &local);
    const std::string line = fmt::format("{:02}:{:02}:{:02}  {}", local.tm_hour, local.tm_min, local.tm_sec, entry.text);
    const ImVec4 color = entry.level == LogLevel::Error     ? widgets::kBad
                         : entry.level == LogLevel::Warning ? widgets::kWarn
                                                            : ImGui::GetStyleColorVec4(ImGuiCol_Text);
    ImGui::TextColored(color, "%s", line.c_str());
    if (copy) {
      copied += line + "\n";
    }
  }
  if (ui.log_autoscroll && ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 4.0f) {
    ImGui::SetScrollHereY(1.0f);
  }
  ImGui::EndChild();
  if (copy) {
    ImGui::SetClipboardText(copied.c_str());
  }
  ImGui::End();
}

}  // namespace guiding::gui

#include <cstdio>
#include <exception>
#include <string>

#include <CLI/CLI.hpp>
#include <fmt/format.h>

#include "app.hpp"

int main(int argc, char** argv) {
  guiding::gui::AppOptions options;
  CLI::App cli{"guiding_gui: dashboard for WarpX RZ capillary guiding campaigns"};
  cli.set_version_flag("--version", "guiding_gui 0.1.0");
  cli.add_option("--campaign-root", options.campaign_root, "Campaign directory to open at start-up");
  cli.add_option("--case-metrics-root", options.case_metrics_root,
                 "Directory holding CASE_ID/guiding_metrics.csv (default: analysis_outputs/campaign/case_metrics "
                 "next to the campaign)");
  cli.add_option("--min-h5", options.min_h5, "Readiness: minimum number of *.h5 files")->capture_default_str();
  cli.add_option("--min-last-h5-age-min", options.min_last_h5_age_min, "Readiness: minimum age of the newest *.h5")
      ->capture_default_str();
  cli.add_option("--late-fraction", options.late_fraction, "Fraction of dumps in the triplet late window")
      ->capture_default_str();
  cli.add_option("--poll-seconds", options.poll_seconds, "Campaign rescan interval")->capture_default_str();
  cli.add_flag("!--no-auto-refresh", options.auto_refresh, "Do not rescan periodically");
  cli.add_flag("!--no-vsync", options.vsync, "Render without waiting for the display refresh");
  cli.add_option("--width", options.width, "Initial window width (logical pixels)")->capture_default_str();
  cli.add_option("--height", options.height, "Initial window height (logical pixels)")->capture_default_str();
  cli.add_option("--self-test", options.self_test_frames,
                 "Render N frames cycling through cases/triplets, print frame-time statistics and exit");
  cli.add_option("--screenshot", options.screenshot, "With --self-test: save the last frame as PNG");
  cli.add_option("--select-case", options.select_case, "Case to select at start-up");
  cli.add_option("--select-triplet", options.select_triplet, "Triplet label to select at start-up");
  cli.add_option("--focus", options.focus_window, "Window to bring to front: Campaign, Case, Triplet, Overview, Log");
  CLI11_PARSE(cli, argc, argv);

  try {
    guiding::gui::App app(options);
    return app.run();
  } catch (const std::exception& error) {
    fmt::print(stderr, "[FAIL] {}\n", error.what());
    return 1;
  }
}

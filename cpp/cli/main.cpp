#include <cstdio>
#include <exception>
#include <vector>

#include <CLI/CLI.hpp>
#include <fmt/format.h>

#include "commands.hpp"

int main(int argc, char** argv) {
  CLI::App app{"guiding_cli: reduction pipeline for WarpX RZ capillary guiding simulations"};
  app.set_version_flag("--version", "guiding_cli 0.1.0");
  app.require_subcommand(1);

  const std::vector<guiding::cli::Command> commands{
      guiding::cli::add_case_command(app),
      guiding::cli::add_campaign_command(app),
      guiding::cli::add_triplet_command(app),
      guiding::cli::add_inspect_command(app),
  };

  CLI11_PARSE(app, argc, argv);

  for (const auto& [subcommand, run] : commands) {
    if (!subcommand->parsed()) {
      continue;
    }
    try {
      return run();
    } catch (const std::exception& error) {
      std::fflush(stdout);
      fmt::print(stderr, "[FAIL] {}\n", error.what());
      return 1;
    }
  }
  return 0;
}

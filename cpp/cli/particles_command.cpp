#include <algorithm>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "commands.hpp"
#include "guiding/exec/parallel.hpp"
#include "guiding/products/particle_campaign.hpp"
#include "guiding/products/particle_reduction.hpp"
#include "guiding/table/py_format.hpp"

namespace guiding::cli {
namespace {

namespace fs = std::filesystem;
using products::ParticleCaseOptions;

// Options shared by `particles` and `particles-campaign` (flag names of the
// Python scripts; plot options are accepted and ignored).
struct ParticleArgs {
  std::string species = "electrons";
  std::string which = "last";
  int stride = 1;
  double hot_energy_mev = 10.0;
  std::string longitudinal = "z";
  bool no_forward_cut = false;
  std::optional<double> exit_window_mm;
  std::string exit_kind = "plateau";
  std::optional<double> target_propagation_mm;
  std::optional<std::int64_t> maximum_target_iteration_delta;
  std::optional<double> downramp_mm;
  std::string acceptance_theta_cuts_mrad = "2,5,10,20,50";
  std::string acceptance_energy_cuts_mev = "10,25,50,100,150,200,250,300";
  double soft50_energy_low_mev = 10.0;
  double soft50_energy_target_mev = 50.0;
  double soft50_reliability_floor = 0.05;
  double soft50_effective_count_reference = 200.0;
  std::string soft50_curve_energy_low_mev = "5,10";
  bool skip_existing = false;
  bool overwrite = false;
  // plots (no-ops)
  int bins = 200;
  std::optional<double> emax_mev;
  double spectrum_emin_mev = 0.0;
  bool spectrum_log_y = false;
  int max_phase_points = 200000;

  unsigned threads = 0;
  bool no_raw_reads = false;
};

unsigned bounded_default_threads() { return std::min(4U, exec::default_thread_count()); }

void add_particle_options(CLI::App& command, ParticleArgs& args) {
  command.add_option("--species", args.species,
                     "Comma- or whitespace-separated electron species. With multiple species the first summary row "
                     "is their concatenated all_electrons scope.")
      ->capture_default_str();
  command.add_option("--which", args.which)->check(CLI::IsMember({"last", "all", "exit"}))->capture_default_str();
  command.add_option("--stride", args.stride)->capture_default_str();
  command.add_option("--hot-energy-mev", args.hot_energy_mev)->capture_default_str();
  command
      .add_option("--longitudinal", args.longitudinal,
                  "Longitudinal coordinate/momentum component for phase space and forward cut.")
      ->check(CLI::IsMember({"x", "y", "z"}))
      ->capture_default_str();
  command.add_flag("--no-forward-cut", args.no_forward_cut,
                   "Do not require positive longitudinal momentum for hot-electron selection.");
  command.add_option("--exit-window-mm", args.exit_window_mm,
                     "Optional particle-space window behind max longitudinal coordinate in selected dump.");
  command.add_option("--exit-kind", args.exit_kind, "Physical target used by --which exit.")
      ->check(CLI::IsMember({"plateau", "capillary"}))
      ->capture_default_str();
  command.add_option("--target-propagation-mm", args.target_propagation_mm,
                     "Override physical target propagation in mm for legacy --which exit selection.");
  command.add_option("--maximum-target-iteration-delta", args.maximum_target_iteration_delta,
                     "For legacy --which exit selection, abort unless the selected particle iteration is within this "
                     "many steps of the guiding iteration nearest the physical exit.");
  command.add_option("--downramp-mm", args.downramp_mm,
                     "Downramp length in mm used by legacy --which exit --exit-kind capillary when no explicit total "
                     "capillary length is available.");
  command.add_option("--acceptance-theta-cuts-mrad", args.acceptance_theta_cuts_mrad)->capture_default_str();
  command.add_option("--acceptance-energy-cuts-mev", args.acceptance_energy_cuts_mev)->capture_default_str();
  command.add_option("--soft50-energy-low-mev", args.soft50_energy_low_mev)->capture_default_str();
  command.add_option("--soft50-energy-target-mev", args.soft50_energy_target_mev)->capture_default_str();
  command.add_option("--soft50-reliability-floor", args.soft50_reliability_floor)->capture_default_str();
  command.add_option("--soft50-effective-count-reference", args.soft50_effective_count_reference)
      ->capture_default_str();
  command.add_option("--soft50-curve-energy-low-mev", args.soft50_curve_energy_low_mev)->capture_default_str();
  command.add_flag("--skip-existing", args.skip_existing);
  command.add_flag("--overwrite", args.overwrite);
  command.add_option("--bins", args.bins, "Accepted for compatibility (plots are not rendered).");
  command.add_option("--emax-mev", args.emax_mev, "Accepted for compatibility (plots are not rendered).");
  command.add_option("--spectrum-emin-mev", args.spectrum_emin_mev,
                     "Accepted for compatibility (plots are not rendered).");
  command.add_flag("--spectrum-log-y", args.spectrum_log_y, "Accepted for compatibility (plots are not rendered).");
  command.add_option("--max-phase-points", args.max_phase_points,
                     "Accepted for compatibility (plots are not rendered).");
  command.add_flag("--no-raw-reads", args.no_raw_reads,
                   "Read every dataset through HDF5 (disables the pread fast path)");
}

ParticleCaseOptions case_options(const ParticleArgs& args) {
  ParticleCaseOptions options;
  options.species = products::parse_species_list(args.species);
  options.which = products::parse_particle_which(args.which);
  options.stride = args.stride;
  options.hot_energy_mev = args.hot_energy_mev;
  options.longitudinal = physics::parse_longitudinal(args.longitudinal);
  options.forward_only = !args.no_forward_cut;
  options.exit_window_mm = args.exit_window_mm;
  options.exit_kind = products::parse_exit_kind(args.exit_kind);
  options.target_propagation_mm = args.target_propagation_mm;
  options.maximum_target_iteration_delta = args.maximum_target_iteration_delta;
  options.downramp_mm = args.downramp_mm;
  options.acceptance_theta_cuts_mrad = products::parse_float_list(args.acceptance_theta_cuts_mrad);
  options.acceptance_energy_cuts_mev = products::parse_float_list(args.acceptance_energy_cuts_mev);
  options.soft50 = {args.soft50_energy_low_mev, args.soft50_energy_target_mev, args.soft50_reliability_floor,
                    args.soft50_effective_count_reference};
  options.soft50_curve_energy_low_mev = products::parse_float_list(args.soft50_curve_energy_low_mev);
  options.spectrum_emin_mev = args.spectrum_emin_mev;
  options.spectrum_log_y = args.spectrum_log_y;
  options.skip_existing = args.skip_existing;
  options.overwrite = args.overwrite;
  options.raw_reads = !args.no_raw_reads;
  return options;
}

// ---- particles ---------------------------------------------------------------

struct ParticleCaseArgs {
  std::string diag;
  std::string outdir;
  std::optional<std::string> guiding_metrics;
  std::optional<std::string> resolved_parameters;
  bool plots_only = false;
  ParticleArgs common;
};

int run_particles(const ParticleCaseArgs& args) {
  auto options = case_options(args.common);
  options.diag = args.diag;
  options.outdir = args.outdir;
  if (args.guiding_metrics) {
    options.guiding_metrics = fs::path(*args.guiding_metrics);
  }
  if (args.resolved_parameters) {
    options.resolved_parameters = fs::path(*args.resolved_parameters);
  }
  options.plots_only = args.plots_only;
  options.threads = args.common.threads == 0 ? bounded_default_threads() : args.common.threads;
  products::run_particle_case(options, print_line);
  return 0;
}

// ---- particles-campaign ------------------------------------------------------

struct ParticleCampaignArgs {
  std::string campaign_root;
  std::string particle_diag_name = "auto";
  std::string case_glob = "0*_from_*";
  std::string outdir_name = "particle_analysis";
  bool use_resolved_parameters = false;
  ParticleArgs common;
};

int run_particles_campaign(const ParticleCampaignArgs& args) {
  products::ParticleCampaignOptions options;
  options.campaign_root = args.campaign_root;
  options.particle_diag_name = args.particle_diag_name;
  options.species_text = args.common.species;
  options.case_glob = args.case_glob;
  options.outdir_name = args.outdir_name;
  options.use_resolved_parameters = args.use_resolved_parameters;
  options.case_options = case_options(args.common);
  options.threads = args.common.threads == 0 ? bounded_default_threads() : args.common.threads;
  const auto result = products::run_particle_campaign(options, print_line);
  return result.failed > 0 ? 1 : 0;
}

}  // namespace

Command add_particles_command(CLI::App& app) {
  auto args = std::make_shared<ParticleCaseArgs>();
  auto* command = app.add_subcommand(
      "particles", "Analyze WarpX electron particle openPMD diagnostics (scripts/analyze_particle_case.py)");
  command->add_option("--diag", args->diag, "Path to particle openPMD diagnostic directory, e.g. CASE/diags/plasma_electrons")
      ->required();
  command->add_option("--outdir", args->outdir, "Output directory, usually CASE/particle_analysis")->required();
  command->add_option("--guiding-metrics", args->guiding_metrics,
                      "Optional guiding_metrics.csv. Defaults to CASE_DIR/guiding_metrics.csv.");
  command->add_option("--resolved-parameters", args->resolved_parameters,
                      "Use particle_diagnostic_targets in this resolved_parameters.json as the authoritative exact "
                      "exit iteration. Only valid with --which exit.");
  command->add_flag("--plots-only", args->plots_only,
                    "Python regenerates plots only; guiding_cli renders no plots and writes nothing.");
  add_particle_options(*command, args->common);
  command->add_option("--threads", args->common.threads,
                      "Iterations analysed concurrently with --which all (default: min(4, available cores))");
  return {command, [args] { return run_particles(*args); }};
}

Command add_particles_campaign_command(CLI::App& app) {
  auto args = std::make_shared<ParticleCampaignArgs>();
  auto* command = app.add_subcommand(
      "particles-campaign",
      "Run particle analysis over a campaign in one process (scripts/analyze_particle_campaign.py)");
  command->add_option("--campaign-root", args->campaign_root)->required();
  command->add_option("--particle-diag-name", args->particle_diag_name,
                      "Diagnostic directory name under CASE/diags, or 'auto'.")
      ->capture_default_str();
  command->add_option("--case-glob", args->case_glob, "Glob for case directories under campaign root.")
      ->capture_default_str();
  command->add_option("--outdir-name", args->outdir_name, "Output directory name inside each case directory.")
      ->capture_default_str();
  command->add_flag("--use-resolved-parameters", args->use_resolved_parameters,
                    "Exact exit selection from CASE/resolved_parameters.json (--which exit only).");
  add_particle_options(*command, args->common);
  command->add_option("--threads", args->common.threads,
                      "Cases analysed concurrently (default: min(4, available cores))");
  return {command, [args] { return run_particles_campaign(*args); }};
}

}  // namespace guiding::cli

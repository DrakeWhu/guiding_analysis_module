#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <numeric>

#include "guiding/io/openpmd_series.hpp"
#include "guiding/io/particle_reader.hpp"
#include "guiding/physics/field_metrics.hpp"
#include "guiding/products/case_reduction.hpp"
#include "guiding/products/field_map.hpp"
#include "guiding/products/particle_view.hpp"
#include "guiding/table/csv.hpp"

namespace {

namespace fs = std::filesystem;
using namespace guiding;

fs::path campaign_dir() { return fs::path(GUIDING_TEST_DATA_DIR) / "synthetic" / "campaign"; }
const char* kChannel = "000_f20_chan_n4e18cm3_L2mm_d150um_foc0um_rz";

}  // namespace

TEST_CASE("max_pool keeps block maxima and ignores NaN", "[view]") {
  const double nan = std::nan("");
  const std::vector<double> grid{1, 2, 3, 4, 5,  //
                                 6, nan, 8, 9, 10,  //
                                 nan, nan, 0, 1, 2};
  const auto pooled = products::max_pool(grid, 3, 5, 2, 3);
  REQUIRE(pooled.rows == 2);
  REQUIRE(pooled.cols == 3);
  CHECK(pooled.values[0] == 6.0);
  CHECK(pooled.values[1] == 9.0);
  CHECK(pooled.values[2] == 10.0);
  CHECK(std::isnan(pooled.values[3]));
  CHECK(pooled.values[4] == 1.0);
  CHECK(pooled.values[5] == 2.0);
  CHECK(products::max_pool(std::vector<double>{}, 0, 0, 4, 4).values.empty());
}

TEST_CASE("field map agrees with the guiding reduction of the same dump", "[view]") {
  const auto series = io::FileSeries::scan(campaign_dir() / kChannel / "diags" / "diag1");
  const std::int64_t iteration = series.iterations().back();
  products::FieldMapOptions options;
  options.max_z_cells = 64;
  options.max_r_cells = 8;
  const auto map = products::load_field_map(series.file(iteration), iteration, options);
  const auto row = products::reduce_field_iteration(series.file(iteration), iteration, options.params);

  CHECK(map.laser.z_peak_um == row.laser.z_peak_um);
  CHECK(map.laser.waist_um == row.laser.waist_um);
  CHECK(map.laser.a0_peak == row.laser.a0_peak);
  CHECK(map.wake.Ez_wake_absmax == row.wake.Ez_wake_absmax);
  CHECK(map.intensity_max == row.laser.peak_I_proxy);  // pooling keeps the global maximum
  CHECK(map.cols <= 64);
  CHECK(map.rows <= 8);
  CHECK(map.intensity.size() == map.rows * map.cols);

  const auto peak = std::max_element(map.I_z_smooth.begin(), map.I_z_smooth.end());
  CHECK(map.z_um[static_cast<std::size_t>(peak - map.I_z_smooth.begin())] == map.laser.z_peak_um);
  CHECK(std::is_sorted(map.r_um.begin(), map.r_um.end()));
  CHECK(map.Ez_axis_GVm.size() == map.z_um.size());
  CHECK(map.r_max_um > map.r_min_um);
}

TEST_CASE("particle view matches the summary and acceptance products", "[view]") {
  const auto series = io::FileSeries::scan(campaign_dir() / kChannel / "diags" / "plasma_electrons");
  const auto info = io::read_particle_series_info(series);
  const std::int64_t iteration = series.iterations().back();
  const auto dump = io::read_particle_dump(series, info, "electrons", iteration);
  const auto view = products::build_particle_view(dump, "electrons");

  const auto summary_charge = std::get<double>(*view.summary.get("charge_hot_pC"));
  CHECK(view.n_total == dump.size());
  CHECK(static_cast<std::int64_t>(view.n_hot) == std::get<std::int64_t>(*view.summary.get("n_macroparticles_hot")));

  // Accepted charges equal the golden particle_acceptance_curves.csv (chan_last run).
  const auto golden = table::read_csv_file(fs::path(GUIDING_TEST_DATA_DIR) / "golden" / "particles" / "chan_last" /
                                           "particle_acceptance_curves.csv");
  // repr text parsed with correct rounding (numeric_column mimics pandas, which is not exact).
  const auto expected = golden.string_column("accepted_charge_pC");
  REQUIRE(view.accepted_pC.size() == expected.size());
  for (std::size_t i = 0; i < expected.size(); ++i) {
    CHECK(view.accepted_pC[i] == table::parse_number(expected[i]).value());
  }
  CHECK(view.theta_cuts_mrad.size() * view.energy_cuts_mev.size() == view.accepted_pC.size());

  // Spectra and histograms conserve the charge they bin.
  const double bin_width = view.energy_edges_mev[1] - view.energy_edges_mev[0];
  const double hot_from_spectrum =
      std::accumulate(view.spectrum_hot.begin(), view.spectrum_hot.end(), 0.0) * bin_width;
  CHECK(hot_from_spectrum == Catch::Approx(summary_charge).epsilon(1e-12));
  REQUIRE(view.phase_spaces.size() == 5);
  for (const auto& histogram : view.phase_spaces) {
    const double total = std::accumulate(histogram.charge_pC.begin(), histogram.charge_pC.end(), 0.0);
    CHECK(total == Catch::Approx(summary_charge).epsilon(1e-12));
    CHECK(histogram.x_max > histogram.x_min);
  }
}

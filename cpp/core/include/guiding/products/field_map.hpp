#pragma once

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <span>
#include <vector>

#include "guiding/physics/field_metrics.hpp"

// Data behind the dashboard's field view: the transverse intensity map of one
// dump and the lineouts that the guiding reduction is built from.
namespace guiding::products {

struct FieldMapOptions {
  physics::FieldParams params;
  std::size_t max_z_cells = 1200;  // image size after max-pooling
  std::size_t max_r_cells = 400;
  bool raw_reads = true;
};

struct FieldMap {
  std::int64_t iteration = 0;
  double time_fs = 0.0;
  physics::LaserMetrics laser{};
  physics::WakeMetrics wake{};

  // |E_perp|^2 = Ex^2 + Ey^2 at theta = 0 on the positive-r grid, max-pooled
  // to rows x cols; row 0 is the largest r (top of an image), columns go along z.
  std::size_t rows = 0;
  std::size_t cols = 0;
  std::vector<double> intensity;
  double intensity_max = 0.0;
  double z_min_um = 0.0;
  double z_max_um = 0.0;
  double r_min_um = 0.0;
  double r_max_um = 0.0;

  // Full-resolution lineouts as used by metrics.weighted_laser_metrics.
  std::vector<double> z_um;
  std::vector<double> I_z;          // sum_r I * max(r, 0)
  std::vector<double> I_z_smooth;   // moving average over smooth_um
  std::vector<double> r_um;
  std::vector<double> P_r;          // sum of I over |z - z_peak| <= 5 um
  std::vector<double> Ez_axis_GVm;  // E_z on the row nearest the axis
};

[[nodiscard]] FieldMap load_field_map(const std::filesystem::path& file, std::int64_t iteration,
                                      const FieldMapOptions& options = {});

struct PooledGrid {
  std::vector<double> values;  // row-major
  std::size_t rows = 0;
  std::size_t cols = 0;
};

// Max-pools a row-major rows x cols grid onto at most out_rows x out_cols
// cells with equal integer blocks (NaN ignored; all-NaN blocks stay NaN).
[[nodiscard]] PooledGrid max_pool(std::span<const double> values, std::size_t rows, std::size_t cols,
                                  std::size_t out_rows, std::size_t out_cols);

}  // namespace guiding::products

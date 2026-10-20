#include "guiding/physics/transverse.hpp"

#include <cmath>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include <fmt/format.h>

#include "guiding/numeric/npcompat.hpp"
#include "guiding/physics/particles.hpp"

namespace guiding::physics {
namespace {

constexpr double kNaN = std::numeric_limits<double>::quiet_NaN();
constexpr double kMradPerRad = 1.0e3;
constexpr double kUmPerM = 1.0e6;
constexpr double kMmMradPerMRad = 1.0e6;

const std::vector<std::string>& transverse_output_columns() {
  static const std::vector<std::string> columns{
      "transverse_status",       "n_macroparticles_transverse",    "weight_transverse",
      "theta_x_rms_mrad",        "theta_y_rms_mrad",               "theta_rms_mrad",
      "theta_x_p95_mrad",        "theta_y_p95_mrad",               "theta_r_p95_mrad",
      "x_rms_um",                "y_rms_um",                       "x_p95_um",
      "y_p95_um",                "emit_x_norm_mm_mrad",            "emit_y_norm_mm_mrad",
      "emit_geom_norm_mm_mrad",  "transverse_theta_rms_component", "transverse_theta_p95_component",
      "transverse_emit_component", "beam_transverse_quality_score"};
  return columns;
}

table::Record empty_transverse_metrics(const std::string& status) {
  table::Record out;
  out.set("transverse_status", status);
  out.set("n_macroparticles_transverse", std::int64_t{0});
  out.set("weight_transverse", 0.0);
  out.set("beam_transverse_quality_score", 0.0);
  out.set("transverse_theta_rms_component", 0.0);
  out.set("transverse_theta_p95_component", 0.0);
  out.set("transverse_emit_component", 0.0);
  for (const auto& column : transverse_output_columns()) {
    if (!out.get(column)) {
      out.set(column, kNaN);
    }
  }
  return out;
}

double weighted_rms(std::span<const double> values, std::span<const double> weights) {
  std::vector<double> squared(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    squared[i] = values[i] * values[i];
  }
  return std::sqrt(weighted_average(squared, weights));
}

double weighted_centered_rms(std::span<const double> values, std::span<const double> weights) {
  const double mean = weighted_average(values, weights);
  std::vector<double> centered(values.size());
  for (std::size_t i = 0; i < values.size(); ++i) {
    centered[i] = values[i] - mean;
  }
  return weighted_rms(centered, weights);
}

double normalized_emittance_m_rad(std::span<const double> q, std::span<const double> u,
                                  std::span<const double> weights) {
  const double q_var = weighted_covariance(q, q, weights);
  const double u_var = weighted_covariance(u, u, weights);
  const double qu_cov = weighted_covariance(q, u, weights);
  double determinant = q_var * u_var - qu_cov * qu_cov;
  if (determinant < 0.0 && std::abs(determinant) <= 1.0e-24) {
    determinant = 0.0;
  }
  return std::sqrt((0.0 > determinant) ? 0.0 : determinant);
}

double inverse_scale_component(double value, double reference) {
  if (!std::isfinite(value) || reference <= 0.0) {
    return 0.0;
  }
  const double clipped = (0.0 > value) ? 0.0 : value;
  return 1.0 / (1.0 + clipped / reference);
}

// Python max(value, 0.0)
double at_least_zero(double value) { return (0.0 > value) ? 0.0 : value; }

}  // namespace

table::Record beam_transverse_quality_score(double theta_rms_mrad, double theta_r_p95_mrad, double emit_x_norm_mm_mrad,
                                            double emit_y_norm_mm_mrad, const TransverseQualityConfig& config) {
  const double theta_rms_component = inverse_scale_component(theta_rms_mrad, config.theta_rms_ref_mrad);
  const double theta_p95_component = inverse_scale_component(theta_r_p95_mrad, config.theta_p95_ref_mrad);
  double emit_component = 0.0;
  if (std::isfinite(emit_x_norm_mm_mrad) && std::isfinite(emit_y_norm_mm_mrad)) {
    const double emit_geom = std::sqrt(at_least_zero(emit_x_norm_mm_mrad) * at_least_zero(emit_y_norm_mm_mrad));
    emit_component = inverse_scale_component(emit_geom, config.emit_ref_mm_mrad);
  }
  const double score = config.score_scale * theta_rms_component * theta_p95_component * emit_component;
  table::Record out;
  out.set("transverse_theta_rms_component", theta_rms_component);
  out.set("transverse_theta_p95_component", theta_p95_component);
  out.set("transverse_emit_component", emit_component);
  out.set("beam_transverse_quality_score", score);
  return out;
}

table::Record summarize_transverse_metrics(const io::ParticleDump& dump, const Mask& mask, Longitudinal longitudinal,
                                           const TransverseQualityConfig& config) {
  if (longitudinal != Longitudinal::Z) {
    return empty_transverse_metrics(fmt::format("unsupported_longitudinal_axis:{}", longitudinal_name(longitudinal)));
  }
  if (mask.size() != dump.w.size()) {
    throw std::invalid_argument(fmt::format("Transverse mask shape mismatch: mask.shape=({},), w.shape=({},)",
                                            mask.size(), dump.w.size()));
  }
  Mask valid(mask.size());
  for (std::size_t i = 0; i < mask.size(); ++i) {
    valid[i] = (mask[i] != 0 && std::isfinite(dump.x_m[i]) && std::isfinite(dump.y_m[i]) && std::isfinite(dump.ux[i]) &&
                std::isfinite(dump.uy[i]) && std::isfinite(dump.uz[i]) && std::isfinite(dump.w[i]) && dump.w[i] > 0.0)
                   ? 1
                   : 0;
  }
  if (!any(valid)) {
    return empty_transverse_metrics("no_selected_particles");
  }

  const auto x = gather(dump.x_m, valid);
  const auto y = gather(dump.y_m, valid);
  const auto ux = gather(dump.ux, valid);
  const auto uy = gather(dump.uy, valid);
  const auto uz = gather(dump.uz, valid);
  const auto w = gather(dump.w, valid);
  const std::size_t n = w.size();

  std::vector<double> theta_x(n), theta_y(n), theta_r(n), abs_theta_x(n), abs_theta_y(n);
  for (std::size_t i = 0; i < n; ++i) {
    theta_x[i] = std::atan2(ux[i], uz[i]);
    theta_y[i] = std::atan2(uy[i], uz[i]);
    theta_r[i] = std::sqrt(theta_x[i] * theta_x[i] + theta_y[i] * theta_y[i]);
    abs_theta_x[i] = std::abs(theta_x[i]);
    abs_theta_y[i] = std::abs(theta_y[i]);
  }

  const double x_mean = weighted_average(x, w);
  const double y_mean = weighted_average(y, w);
  std::vector<double> x_offset(n), y_offset(n);
  for (std::size_t i = 0; i < n; ++i) {
    x_offset[i] = std::abs(x[i] - x_mean);
    y_offset[i] = std::abs(y[i] - y_mean);
  }

  const double emit_x = normalized_emittance_m_rad(x, ux, w) * kMmMradPerMRad;
  const double emit_y = normalized_emittance_m_rad(y, uy, w) * kMmMradPerMRad;

  table::Record out;
  out.set("transverse_status", std::string("ok"));
  out.set("n_macroparticles_transverse", static_cast<std::int64_t>(n));
  out.set("weight_transverse", np::pairwise_sum<double>(w));
  out.set("theta_x_rms_mrad", weighted_rms(theta_x, w) * kMradPerRad);
  out.set("theta_y_rms_mrad", weighted_rms(theta_y, w) * kMradPerRad);
  out.set("theta_rms_mrad", weighted_rms(theta_r, w) * kMradPerRad);
  out.set("theta_x_p95_mrad", weighted_percentile(abs_theta_x, w, 95.0) * kMradPerRad);
  out.set("theta_y_p95_mrad", weighted_percentile(abs_theta_y, w, 95.0) * kMradPerRad);
  out.set("theta_r_p95_mrad", weighted_percentile(theta_r, w, 95.0) * kMradPerRad);
  out.set("x_rms_um", weighted_centered_rms(x, w) * kUmPerM);
  out.set("y_rms_um", weighted_centered_rms(y, w) * kUmPerM);
  out.set("x_p95_um", weighted_percentile(x_offset, w, 95.0) * kUmPerM);
  out.set("y_p95_um", weighted_percentile(y_offset, w, 95.0) * kUmPerM);
  out.set("emit_x_norm_mm_mrad", emit_x);
  out.set("emit_y_norm_mm_mrad", emit_y);
  out.set("emit_geom_norm_mm_mrad", std::sqrt(at_least_zero(emit_x) * at_least_zero(emit_y)));

  const auto value = [&](const char* key) { return std::get<double>(*out.get(key)); };
  out.merge(beam_transverse_quality_score(value("theta_rms_mrad"), value("theta_r_p95_mrad"),
                                          value("emit_x_norm_mm_mrad"), value("emit_y_norm_mm_mrad"), config));
  return out;
}

}  // namespace guiding::physics

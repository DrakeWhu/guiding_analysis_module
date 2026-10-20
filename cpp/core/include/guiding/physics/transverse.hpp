#pragma once

#include "guiding/io/particle_reader.hpp"
#include "guiding/physics/longitudinal.hpp"
#include "guiding/physics/weighted.hpp"
#include "guiding/table/record.hpp"

// Port of cap_guiding/transverse.py.
namespace guiding::physics {

struct TransverseQualityConfig {
  double score_scale = 1000.0;
  double theta_rms_ref_mrad = 5.0;
  double theta_p95_ref_mrad = 15.0;
  double emit_ref_mm_mrad = 2.0;
};

// beam_transverse_quality_score: 1000 * prod 1 / (1 + value / reference)
[[nodiscard]] table::Record beam_transverse_quality_score(double theta_rms_mrad, double theta_r_p95_mrad,
                                                          double emit_x_norm_mm_mrad, double emit_y_norm_mm_mrad,
                                                          const TransverseQualityConfig& config = {});

// Divergence, size and normalised emittance of the selected particles.
[[nodiscard]] table::Record summarize_transverse_metrics(const io::ParticleDump& dump, const Mask& mask,
                                                         Longitudinal longitudinal,
                                                         const TransverseQualityConfig& config = {});

}  // namespace guiding::physics

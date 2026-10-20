#pragma once

#include "guiding/io/thetamode.hpp"

// Port of the per-dump laser and wake metrics in cap_guiding/metrics.py.
namespace guiding::physics {

inline constexpr double kElementaryChargeC = 1.602176634e-19;
inline constexpr double kElectronMassKg = 9.1093837015e-31;
inline constexpr double kSpeedOfLightMPerS = 299792458.0;
inline constexpr double kPi = 3.141592653589793;  // math.pi

struct FieldParams {
  int stride = 1;
  double smooth_um = 2.0;
  double wake_behind_um = 120.0;
  double wake_gap_um = 5.0;
  double lambda0_m = 0.8e-6;
};

struct LaserMetrics {
  double z_peak_um;
  double front_margin_um;
  double back_margin_um;
  double waist_um;
  double peak_I_proxy;
  double Eperp_peak_Vm;
  double a0_peak;
  double energy_proxy;
};

struct WakeMetrics {
  double Ez_wake_max;
  double Ez_wake_min;
  double Ez_wake_absmax;
  double Ez_wake_rms;
  double z_Ez_absmax_um;
  double z_Ez_absmax_rel_um;
};

// a0 = e E0 / (m_e c omega0), omega0 = 2 pi c / lambda0; NaN for invalid input.
[[nodiscard]] double eperp_peak_to_a0(double eperp_peak_vm, double lambda0_m);

// weighted_laser_metrics(Ex_rz, Ey_rz, Ex.r_um, Ex.z_um, smooth_um, lambda0_m)
[[nodiscard]] LaserMetrics weighted_laser_metrics(const io::PositiveRSlice& ex, const io::PositiveRSlice& ey,
                                                  double smooth_um, double lambda0_m);

// wake_metrics(Ez_rz, Ez.r_um, Ez.z_um, z_peak_um, wake_behind_um, wake_gap_um)
[[nodiscard]] WakeMetrics wake_metrics(const io::PositiveRSlice& ez, double z_peak_um, double wake_behind_um,
                                       double wake_gap_um);

}  // namespace guiding::physics

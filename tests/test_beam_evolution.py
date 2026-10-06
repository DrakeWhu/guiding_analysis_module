from __future__ import annotations

from dataclasses import replace
import math
import unittest

import numpy as np

from cap_guiding.beam_evolution import (
    BeamEvolutionConfig, QUALITY_COLUMNS, resolve_frame_coordinate,
    summarize_beam_evolution, summarize_beam_frame, unavailable_beam_frame,
)
from cap_guiding.metrics import E_CHARGE_C
from cap_guiding.particles import ParticleDump


def make_dump(energy=(60., 70., 80., 90.), *, w=None, theta=None, iteration=0):
    e = np.array(energy, dtype=float)
    n = len(e)
    p = np.sqrt((e / .51099895 + 1) ** 2 - 1)
    angles = np.zeros(n) if theta is None else np.array(theta, dtype=float)
    return ParticleDump(
        iteration=iteration, time_fs=float(iteration),
        x_m=np.linspace(-2e-6, 2e-6, n), y_m=np.linspace(1e-6, -1e-6, n),
        z_m=np.zeros(n), ux=p * np.sin(angles), uy=np.zeros(n),
        uz=p * np.cos(angles), w=np.ones(n) if w is None else np.array(w, dtype=float),
    )


def frame(z, *, charge=10., quality=2., iteration=None):
    row = summarize_beam_frame(make_dump(iteration=int(z) if iteration is None else iteration),
                               species_scope="electrons", z_frame_um=z)
    row["charge_pC"] = charge
    for key in QUALITY_COLUMNS:
        row[key] = quality
    return row


class BeamFrameTests(unittest.TestCase):
    def test_physical_weights_energy_moments_and_effective_sample_size(self):
        dump = make_dump(w=[1, 2, 3, 4])
        row = summarize_beam_frame(dump, species_scope="electrons", z_frame_um=20)
        self.assertAlmostEqual(row["charge_pC"], 10 * E_CHARGE_C * 1e12)
        self.assertAlmostEqual(row["n_effective"], 100 / 30)
        self.assertAlmostEqual(row["energy_mean_MeV"], 80)
        self.assertAlmostEqual(row["energy_spread_rms_MeV"], 10)
        self.assertAlmostEqual(row["energy_relative_spread_rms"], .125)
        self.assertEqual(row["quality_status"], "ok")
        self.assertEqual(row["energy_p50_MeV"], 80)

    def test_below_threshold_and_backward_electrons_do_not_contribute(self):
        dump = make_dump([49., 60., 70., 80.], theta=[0, math.pi, 0, 0], w=[10, 20, 3, 4])
        row = summarize_beam_frame(dump, species_scope="electrons", z_frame_um=0)
        self.assertEqual(row["n_macroparticles"], 2)
        self.assertAlmostEqual(row["charge_pC"], 7 * E_CHARGE_C * 1e12)
        self.assertEqual(row["quality_status"], "insufficient_statistics")

    def test_pointing_does_not_become_centered_divergence(self):
        dump = make_dump([80.] * 4, theta=[.1] * 4)
        row = summarize_beam_frame(dump, species_scope="electrons", z_frame_um=0)
        self.assertAlmostEqual(row["theta_x_mean_mrad"], 100)
        self.assertAlmostEqual(row["theta_x_std_mrad"], 0)
        self.assertAlmostEqual(row["theta_x_axis_rms_mrad"], 100)

    def test_weighted_angular_variance_separates_centroid(self):
        angles = np.array([.01, .02, .03, .04])
        weights = np.array([1., 2., 3., 4.])
        row = summarize_beam_frame(make_dump(theta=angles, w=weights),
                                   species_scope="electrons", z_frame_um=0)
        self.assertAlmostEqual(row["theta_x_mean_mrad"], 30)
        self.assertAlmostEqual(row["theta_x_std_mrad"], 10)
        self.assertAlmostEqual(row["theta_x_axis_rms_mrad"] ** 2,
                               row["theta_x_mean_mrad"] ** 2 + row["theta_x_std_mrad"] ** 2)

    def test_linear_force_free_drift_preserves_emittance_and_angles(self):
        # Constant p_z gives a common linear shear x -> x + L*u_x/u_z.
        dump = replace(make_dump(), ux=np.array([-.4, .3, -.2, .5]),
                       uy=np.array([.2, -.3, .4, -.1]), uz=np.full(4, 150.))
        drift = replace(dump, x_m=dump.x_m + .01 * dump.ux / dump.uz,
                        y_m=dump.y_m + .01 * dump.uy / dump.uz)
        before = summarize_beam_frame(dump, species_scope="electrons", z_frame_um=0)
        after = summarize_beam_frame(drift, species_scope="electrons", z_frame_um=10000)
        for key in ("emit_x_norm_mm_mrad", "emit_y_norm_mm_mrad", "theta_x_std_mrad", "theta_y_std_mrad"):
            self.assertAlmostEqual(before[key], after[key], places=10)
        self.assertGreater(after["x_rms_um"], before["x_rms_um"])

    def test_empty_and_unavailable_are_different(self):
        empty = summarize_beam_frame(make_dump([]), species_scope="electrons", z_frame_um=0)
        unknown = unavailable_beam_frame(iteration=0, time_fs=0, species_scope="electrons",
                                         z_frame_um=0, coordinate_source="guiding_exact", error="missing dump")
        self.assertEqual(empty["population_status"], "empty")
        self.assertEqual(empty["charge_pC"], 0)
        self.assertTrue(math.isnan(empty["emit_x_norm_mm_mrad"]))
        self.assertTrue(math.isnan(unknown["charge_pC"]))

    def test_single_particle_does_not_get_good_quality(self):
        row = summarize_beam_frame(make_dump([80.]), species_scope="electrons", z_frame_um=0)
        self.assertEqual(row["quality_status"], "insufficient_statistics")
        self.assertEqual(row["n_effective"], 1)

    def test_invalid_particle_records_are_flagged_not_empty(self):
        dump = replace(make_dump([80.]), x_m=np.array([np.nan]))
        row = summarize_beam_frame(dump, species_scope="electrons", z_frame_um=0)
        self.assertEqual(row["population_status"], "invalid_particle_data")
        self.assertEqual(row["n_invalid_macroparticles"], 1)


class CoordinateTests(unittest.TestCase):
    def setUp(self):
        self.guiding = [dict(iteration=0, time_fs=0, z_max_um=20, z_peak_um=-50),
                        dict(iteration=10, time_fs=100, z_max_um=1020, z_peak_um=900)]

    def test_absolute_axis_does_not_use_laser_peak_or_relative_origin(self):
        self.assertEqual(resolve_frame_coordinate(self.guiding, iteration=0, time_fs=0),
                         (20, "guiding_exact"))

    def test_bracketed_physical_time_interpolation_and_no_extrapolation(self):
        self.assertEqual(resolve_frame_coordinate(self.guiding, iteration=7, time_fs=25),
                         (270, "guiding_time_interpolated"))
        self.assertTrue(math.isnan(resolve_frame_coordinate(self.guiding, iteration=11, time_fs=150)[0]))
        self.assertTrue(math.isnan(resolve_frame_coordinate(self.guiding, iteration=7, time_fs=25,
                                                             max_coordinate_gap_um=500)[0]))

    def test_same_iteration_with_conflicting_timestamp_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "timestamps disagree"):
            resolve_frame_coordinate(self.guiding, iteration=0, time_fs=10)


class EvolutionSummaryTests(unittest.TestCase):
    def test_integrates_full_plasma_with_ramps_and_clips_bracketing_frames(self):
        rows = [frame(z, charge=10 + z / 100, quality=2) for z in (-100, 100, 300)]
        out = summarize_beam_evolution(rows, plasma_start_um=0, plasma_end_um=200, max_gap_um=250)
        self.assertEqual(out["integration_status"], "complete")
        self.assertAlmostEqual(out["charge_distance_pC_um"], 2200)
        self.assertAlmostEqual(out["coverage_fraction"], 1)
        for key in QUALITY_COLUMNS:
            self.assertAlmostEqual(out[f"{key}_charge_weighted_mean"], 2)

    def test_quality_average_uses_charge_not_frame_count(self):
        out = summarize_beam_evolution([frame(0, charge=1, quality=2), frame(100, charge=3, quality=10)],
                                       plasma_start_um=0, plasma_end_um=100)
        self.assertAlmostEqual(out["theta_x_std_mrad_charge_weighted_mean"], 8)

    def test_monoenergetic_frames_with_changing_mean_energy_are_not_pooled(self):
        rows = [summarize_beam_frame(make_dump([e] * 4, iteration=it), species_scope="electrons", z_frame_um=z)
                for e, it, z in ((60., 0, 0), (100., 1, 100))]
        out = summarize_beam_evolution(rows, plasma_start_um=0, plasma_end_um=100)
        self.assertAlmostEqual(out["energy_spread_rms_MeV_charge_weighted_mean"], 0)

    def test_missing_internal_frame_is_not_bridged(self):
        rows = [frame(z) for z in (0, 100, 200, 300, 400)]
        rows[2] = unavailable_beam_frame(iteration=200, time_fs=200, species_scope="electrons",
                                        z_frame_um=200, coordinate_source="guiding_exact", error="read failed")
        out = summarize_beam_evolution(rows, plasma_start_um=0, plasma_end_um=400)
        self.assertEqual(out["integration_status"], "partial")
        self.assertAlmostEqual(out["coverage_fraction"], .5)
        self.assertAlmostEqual(out["charge_distance_observed_pC_um"], 2000)
        self.assertTrue(math.isnan(out["charge_distance_pC_um"]))

    def test_large_cadence_gap_is_not_bridged(self):
        out = summarize_beam_evolution([frame(z) for z in (0, 100, 300, 400)],
                                       plasma_start_um=0, plasma_end_um=400, max_gap_um=150)
        self.assertEqual(out["n_skipped_gap_segments"], 1)
        self.assertAlmostEqual(out["coverage_fraction"], .5)

    def test_truncated_run_is_partial_without_extrapolated_zeros(self):
        out = summarize_beam_evolution([frame(0), frame(100)], plasma_start_um=0, plasma_end_um=200)
        self.assertAlmostEqual(out["charge_distance_observed_pC_um"], 1000)
        self.assertTrue(math.isnan(out["charge_distance_pC_um"]))

    def test_no_population_gives_zero_charge_integral_and_undefined_quality(self):
        rows = [summarize_beam_frame(make_dump([], iteration=it), species_scope="electrons", z_frame_um=z)
                for it, z in ((0, 0), (1, 100))]
        out = summarize_beam_evolution(rows, plasma_start_um=0, plasma_end_um=100)
        self.assertEqual(out["charge_distance_pC_um"], 0)
        self.assertTrue(math.isnan(out["theta_x_std_mrad_charge_weighted_mean"]))

    def test_insufficient_statistics_do_not_become_good_quality_integral(self):
        rows = [frame(0), frame(100)]
        rows[1]["quality_status"] = "insufficient_statistics"
        out = summarize_beam_evolution(rows, plasma_start_um=0, plasma_end_um=100)
        self.assertEqual(out["integration_status"], "complete")
        self.assertTrue(math.isnan(out["theta_x_std_mrad_charge_weighted_mean"]))
        self.assertEqual(out["theta_x_std_mrad_coverage_fraction"], 0)

    def test_unknown_coordinate_breaks_segments_instead_of_dropping_row(self):
        rows = [frame(z) for z in (0, 100, 200, 300, 400)]
        rows[2]["z_frame_um"] = float("nan")
        out = summarize_beam_evolution(rows, plasma_start_um=0, plasma_end_um=400, max_gap_um=150)
        self.assertAlmostEqual(out["coverage_fraction"], .5)

    def test_exact_exit_does_not_fallback_to_nearest_or_last(self):
        rows = [frame(0, charge=20), frame(100, charge=10)]
        out = summarize_beam_evolution(rows, plasma_start_um=0, plasma_end_um=100, exit_iteration=99)
        self.assertEqual(out["exit_snapshot_status"], "missing")
        self.assertTrue(math.isnan(out["charge_exit_snapshot_pC"]))
        exact = summarize_beam_evolution(rows, plasma_start_um=0, plasma_end_um=100, exit_iteration=100)
        self.assertEqual(exact["charge_exit_snapshot_pC"], 10)

    def test_species_threshold_and_coordinate_order_cannot_be_mixed(self):
        rows = [frame(0), frame(100)]
        rows[1]["species_scope"] = "ionized_electrons"
        with self.assertRaisesRegex(ValueError, "species_scope"):
            summarize_beam_evolution(rows, plasma_start_um=0, plasma_end_um=100)
        rows[1]["species_scope"] = "electrons"
        rows[1]["energy_threshold_MeV"] = 100
        with self.assertRaisesRegex(ValueError, "energy_threshold"):
            summarize_beam_evolution(rows, plasma_start_um=0, plasma_end_um=100)
        rows[1]["energy_threshold_MeV"] = 50
        rows[1]["z_frame_um"] = -10
        with self.assertRaisesRegex(ValueError, "non-decreasing"):
            summarize_beam_evolution(rows, plasma_start_um=0, plasma_end_um=100)


if __name__ == "__main__":
    unittest.main()

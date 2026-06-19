from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cap_guiding.transverse import summarize_transverse_metrics


class DummyDump:
    def __init__(
        self,
        *,
        x_m,
        y_m,
        z_m,
        ux,
        uy,
        uz,
        w,
    ) -> None:
        self.x_m = np.asarray(x_m, dtype=float)
        self.y_m = np.asarray(y_m, dtype=float)
        self.z_m = np.asarray(z_m, dtype=float)
        self.ux = np.asarray(ux, dtype=float)
        self.uy = np.asarray(uy, dtype=float)
        self.uz = np.asarray(uz, dtype=float)
        self.w = np.asarray(w, dtype=float)


class TransverseMetricsTests(unittest.TestCase):
    def test_hot_electrons_have_finite_divergence_and_emittance(self) -> None:
        dump = DummyDump(
            x_m=[-1.0e-6, 0.0, 1.0e-6, 2.0e-6],
            y_m=[0.5e-6, -0.5e-6, 1.0e-6, -1.0e-6],
            z_m=[0.0, 1.0e-6, 2.0e-6, 3.0e-6],
            ux=[0.05, -0.02, 0.01, 0.03],
            uy=[0.01, 0.02, -0.03, 0.04],
            uz=[50.0, 55.0, 60.0, 65.0],
            w=[1.0, 1.0, 1.0, 1.0],
        )
        mask = np.array([True, True, True, True])

        row = summarize_transverse_metrics(dump, mask=mask, longitudinal="z")

        self.assertEqual(row["transverse_status"], "ok")
        self.assertEqual(row["n_macroparticles_transverse"], 4)
        self.assertTrue(math.isfinite(row["theta_x_rms_mrad"]))
        self.assertTrue(math.isfinite(row["theta_y_rms_mrad"]))
        self.assertTrue(math.isfinite(row["theta_rms_mrad"]))
        self.assertTrue(math.isfinite(row["theta_r_p95_mrad"]))
        self.assertTrue(math.isfinite(row["emit_x_norm_mm_mrad"]))
        self.assertTrue(math.isfinite(row["emit_y_norm_mm_mrad"]))
        self.assertGreater(row["beam_transverse_quality_score"], 0.0)

    def test_no_selected_hot_electrons_returns_nan_metrics_and_zero_score(self) -> None:
        dump = DummyDump(
            x_m=[0.0, 1.0e-6],
            y_m=[0.0, 1.0e-6],
            z_m=[0.0, 1.0e-6],
            ux=[0.0, 0.0],
            uy=[0.0, 0.0],
            uz=[10.0, 10.0],
            w=[1.0, 1.0],
        )
        mask = np.array([False, False])

        row = summarize_transverse_metrics(dump, mask=mask, longitudinal="z")

        self.assertEqual(row["transverse_status"], "no_selected_particles")
        self.assertEqual(row["n_macroparticles_transverse"], 0)
        self.assertTrue(math.isnan(row["theta_rms_mrad"]))
        self.assertTrue(math.isnan(row["emit_x_norm_mm_mrad"]))
        self.assertEqual(row["beam_transverse_quality_score"], 0.0)

    def test_nonuniform_weights_affect_rms_divergence(self) -> None:
        dump = DummyDump(
            x_m=[0.0, 0.0],
            y_m=[0.0, 0.0],
            z_m=[0.0, 0.0],
            ux=[0.0, 1.0],
            uy=[0.0, 0.0],
            uz=[100.0, 100.0],
            w=[99.0, 1.0],
        )
        mask = np.array([True, True])

        row = summarize_transverse_metrics(dump, mask=mask, longitudinal="z")
        unweighted_theta = math.atan2(1.0, 100.0) * 1.0e3 / math.sqrt(2.0)

        self.assertLess(row["theta_x_rms_mrad"], unweighted_theta)
        self.assertGreater(row["theta_x_rms_mrad"], 0.0)

    def test_emittance_is_nonnegative(self) -> None:
        dump = DummyDump(
            x_m=[-2.0e-6, -1.0e-6, 1.0e-6, 2.0e-6],
            y_m=[-1.0e-6, 2.0e-6, -2.0e-6, 1.0e-6],
            z_m=[0.0, 0.0, 0.0, 0.0],
            ux=[-0.02, 0.01, 0.03, -0.01],
            uy=[0.04, -0.03, 0.02, -0.01],
            uz=[80.0, 80.0, 80.0, 80.0],
            w=[1.0, 2.0, 3.0, 4.0],
        )
        mask = np.array([True, True, True, True])

        row = summarize_transverse_metrics(dump, mask=mask, longitudinal="z")

        self.assertGreaterEqual(row["emit_x_norm_mm_mrad"], 0.0)
        self.assertGreaterEqual(row["emit_y_norm_mm_mrad"], 0.0)
        self.assertGreaterEqual(row["emit_geom_norm_mm_mrad"], 0.0)

    def test_degenerate_zero_variance_has_zero_emittance(self) -> None:
        dump = DummyDump(
            x_m=[1.0e-6, 1.0e-6, 1.0e-6],
            y_m=[-2.0e-6, -2.0e-6, -2.0e-6],
            z_m=[0.0, 1.0e-6, 2.0e-6],
            ux=[0.01, 0.01, 0.01],
            uy=[-0.02, -0.02, -0.02],
            uz=[50.0, 50.0, 50.0],
            w=[1.0, 1.0, 1.0],
        )
        mask = np.array([True, True, True])

        row = summarize_transverse_metrics(dump, mask=mask, longitudinal="z")

        self.assertAlmostEqual(row["x_rms_um"], 0.0)
        self.assertAlmostEqual(row["y_rms_um"], 0.0)
        self.assertAlmostEqual(row["emit_x_norm_mm_mrad"], 0.0)
        self.assertAlmostEqual(row["emit_y_norm_mm_mrad"], 0.0)

    def test_particle_summary_csv_contains_transverse_columns(self) -> None:
        from cap_guiding.particles import (
            ParticleDump,
            summarize_dump,
            write_summary_csv,
        )

        dump = ParticleDump(
            iteration=10,
            time_fs=0.0,
            x_m=np.array([-1.0e-6, 0.0, 1.0e-6]),
            y_m=np.array([0.0, 1.0e-6, -1.0e-6]),
            z_m=np.array([0.0, 1.0e-6, 2.0e-6]),
            ux=np.array([0.02, -0.01, 0.03]),
            uy=np.array([0.01, 0.02, -0.02]),
            uz=np.array([50.0, 60.0, 70.0]),
            w=np.array([1.0, 2.0, 3.0]),
        )

        row = summarize_dump(
            dump,
            hot_energy_mev=10.0,
            longitudinal="z",
            forward_only=True,
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "particle_summary.csv"
            write_summary_csv([row], path)

            with path.open("r", encoding="utf-8", newline="") as f_in:
                loaded = list(csv.DictReader(f_in))[0]

        self.assertIn("theta_rms_mrad", loaded)
        self.assertIn("theta_r_p95_mrad", loaded)
        self.assertIn("emit_x_norm_mm_mrad", loaded)
        self.assertIn("beam_transverse_quality_score", loaded)
        self.assertGreater(float(loaded["beam_transverse_quality_score"]), 0.0)


if __name__ == "__main__":
    unittest.main()

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import h5py
import numpy as np

from laser_guiding.visualization import visualize
from laser_guiding_fixtures import config, make_case


class VisualizationIntegrationTests(unittest.TestCase):
    def test_fields_only_does_not_open_particle_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = make_case(tmp, particle=False)
            output = Path(tmp) / "output"
            with patch("laser_guiding.particle_histograms.histograms",
                       side_effect=AssertionError("Particle reader was invoked")):
                metadata = visualize(case, config(), output, requested=[10, 20],
                                     static_diagnostics=True, min_age_seconds=0)
            self.assertEqual(len(list((output / "field_frames").glob("*.png"))), 2)
            self.assertFalse((output / "particle_histograms").exists())
            self.assertFalse(metadata["particles_enabled"])
            self.assertEqual(metadata["frames"][0]["reference_z_m"],
                             -333e-6 + 299792458.0 * 1e-13)
            self.assertEqual(json.loads((output / "visualization.json").read_text())["status"], "complete")
            self.assertEqual(metadata["field_scales"]["density"]["data_max"], 1.2e16)

    def test_particles_are_read_once_per_frame_and_small_products_persist(self):
        from laser_guiding.particle_histograms import histograms
        with tempfile.TemporaryDirectory() as tmp:
            case = make_case(tmp)
            output = Path(tmp) / "output"
            with patch("laser_guiding.particle_histograms.histograms", wraps=histograms) as read:
                metadata = visualize(case, config(), output, requested=[10, 20],
                                     with_particles=True, static_diagnostics=True, min_age_seconds=0)
            self.assertEqual(read.call_count, 2)
            self.assertTrue(metadata["particles_enabled"])
            self.assertEqual(len(list((output / "particle_frames").glob("*.png"))), 2)
            with np.load(output / "particle_histograms" / "00000010.npz") as hist:
                row = metadata["frames"][0]["electron_species"][0]
                self.assertAlmostEqual(hist["charge_spectrum_pc"].sum(), row["Q_selected_pC"])

    def test_closed_or_logged_input_required_and_raw_paths_protected(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = make_case(tmp)
            with self.assertRaisesRegex(ValueError, "completed-step log"):
                visualize(case, config(), Path(tmp) / "out", requested=[10])
            with self.assertRaisesRegex(ValueError, "outside raw"):
                visualize(case, config(), case / "diags" / "fields" / "out",
                          requested=[10], static_diagnostics=True)

    def test_timestamp_mismatch_fails_without_complete_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = make_case(tmp)
            with h5py.File(case / "diags/plasma_electrons/particles_10.h5", "r+") as handle:
                handle["data/10"].attrs["time"] = 9e-13
            output = Path(tmp) / "out"
            with self.assertRaisesRegex(ValueError, "timestamps"):
                visualize(case, config(), output, requested=[10], with_particles=True,
                          static_diagnostics=True, min_age_seconds=0)
            self.assertFalse((output / "visualization.json").exists())

    def test_actual_mp4_encoding(self):
        with tempfile.TemporaryDirectory() as tmp:
            case = make_case(tmp, particle=False)
            output = Path(tmp) / "out"
            visualize(case, config(), output, requested=[10, 20],
                      static_diagnostics=True, min_age_seconds=0, movie=True)
            self.assertGreater((output / "fields_xz.mp4").stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

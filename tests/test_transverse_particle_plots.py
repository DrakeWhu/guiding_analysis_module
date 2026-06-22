from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from cap_guiding.particles import ParticleDump, save_transverse_phase_space_plots


class TransverseParticlePlotTests(unittest.TestCase):
    def test_transverse_phase_space_plots_are_written(self) -> None:
        dump = ParticleDump(
            iteration=10,
            time_fs=0.0,
            x_m=np.array([-1.0e-6, 0.0, 1.0e-6, 2.0e-6]),
            y_m=np.array([0.5e-6, -0.5e-6, 1.0e-6, -1.0e-6]),
            z_m=np.array([0.0, 1.0e-6, 2.0e-6, 3.0e-6]),
            ux=np.array([0.05, -0.02, 0.01, 0.03]),
            uy=np.array([0.01, 0.02, -0.03, 0.04]),
            uz=np.array([50.0, 55.0, 60.0, 65.0]),
            w=np.array([1.0, 1.0, 1.0, 1.0]),
        )

        with tempfile.TemporaryDirectory() as tmp:
            paths = save_transverse_phase_space_plots(
                dump,
                outdir=Path(tmp),
                suffix="it00000010",
                hot_energy_mev=10.0,
                longitudinal="z",
                forward_only=True,
            )

            self.assertEqual(len(paths), 4)
            for path in paths:
                self.assertTrue(path.exists())
                self.assertGreater(path.stat().st_size, 0)


if __name__ == "__main__":
    unittest.main()

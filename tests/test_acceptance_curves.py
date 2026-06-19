from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cap_guiding.metrics import E_CHARGE_C
from cap_guiding.particles import (
    ELECTRON_REST_ENERGY_MEV,
    ParticleDump,
    summarize_acceptance_curves,
    write_acceptance_curves_csv,
)


def momentum_for_energy_and_theta(
    energy_mev: float,
    theta_mrad: float,
) -> tuple[float, float, float]:
    gamma = 1.0 + float(energy_mev) / ELECTRON_REST_ENERGY_MEV
    u_total = math.sqrt(gamma * gamma - 1.0)
    theta = float(theta_mrad) * 1.0e-3
    uz = u_total / math.sqrt(1.0 + math.tan(theta) ** 2)
    ux = uz * math.tan(theta)
    return ux, 0.0, uz


class AcceptanceCurveTests(unittest.TestCase):
    def make_dump(self) -> ParticleDump:
        momenta = [
            momentum_for_energy_and_theta(20.0, 1.0),
            momentum_for_energy_and_theta(120.0, 6.0),
            momentum_for_energy_and_theta(260.0, 30.0),
            momentum_for_energy_and_theta(400.0, 60.0),
        ]
        ux, uy, uz = [np.asarray(values, dtype=float) for values in zip(*momenta)]
        return ParticleDump(
            iteration=32000,
            time_fs=1.0,
            x_m=np.zeros(4),
            y_m=np.zeros(4),
            z_m=np.zeros(4),
            ux=ux,
            uy=uy,
            uz=uz,
            w=np.array([2.0, 3.0, 5.0, 7.0]),
        )

    def row_for(self, rows, *, theta_cut_mrad: float, e_min_mev: float):
        matches = [
            row
            for row in rows
            if row["theta_cut_mrad"] == theta_cut_mrad and row["E_min_MeV"] == e_min_mev
        ]
        self.assertEqual(len(matches), 1)
        return matches[0]

    def test_acceptance_grid_uses_energy_divergence_and_weights(self) -> None:
        rows = summarize_acceptance_curves(
            self.make_dump(),
            case_id="000",
            case_name="000_case",
            selection_mode="exit",
            selected_particle_iteration=32000,
            theta_cuts_mrad=[5.0, 10.0, 50.0],
            e_min_mev=[10.0, 100.0, 250.0, 300.0],
            longitudinal="z",
            forward_only=True,
        )

        self.assertEqual(len(rows), 12)
        self.assertEqual(
            self.row_for(rows, theta_cut_mrad=5.0, e_min_mev=10.0)["accepted_weight"],
            2.0,
        )
        self.assertEqual(
            self.row_for(rows, theta_cut_mrad=10.0, e_min_mev=100.0)["accepted_weight"],
            3.0,
        )
        self.assertEqual(
            self.row_for(rows, theta_cut_mrad=50.0, e_min_mev=250.0)["accepted_weight"],
            5.0,
        )
        self.assertEqual(
            self.row_for(rows, theta_cut_mrad=50.0, e_min_mev=300.0)["accepted_weight"],
            0.0,
        )

        row = self.row_for(rows, theta_cut_mrad=10.0, e_min_mev=100.0)
        self.assertEqual(row["case_id"], "000")
        self.assertEqual(row["case_name"], "000_case")
        self.assertEqual(row["iteration"], 32000)
        self.assertEqual(row["selection_mode"], "exit")
        self.assertEqual(row["selected_particle_iteration"], 32000)
        self.assertEqual(row["accepted_n_macroparticles"], 1)
        self.assertAlmostEqual(
            row["accepted_charge_pC"],
            3.0 * E_CHARGE_C / 1.0e-12,
        )

    def test_forward_cut_removes_backward_particles(self) -> None:
        dump = self.make_dump()
        dump = ParticleDump(
            iteration=dump.iteration,
            time_fs=dump.time_fs,
            x_m=dump.x_m,
            y_m=dump.y_m,
            z_m=dump.z_m,
            ux=dump.ux,
            uy=dump.uy,
            uz=np.array([dump.uz[0], -dump.uz[1], dump.uz[2], dump.uz[3]]),
            w=dump.w,
        )

        forward_rows = summarize_acceptance_curves(
            dump,
            theta_cuts_mrad=[5000.0],
            e_min_mev=[100.0],
            forward_only=True,
        )
        no_forward_rows = summarize_acceptance_curves(
            dump,
            theta_cuts_mrad=[5000.0],
            e_min_mev=[100.0],
            forward_only=False,
        )

        self.assertEqual(forward_rows[0]["accepted_weight"], 12.0)
        self.assertEqual(no_forward_rows[0]["accepted_weight"], 15.0)

    def test_write_acceptance_csv_has_expected_columns(self) -> None:
        rows = summarize_acceptance_curves(
            self.make_dump(),
            theta_cuts_mrad=[5.0],
            e_min_mev=[10.0],
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "particle_acceptance_curves.csv"
            write_acceptance_curves_csv(rows, path)

            with path.open("r", encoding="utf-8", newline="") as f_in:
                reader = csv.DictReader(f_in)
                loaded = list(reader)
                fieldnames = reader.fieldnames

        self.assertEqual(loaded[0]["theta_cut_mrad"], "5.0")
        self.assertIn("accepted_charge_pC", fieldnames or [])
        self.assertIn("accepted_weight", fieldnames or [])
        self.assertIn("accepted_n_macroparticles", fieldnames or [])


if __name__ == "__main__":
    unittest.main()

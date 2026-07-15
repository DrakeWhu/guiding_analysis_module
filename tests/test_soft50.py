from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import numpy as np

from cap_guiding.metrics import E_CHARGE_C
from cap_guiding.particles import ParticleDump, concatenate_particle_dumps
from cap_guiding.soft50 import (
    SOFT50_SCHEMA_VERSION,
    Soft50Config,
    effective_sample_size,
    smooth_energy_acceptance,
    soft_reliability,
    summarize_soft50_curve,
    summarize_soft50_metrics,
    weighted_percentile,
    write_soft50_curve_csv,
)


REST_MEV = 0.51099895


def dump_from_energy(
    energy_mev: list[float],
    *,
    weights: list[float] | None = None,
    directions: list[float] | None = None,
    z_m: list[float] | None = None,
) -> ParticleDump:
    energy = np.asarray(energy_mev, dtype=float)
    gamma = energy / REST_MEV + 1.0
    momentum = np.sqrt(np.maximum(gamma * gamma - 1.0, 0.0))
    momentum[~np.isfinite(energy)] = np.nan
    if directions is not None:
        momentum *= np.asarray(directions, dtype=float)
    n = len(energy)
    return ParticleDump(
        iteration=100,
        time_fs=1.0,
        x_m=np.linspace(-1.0e-6, 1.0e-6, n),
        y_m=np.linspace(1.0e-6, -1.0e-6, n),
        z_m=np.asarray(z_m if z_m is not None else np.zeros(n), dtype=float),
        ux=np.linspace(-0.1, 0.1, n),
        uy=np.linspace(0.1, -0.1, n),
        uz=momentum,
        w=np.asarray(weights if weights is not None else np.ones(n), dtype=float),
    )


class Soft50Tests(unittest.TestCase):
    def test_acceptance_is_continuous_and_clipped(self) -> None:
        values = smooth_energy_acceptance(
            np.asarray([0.0, 10.0, 30.0, 50.0, 100.0, np.nan]),
            energy_low_mev=10.0,
            energy_target_mev=50.0,
        )
        np.testing.assert_allclose(values, [0.0, 0.0, 0.5, 1.0, 1.0, 0.0])

    def test_all_particles_below_low_have_zero_soft_charge(self) -> None:
        row = summarize_soft50_metrics(dump_from_energy([1.0, 5.0, 9.9]))
        self.assertEqual(row["soft50_status"], "no_accepted_particles")
        self.assertEqual(row["charge_soft50_pC"], 0.0)
        self.assertEqual(row["n_effective_soft50"], 0.0)
        self.assertEqual(row["n_macroparticles_Ege50MeV"], 0)

    def test_all_particles_above_target_count_fully(self) -> None:
        row = summarize_soft50_metrics(
            dump_from_energy([50.0, 60.0], weights=[2.0, 3.0])
        )
        expected = 5.0 * E_CHARGE_C / 1.0e-12
        self.assertAlmostEqual(row["charge_soft50_pC"], expected)
        self.assertAlmostEqual(row["charge_Ege50MeV_pC"], expected)
        self.assertEqual(row["n_macroparticles_Ege50MeV"], 2)

    def test_intermediate_particles_receive_smooth_credit(self) -> None:
        row = summarize_soft50_metrics(
            dump_from_energy([30.0], weights=[4.0]),
            config=Soft50Config(energy_low_mev=10.0, energy_target_mev=50.0),
        )
        self.assertAlmostEqual(row["weight_soft50"], 2.0, places=4)
        self.assertEqual(row["charge_Ege50MeV_pC"], 0.0)

    def test_nonuniform_weights_and_effective_count(self) -> None:
        self.assertAlmostEqual(effective_sample_size(np.asarray([1.0, 2.0, 3.0])), 36 / 14)
        row = summarize_soft50_metrics(
            dump_from_energy([60.0, 60.0, 60.0], weights=[1.0, 2.0, 3.0])
        )
        self.assertAlmostEqual(row["n_effective_soft50"], 36 / 14)

    def test_one_huge_macroparticle_has_effective_count_near_one(self) -> None:
        n_eff = effective_sample_size(np.asarray([1.0e12, 1.0, 1.0]))
        self.assertLess(n_eff, 1.00000000001)
        self.assertGreaterEqual(n_eff, 1.0)

    def test_reliability_is_continuous_and_bounded(self) -> None:
        self.assertAlmostEqual(
            soft_reliability(
                0.0, reliability_floor=0.1, effective_count_reference=100.0
            ),
            0.1,
        )
        self.assertLess(
            soft_reliability(
                100.0, reliability_floor=0.1, effective_count_reference=100.0
            ),
            1.0,
        )

    def test_nan_infinite_and_nonpositive_weights_are_ignored(self) -> None:
        dump = dump_from_energy(
            [60.0, np.nan, 70.0, 80.0], weights=[1.0, 1.0, np.inf, -2.0]
        )
        row = summarize_soft50_metrics(dump)
        self.assertEqual(row["n_macroparticles_soft50"], 1)
        self.assertEqual(row["n_macroparticles_Ege50MeV"], 1)

    def test_empty_dump_is_supported(self) -> None:
        row = summarize_soft50_metrics(dump_from_energy([]))
        self.assertEqual(row["soft50_status"], "no_accepted_particles")
        self.assertTrue(np.isnan(row["energy_p50_soft50_MeV"]))
        self.assertTrue(np.isnan(row["energy_relative_spread_rms_soft50"]))

    def test_energy_spread_is_zero_for_monoenergetic_particles(self) -> None:
        row = summarize_soft50_metrics(
            dump_from_energy([60.0, 60.0, 60.0], weights=[1.0, 2.0, 4.0])
        )
        self.assertAlmostEqual(row["energy_mean_soft50_MeV"], 60.0, places=3)
        self.assertLess(row["energy_spread_rms_soft50_MeV"], 1.0e-3)
        self.assertLess(row["energy_relative_spread_rms_soft50"], 1.0e-6)

    def test_energy_spread_uses_physical_and_soft_acceptance_weights(self) -> None:
        row = summarize_soft50_metrics(
            dump_from_energy([20.0, 40.0, 80.0], weights=[1.0, 2.0, 1.0])
        )
        self.assertGreater(row["energy_mean_soft50_MeV"], 40.0)
        self.assertGreater(row["energy_spread_rms_soft50_MeV"], 0.0)
        self.assertGreater(row["energy_relative_spread_rms_soft50"], 0.0)
        self.assertLessEqual(
            row["energy_p10_soft50_MeV"], row["energy_p90_soft50_MeV"]
        )

    def test_forward_only_and_exit_window_are_applied_before_acceptance(self) -> None:
        dump = dump_from_energy(
            [60.0, 60.0, 60.0],
            directions=[1.0, -1.0, 1.0],
            z_m=[0.0, 1.0e-3, 2.0e-3],
        )
        row = summarize_soft50_metrics(dump, exit_window_mm=0.5)
        self.assertEqual(row["n_macroparticles_soft50"], 1)

    def test_weighted_percentiles_use_soft_weights(self) -> None:
        self.assertEqual(
            weighted_percentile(
                np.asarray([20.0, 40.0, 60.0]),
                np.asarray([0.01, 0.09, 0.90]),
                50.0,
            ),
            60.0,
        )
        row = summarize_soft50_metrics(dump_from_energy([20.0, 40.0, 60.0]))
        self.assertGreaterEqual(row["energy_p50_soft50_MeV"], 40.0)

    def test_soft_charge_is_never_below_hard_charge(self) -> None:
        row = summarize_soft50_metrics(dump_from_energy([20.0, 49.0, 50.0, 80.0]))
        self.assertGreaterEqual(row["charge_soft50_pC"], row["charge_Ege50MeV_pC"])

    def test_species_dumps_can_be_combined(self) -> None:
        first = dump_from_energy([20.0, 60.0])
        second = dump_from_energy([55.0])
        combined = concatenate_particle_dumps([first, second])
        self.assertEqual(len(combined.w), 3)
        row = summarize_soft50_metrics(combined)
        self.assertEqual(row["n_macroparticles_Ege50MeV"], 2)

    def test_curve_persists_multiple_low_energy_choices(self) -> None:
        rows = summarize_soft50_curve(
            dump_from_energy([6.0, 20.0, 60.0]),
            energy_low_values_mev=[10.0, 5.0],
            metadata={
                "case_id": "1",
                "case_name": "case",
                "species_scope": "all_electrons",
                "selection_mode": "exit",
                "selected_particle_iteration": 100,
            },
        )
        self.assertEqual([row["soft50_energy_low_MeV"] for row in rows], [5.0, 10.0])
        with tempfile.TemporaryDirectory() as tmp:
            path = write_soft50_curve_csv(rows, Path(tmp) / "soft50.csv")
            with path.open(newline="", encoding="utf-8") as stream:
                written = list(csv.DictReader(stream))
        self.assertEqual(len(written), 2)
        self.assertEqual(written[0]["soft50_schema_version"], SOFT50_SCHEMA_VERSION)
        self.assertIn("energy_relative_spread_rms_soft50", written[0])

    def test_invalid_config_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            Soft50Config(energy_low_mev=50.0, energy_target_mev=50.0)


if __name__ == "__main__":
    unittest.main()

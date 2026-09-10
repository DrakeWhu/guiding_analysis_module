from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from cap_guiding.metrics import E_CHARGE_C
from cap_guiding import particles
from cap_guiding.particles import ParticleDump, concatenate_particle_dumps
from cap_guiding.soft50 import Soft50Config, summarize_soft50_metrics


REST_MEV = 0.51099895


def dump_from_energy(
    energy_mev: list[float],
    *,
    weights: list[float] | None = None,
    iteration: int = 100,
    time_fs: float = 1.0,
) -> ParticleDump:
    energy = np.asarray(energy_mev, dtype=float)
    gamma = energy / REST_MEV + 1.0
    momentum = np.sqrt(np.maximum(gamma * gamma - 1.0, 0.0))
    n = len(energy)
    return ParticleDump(
        iteration=int(iteration),
        time_fs=float(time_fs),
        x_m=np.linspace(-1.0e-6, 1.0e-6, n),
        y_m=np.linspace(1.0e-6, -1.0e-6, n),
        z_m=np.zeros(n, dtype=float),
        ux=np.zeros(n, dtype=float),
        uy=np.zeros(n, dtype=float),
        uz=momentum,
        w=np.asarray(weights if weights is not None else np.ones(n), dtype=float),
    )


class FakeParticleSeries:
    def __init__(self, *, arrays=None, error: Exception | None = None) -> None:
        self.arrays = arrays
        self.error = error
        self.iterations = np.asarray([100], dtype=int)
        self.t = np.asarray([1.0e-15], dtype=float)

    def get_particle(self, *, var_list, species, iteration):
        if self.error is not None:
            raise self.error
        return self.arrays


class ParticleMultispeciesContractTests(unittest.TestCase):
    def test_two_species_same_iteration_and_time_concatenate_raw_particles(self) -> None:
        background = dump_from_energy([20.0, 60.0], weights=[2.0, 3.0])
        ionized = dump_from_energy([40.0, 80.0], weights=[5.0, 7.0])

        combined = concatenate_particle_dumps([background, ionized])

        self.assertEqual(combined.iteration, 100)
        self.assertEqual(combined.time_fs, 1.0)
        np.testing.assert_allclose(combined.w, [2.0, 3.0, 5.0, 7.0])
        np.testing.assert_allclose(
            combined.kinetic_energy_mev,
            [20.0, 60.0, 40.0, 80.0],
            rtol=0.0,
            atol=1.0e-10,
        )

    def test_iteration_mismatch_is_rejected(self) -> None:
        background = dump_from_energy([60.0], iteration=100)
        ionized = dump_from_energy([80.0], iteration=101)
        with self.assertRaisesRegex(ValueError, "different iterations"):
            concatenate_particle_dumps([background, ionized])

    def test_time_mismatch_is_rejected(self) -> None:
        background = dump_from_energy([60.0], time_fs=1.0)
        ionized = dump_from_energy([80.0], time_fs=2.0)
        with self.assertRaisesRegex(ValueError, "different times"):
            concatenate_particle_dumps([background, ionized])

    def test_empty_registered_species_is_valid_and_does_not_change_aggregate(self) -> None:
        background = dump_from_energy([20.0, 60.0], weights=[2.0, 3.0])
        empty_ionized = dump_from_energy([], weights=[])

        combined = concatenate_particle_dumps([background, empty_ionized])
        background_metrics = summarize_soft50_metrics(background)
        combined_metrics = summarize_soft50_metrics(combined)

        self.assertEqual(len(empty_ionized.w), 0)
        self.assertEqual(len(combined.w), len(background.w))
        for key in (
            "charge_soft50_pC",
            "charge_Ege50MeV_pC",
            "weight_soft50",
            "n_macroparticles_Ege50MeV",
        ):
            self.assertEqual(combined_metrics[key], background_metrics[key])

    def test_absent_species_is_not_silently_converted_to_empty(self) -> None:
        series = FakeParticleSeries(error=KeyError("nitrogen_ionized_electrons"))
        with patch.object(particles, "open_series", return_value=series):
            with self.assertRaises(KeyError):
                particles.read_particle_dump(
                    "unused",
                    species="nitrogen_ionized_electrons",
                    iteration=100,
                )

    def test_registered_empty_species_reads_as_empty_dump(self) -> None:
        arrays = [np.asarray([], dtype=float) for _ in particles.DEFAULT_PARTICLE_VARS]
        series = FakeParticleSeries(arrays=arrays)
        with patch.object(particles, "open_series", return_value=series):
            dump = particles.read_particle_dump(
                "unused",
                species="nitrogen_ionized_electrons",
                iteration=100,
            )

        self.assertEqual(dump.iteration, 100)
        self.assertAlmostEqual(dump.time_fs, 1.0)
        self.assertEqual(len(dump.w), 0)

    def test_soft50_and_hard50_aggregate_are_sum_of_species_charges(self) -> None:
        config = Soft50Config(energy_low_mev=10.0, energy_target_mev=50.0)
        background = dump_from_energy([20.0, 60.0], weights=[2.0, 3.0])
        ionized = dump_from_energy([40.0, 80.0], weights=[5.0, 7.0])
        combined = concatenate_particle_dumps([background, ionized])

        background_metrics = summarize_soft50_metrics(background, config=config)
        ionized_metrics = summarize_soft50_metrics(ionized, config=config)
        combined_metrics = summarize_soft50_metrics(combined, config=config)

        self.assertAlmostEqual(
            combined_metrics["charge_soft50_pC"],
            background_metrics["charge_soft50_pC"]
            + ionized_metrics["charge_soft50_pC"],
            places=12,
        )
        self.assertAlmostEqual(
            combined_metrics["charge_Ege50MeV_pC"],
            background_metrics["charge_Ege50MeV_pC"]
            + ionized_metrics["charge_Ege50MeV_pC"],
            places=12,
        )
        self.assertAlmostEqual(
            combined_metrics["charge_Ege50MeV_pC"],
            10.0 * E_CHARGE_C / 1.0e-12,
            places=12,
        )

    def test_nonlinear_metrics_are_recomputed_on_combined_particles(self) -> None:
        config = Soft50Config(energy_low_mev=10.0, energy_target_mev=50.0)
        background = dump_from_energy([20.0, 60.0], weights=[2.0, 3.0])
        ionized = dump_from_energy([40.0, 80.0], weights=[5.0, 7.0])
        combined = concatenate_particle_dumps([background, ionized])

        background_metrics = summarize_soft50_metrics(background, config=config)
        ionized_metrics = summarize_soft50_metrics(ionized, config=config)
        combined_metrics = summarize_soft50_metrics(combined, config=config)

        per_species_mean_of_medians = 0.5 * (
            background_metrics["energy_p50_soft50_MeV"]
            + ionized_metrics["energy_p50_soft50_MeV"]
        )
        self.assertAlmostEqual(combined_metrics["energy_p50_soft50_MeV"], 60.0, places=8)
        self.assertNotAlmostEqual(
            combined_metrics["energy_p50_soft50_MeV"],
            per_species_mean_of_medians,
            places=8,
        )
        self.assertGreater(combined_metrics["energy_spread_rms_soft50_MeV"], 0.0)


if __name__ == "__main__":
    unittest.main()

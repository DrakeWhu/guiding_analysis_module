from __future__ import annotations

import unittest

from cap_guiding.beamlike import (
    BEAMLIKE_OUTPUT_COLUMNS,
    BeamlikeConfig,
    add_beamlike_metrics,
    divergence_component_from_row,
    score_particle_summary_row,
)


class BeamlikeScoringTests(unittest.TestCase):
    def test_good_beamlike_candidate_is_eligible(self) -> None:
        row = {
            "n_macroparticles_hot": 1137,
            "charge_hot_pC": 1231.64,
            "E95_hot_MeV": 212.02,
            "Emean_hot_MeV": 58.53,
            "Emax_hot_MeV": 340.10,
            "q_long_min_hot_mm": 9.80,
            "q_long_max_hot_mm": 10.20,
        }

        scored = score_particle_summary_row(row)

        self.assertTrue(scored["eligible_beamlike"])
        self.assertEqual(scored["beamlike_status"], "eligible_beamlike")
        self.assertGreater(scored["beamlike_score"], 0.0)
        self.assertGreater(scored["beam_yield_score"], 0.0)
        self.assertEqual(scored["beamlike_rejection_reasons"], "")
        self.assertIn("beamlike_candidate", scored["beamlike_tags"])
        self.assertIn("no_divergence_metric", scored["beamlike_tags"])
        self.assertAlmostEqual(
            scored["mono_proxy_E95_over_Emax"],
            212.02 / 340.10,
        )

    def test_low_hot_statistics_are_rejected(self) -> None:
        row = {
            "n_macroparticles_hot": 15,
            "charge_hot_pC": 500.0,
            "E95_hot_MeV": 150.0,
            "Emean_hot_MeV": 80.0,
            "Emax_hot_MeV": 300.0,
            "q_long_min_hot_mm": 9.90,
            "q_long_max_hot_mm": 10.10,
        }

        scored = score_particle_summary_row(row)

        self.assertFalse(scored["eligible_beamlike"])
        self.assertEqual(
            scored["beamlike_status"],
            "insufficient_hot_electron_statistics",
        )
        self.assertEqual(scored["beamlike_score"], 0.0)
        self.assertEqual(scored["beam_yield_score"], 0.0)
        self.assertIn("low_n_hot<200", scored["beamlike_rejection_reasons"])

    def test_low_charge_is_rejected(self) -> None:
        row = {
            "n_macroparticles_hot": 1000,
            "charge_hot_pC": 20.0,
            "E95_hot_MeV": 150.0,
            "Emean_hot_MeV": 80.0,
            "Emax_hot_MeV": 300.0,
        }

        scored = score_particle_summary_row(row)

        self.assertFalse(scored["eligible_beamlike"])
        self.assertEqual(scored["beamlike_status"], "insufficient_hot_charge")
        self.assertIn("low_charge<100pC", scored["beamlike_rejection_reasons"])

    def test_low_e95_is_rejected(self) -> None:
        row = {
            "n_macroparticles_hot": 1000,
            "charge_hot_pC": 500.0,
            "E95_hot_MeV": 20.0,
            "Emean_hot_MeV": 15.0,
            "Emax_hot_MeV": 200.0,
        }

        scored = score_particle_summary_row(row)

        self.assertFalse(scored["eligible_beamlike"])
        self.assertEqual(scored["beamlike_status"], "insufficient_hot_energy")
        self.assertIn("low_E95<50MeV", scored["beamlike_rejection_reasons"])

    def test_broad_spectrum_proxy_is_quality_flag_not_hard_rejection(self) -> None:
        row = {
            "n_macroparticles_hot": 1000,
            "charge_hot_pC": 500.0,
            "E95_hot_MeV": 60.0,
            "Emean_hot_MeV": 40.0,
            "Emax_hot_MeV": 600.0,
        }

        scored = score_particle_summary_row(row)

        self.assertTrue(scored["eligible_beamlike"])
        self.assertEqual(scored["beamlike_status"], "eligible_with_quality_flags")
        self.assertGreaterEqual(scored["beamlike_score"], 0.0)
        self.assertIn(
            "broad_proxy_E95_over_Emax<0.3",
            scored["beamlike_rejection_reasons"],
        )
        self.assertIn("broad_spectrum_proxy", scored["beamlike_tags"])

    def test_missing_divergence_is_neutral_for_phase_1(self) -> None:
        component, source = divergence_component_from_row({})

        self.assertEqual(component, 1.0)
        self.assertEqual(source, "not_available")

    def test_existing_theta_rms_is_used_for_divergence_component(self) -> None:
        component, source = divergence_component_from_row(
            {"theta_x_rms_mrad": 3.0, "theta_y_rms_mrad": 4.0}
        )

        self.assertEqual(source, "theta_x/y_rms_mrad")
        self.assertGreater(component, 0.0)
        self.assertLess(component, 1.0)

    def test_all_declared_output_columns_are_present(self) -> None:
        row = {
            "n_macroparticles_hot": 0,
            "charge_hot_pC": 0.0,
            "E95_hot_MeV": float("nan"),
            "Emean_hot_MeV": float("nan"),
            "Emax_hot_MeV": float("nan"),
        }

        scored = score_particle_summary_row(row)

        for column in BEAMLIKE_OUTPUT_COLUMNS:
            self.assertIn(column, scored)

    def test_add_beamlike_metrics_preserves_original_row(self) -> None:
        row = {
            "iteration": 123,
            "n_macroparticles_hot": 1000,
            "charge_hot_pC": 500.0,
            "E95_hot_MeV": 120.0,
            "Emean_hot_MeV": 60.0,
            "Emax_hot_MeV": 200.0,
        }

        enriched = add_beamlike_metrics(row)

        self.assertEqual(enriched["iteration"], 123)
        self.assertIn("beamlike_score", enriched)
        self.assertNotIn("beamlike_score", row)

    def test_thresholds_are_configurable(self) -> None:
        row = {
            "n_macroparticles_hot": 150,
            "charge_hot_pC": 80.0,
            "E95_hot_MeV": 40.0,
            "Emean_hot_MeV": 35.0,
            "Emax_hot_MeV": 100.0,
        }

        strict = score_particle_summary_row(row)
        relaxed = score_particle_summary_row(
            row,
            config=BeamlikeConfig(
                min_hot_macroparticles=100.0,
                min_hot_charge_pC=50.0,
                min_hot_E95_MeV=30.0,
            ),
        )

        self.assertFalse(strict["eligible_beamlike"])
        self.assertTrue(relaxed["eligible_beamlike"])

    def test_add_beamlike_metrics_returns_dict_not_none(self) -> None:
        row = {
            "n_macroparticles_hot": 0,
            "charge_hot_pC": 0.0,
            "E95_hot_MeV": float("nan"),
            "Emean_hot_MeV": float("nan"),
            "Emax_hot_MeV": float("nan"),
            "q_long_min_hot_mm": float("nan"),
            "q_long_max_hot_mm": float("nan"),
        }

        out = add_beamlike_metrics(row)

        self.assertIsInstance(out, dict)
        self.assertIn("beamlike_score", out)
        self.assertEqual(out["eligible_beamlike"], False)

    def test_summarize_dump_returns_dict_when_no_hot_electrons(self) -> None:
        import numpy as np

        from cap_guiding.particles import ParticleDump, summarize_dump

        dump = ParticleDump(
            iteration=1,
            time_fs=0.0,
            x_m=np.array([0.0, 1.0e-6]),
            y_m=np.array([0.0, 0.0]),
            z_m=np.array([0.0, 1.0e-6]),
            ux=np.array([0.0, 0.0]),
            uy=np.array([0.0, 0.0]),
            uz=np.array([0.0, 0.0]),
            w=np.array([1.0, 1.0]),
        )

        row = summarize_dump(
            dump,
            hot_energy_mev=10.0,
            longitudinal="z",
            forward_only=True,
        )

        self.assertIsInstance(row, dict)
        self.assertEqual(row["n_macroparticles_hot"], 0)
        self.assertEqual(row["charge_hot_pC"], 0.0)
        self.assertIn("beamlike_score", row)
        self.assertEqual(row["eligible_beamlike"], False)


if __name__ == "__main__":
    unittest.main()

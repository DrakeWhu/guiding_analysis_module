from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from cap_guiding.beamlike_pairs import (
    BeamlikePairConfig,
    compare_beamlike_pair_csvs,
    compare_beamlike_pair_rows,
    read_particle_summary_row,
    write_rows_csv,
)


class BeamlikePairComparisonTests(unittest.TestCase):
    def good_row(self, *, score: float, charge: float = 1000.0) -> dict[str, object]:
        return {
            "iteration": 4000,
            "eligible_beamlike": True,
            "beamlike_status": "eligible_beamlike",
            "beamlike_score": score,
            "beam_yield_score": 1.0e5,
            "charge_hot_pC": charge,
            "n_macroparticles_hot": 1000,
            "E95_hot_MeV": 150.0,
            "Emean_hot_MeV": 70.0,
            "Emax_hot_MeV": 300.0,
            "mono_proxy_E95_over_Emax": 0.5,
            "z_span_hot_mm": 0.4,
        }

    def test_channel_better_than_uniform_gets_positive_gain(self) -> None:
        row = compare_beamlike_pair_rows(
            channel=self.good_row(score=500.0),
            uniform=self.good_row(score=100.0),
            channel_case_id="chan",
            uniform_case_id="uni",
        )

        self.assertEqual(row["status"], "ok")
        self.assertGreater(row["beamlike_reference_factor"], 0.0)
        self.assertGreater(row["beamlike_gain_score"], 0.0)
        self.assertEqual(row["beamlike_score_delta"], 400.0)
        self.assertGreater(row["beamlike_score_ratio"], 1.0)

    def test_channel_worse_than_uniform_gets_negative_gain(self) -> None:
        row = compare_beamlike_pair_rows(
            channel=self.good_row(score=100.0),
            uniform=self.good_row(score=500.0),
        )

        self.assertLess(row["beamlike_reference_factor"], 0.0)
        self.assertLess(row["beamlike_gain_score"], 0.0)
        self.assertLess(row["beamlike_score_delta"], 0.0)

    def test_small_difference_inside_deadband_gets_zero_reference_factor(self) -> None:
        row = compare_beamlike_pair_rows(
            channel=self.good_row(score=102.0),
            uniform=self.good_row(score=100.0),
            config=BeamlikePairConfig(reference_deadband_log=math.log(1.05)),
        )

        self.assertAlmostEqual(row["beamlike_reference_factor"], 0.0)
        self.assertAlmostEqual(row["beamlike_gain_score"], 0.0)

    def test_single_row_selection_fails_on_multiple_rows(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "particle_summary.csv"
            write_rows_csv(
                path,
                [self.good_row(score=1.0), self.good_row(score=2.0)],
            )

            with self.assertRaises(ValueError):
                read_particle_summary_row(path, row_selection="single")

    def test_max_beamlike_selection_picks_best_row(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "particle_summary.csv"
            write_rows_csv(
                path,
                [self.good_row(score=1.0), self.good_row(score=2.0)],
            )

            row = read_particle_summary_row(path, row_selection="max-beamlike")

            self.assertEqual(float(row["beamlike_score"]), 2.0)

    def test_compare_csvs_reports_missing_beamlike_columns(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            channel = Path(tmp) / "channel.csv"
            uniform = Path(tmp) / "uniform.csv"

            write_rows_csv(channel, [{"charge_hot_pC": 1.0}])
            write_rows_csv(uniform, [{"charge_hot_pC": 1.0}])

            row = compare_beamlike_pair_csvs(
                channel_csv=channel,
                uniform_csv=uniform,
            )

            self.assertEqual(row["status"], "failed")
            self.assertIn("missing_columns", row["failure_reason"])

    def test_compare_csvs_computes_beamlike_on_the_fly_for_old_summaries(self) -> None:
        old_channel_row = {
            "iteration": 4000,
            "charge_hot_pC": 500.0,
            "n_macroparticles_hot": 800,
            "E95_hot_MeV": 120.0,
            "Emean_hot_MeV": 60.0,
            "Emax_hot_MeV": 220.0,
            "q_long_min_hot_mm": 9.8,
            "q_long_max_hot_mm": 10.2,
        }
        old_uniform_row = {
            "iteration": 4000,
            "charge_hot_pC": 200.0,
            "n_macroparticles_hot": 500,
            "E95_hot_MeV": 80.0,
            "Emean_hot_MeV": 40.0,
            "Emax_hot_MeV": 180.0,
            "q_long_min_hot_mm": 9.8,
            "q_long_max_hot_mm": 10.2,
        }

        with tempfile.TemporaryDirectory() as tmp:
            channel = Path(tmp) / "channel.csv"
            uniform = Path(tmp) / "uniform.csv"

            write_rows_csv(channel, [old_channel_row])
            write_rows_csv(uniform, [old_uniform_row])

            row = compare_beamlike_pair_csvs(
                channel_csv=channel,
                uniform_csv=uniform,
            )

            self.assertEqual(row["status"], "ok")
            self.assertEqual(
                row["beamlike_score_source_channel"],
                "computed_on_the_fly",
            )
            self.assertEqual(
                row["beamlike_score_source_uniform"],
                "computed_on_the_fly",
            )
            self.assertGreater(row["beamlike_score_channel"], 0.0)
            self.assertGreater(row["beamlike_score_uniform"], 0.0)

    def test_channel_zero_against_good_uniform_gets_negative_gain(self) -> None:
        row = compare_beamlike_pair_rows(
            channel=self.good_row(score=0.0, charge=0.0),
            uniform=self.good_row(score=359.0, charge=1160.0),
        )

        self.assertLess(row["beamlike_reference_factor"], 0.0)
        self.assertLess(row["beamlike_gain_score"], -100.0)
        self.assertAlmostEqual(row["beamlike_reference_scale_score"], 359.0)


if __name__ == "__main__":
    unittest.main()

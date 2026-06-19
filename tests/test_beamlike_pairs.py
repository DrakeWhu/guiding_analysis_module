from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

from cap_guiding.beamlike_pairs import (
    BeamlikePairConfig,
    classify_pair_row,
    classify_transverse_pair_row,
    compare_beamlike_pair_csvs,
    compare_beamlike_pair_rows,
    read_particle_summary_row,
    split_pair_rows,
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

    def test_pair_classification_positive_neutral_negative_failed(self) -> None:
        positive = compare_beamlike_pair_rows(
            channel=self.good_row(score=500.0),
            uniform=self.good_row(score=100.0),
        )
        neutral = compare_beamlike_pair_rows(
            channel=self.good_row(score=102.0),
            uniform=self.good_row(score=100.0),
            config=BeamlikePairConfig(reference_deadband_log=math.log(1.05)),
        )
        negative = compare_beamlike_pair_rows(
            channel=self.good_row(score=100.0),
            uniform=self.good_row(score=500.0),
        )
        failed = {"status": "failed", "failure_reason": "missing_csv"}

        self.assertEqual(classify_pair_row(positive), "positive")
        self.assertEqual(classify_pair_row(neutral), "neutral")
        self.assertEqual(classify_pair_row(negative), "negative")
        self.assertEqual(classify_pair_row(failed), "failed")

        self.assertEqual(positive["comparison_bucket"], "positive")
        self.assertEqual(neutral["comparison_bucket"], "neutral")
        self.assertEqual(negative["comparison_bucket"], "negative")

    def test_split_pair_rows_adds_rows_to_expected_buckets(self) -> None:
        rows = [
            compare_beamlike_pair_rows(
                channel=self.good_row(score=500.0),
                uniform=self.good_row(score=100.0),
            ),
            compare_beamlike_pair_rows(
                channel=self.good_row(score=100.0),
                uniform=self.good_row(score=100.0),
            ),
            compare_beamlike_pair_rows(
                channel=self.good_row(score=100.0),
                uniform=self.good_row(score=500.0),
            ),
            {"status": "failed", "failure_reason": "missing_csv"},
        ]

        buckets = split_pair_rows(rows)

        self.assertEqual(len(buckets["positive"]), 1)
        self.assertEqual(len(buckets["neutral"]), 1)
        self.assertEqual(len(buckets["negative"]), 1)
        self.assertEqual(len(buckets["failed"]), 1)

        for bucket_name, bucket_rows in buckets.items():
            for row in bucket_rows:
                self.assertEqual(row["comparison_bucket"], bucket_name)

    def transverse_row(
        self,
        *,
        score: float,
        theta_rms: float,
        theta_p95: float = 10.0,
        emit_x: float = 1.0,
        emit_y: float = 1.2,
    ) -> dict[str, object]:
        row = self.good_row(score=100.0)
        row.update(
            {
                "transverse_status": "ok",
                "n_macroparticles_transverse": 900,
                "weight_transverse": 1.5e9,
                "theta_x_rms_mrad": theta_rms / math.sqrt(2.0),
                "theta_y_rms_mrad": theta_rms / math.sqrt(2.0),
                "theta_rms_mrad": theta_rms,
                "theta_x_p95_mrad": theta_p95 / math.sqrt(2.0),
                "theta_y_p95_mrad": theta_p95 / math.sqrt(2.0),
                "theta_r_p95_mrad": theta_p95,
                "x_rms_um": 2.0,
                "y_rms_um": 3.0,
                "x_p95_um": 5.0,
                "y_p95_um": 6.0,
                "emit_x_norm_mm_mrad": emit_x,
                "emit_y_norm_mm_mrad": emit_y,
                "emit_geom_norm_mm_mrad": math.sqrt(emit_x * emit_y),
                "transverse_theta_rms_component": 0.5,
                "transverse_theta_p95_component": 0.6,
                "transverse_emit_component": 0.7,
                "beam_transverse_quality_score": score,
            }
        )
        return row

    def test_transverse_quality_channel_better_gets_positive_gain(self) -> None:
        row = compare_beamlike_pair_rows(
            channel=self.transverse_row(score=500.0, theta_rms=2.0),
            uniform=self.transverse_row(score=100.0, theta_rms=8.0),
        )

        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["comparison_bucket"], "neutral")
        self.assertEqual(row["transverse_comparison_status"], "ok")
        self.assertEqual(row["transverse_comparison_bucket"], "positive")
        self.assertEqual(classify_transverse_pair_row(row), "positive")
        self.assertGreater(row["transverse_reference_factor"], 0.0)
        self.assertGreater(row["transverse_gain_score"], 0.0)

    def test_transverse_quality_channel_worse_gets_negative_gain(self) -> None:
        row = compare_beamlike_pair_rows(
            channel=self.transverse_row(score=100.0, theta_rms=8.0),
            uniform=self.transverse_row(score=500.0, theta_rms=2.0),
        )

        self.assertEqual(row["transverse_comparison_bucket"], "negative")
        self.assertLess(row["transverse_reference_factor"], 0.0)
        self.assertLess(row["transverse_gain_score"], 0.0)

    def test_transverse_metrics_are_copied_and_lower_is_better_improves(self) -> None:
        row = compare_beamlike_pair_rows(
            channel=self.transverse_row(
                score=500.0,
                theta_rms=2.0,
                theta_p95=5.0,
                emit_x=0.5,
                emit_y=0.8,
            ),
            uniform=self.transverse_row(
                score=100.0,
                theta_rms=8.0,
                theta_p95=20.0,
                emit_x=2.5,
                emit_y=3.0,
            ),
        )

        self.assertEqual(row["theta_rms_mrad_channel"], 2.0)
        self.assertEqual(row["theta_rms_mrad_uniform"], 8.0)
        self.assertEqual(row["theta_rms_mrad_delta"], -6.0)
        self.assertEqual(row["theta_rms_mrad_improvement"], 6.0)
        self.assertEqual(row["divergence_improvement_mrad"], 6.0)
        self.assertEqual(row["emit_x_norm_mm_mrad_improvement"], 2.0)
        self.assertGreater(row["beam_transverse_quality_score_ratio"], 1.0)

    def test_missing_transverse_metrics_do_not_fail_beamlike_pair(self) -> None:
        row = compare_beamlike_pair_rows(
            channel=self.good_row(score=500.0),
            uniform=self.good_row(score=100.0),
        )

        self.assertEqual(row["status"], "ok")
        self.assertEqual(row["comparison_bucket"], "positive")
        self.assertEqual(row["transverse_comparison_status"], "failed")
        self.assertEqual(row["transverse_comparison_bucket"], "failed")
        self.assertTrue(math.isnan(row["transverse_gain_score"]))

    def test_transverse_columns_are_written_to_pair_csv(self) -> None:
        row = compare_beamlike_pair_rows(
            channel=self.transverse_row(score=500.0, theta_rms=2.0),
            uniform=self.transverse_row(score=100.0, theta_rms=8.0),
        )

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "beamlike_pair_scores.csv"
            write_rows_csv(path, [row])
            header = path.read_text(encoding="utf-8").splitlines()[0].split(",")

        self.assertIn("transverse_comparison_bucket", header)
        self.assertIn("transverse_gain_score", header)
        self.assertIn("theta_r_p95_mrad_channel", header)
        self.assertIn("emit_x_norm_mm_mrad_improvement", header)


if __name__ == "__main__":
    unittest.main()

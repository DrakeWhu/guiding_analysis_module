from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cap_guiding.singlecase_guiding import (
    SINGLECASE_GUIDING_SCORE_FILENAME,
    ensure_singlecase_guiding_score_csv,
    score_singlecase_guiding_dataframe,
)


def _guiding_df(
    *,
    a0: list[float],
    waist: list[float],
    z: list[float] | None = None,
) -> pd.DataFrame:
    if z is None:
        z = [5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    return pd.DataFrame(
        {
            "iteration": list(range(len(z))),
            "time_fs": [float(i) for i in range(len(z))],
            "propagation_mm": z,
            "a0_peak": a0,
            "waist_um": waist,
        }
    )


class SingleCaseGuidingMetricTests(unittest.TestCase):
    def score(self, df: pd.DataFrame) -> dict[str, object]:
        return score_singlecase_guiding_dataframe(
            df,
            case_id="case",
            plateau_start_mm=5.0,
            plateau_end_mm=10.0,
        )

    def test_constant_waist_and_non_decreasing_a0_scores_near_100(self) -> None:
        df = _guiding_df(
            a0=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
            waist=[20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        )

        row = self.score(df)

        self.assertEqual(row["metric_guiding_singlecase_status"], "ok")
        self.assertAlmostEqual(
            float(row["metric_guiding_singlecase_score_v1"]),
            100.0,
            places=12,
        )
        self.assertAlmostEqual(
            float(row["metric_guiding_waist_growth_component_v1"]),
            1.0,
            places=12,
        )
        self.assertAlmostEqual(
            float(row["metric_guiding_a0_retention_component_v1"]),
            1.0,
            places=12,
        )

    def test_waist_growth_is_worse_than_moderate_matching_oscillation(self) -> None:
        oscillatory = _guiding_df(
            a0=[1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            waist=[20.0, 22.0, 18.0, 21.0, 19.0, 20.0],
        )
        growing = _guiding_df(
            a0=[1.5, 1.5, 1.5, 1.5, 1.5, 1.5],
            waist=[20.0, 25.0, 30.0, 35.0, 40.0, 45.0],
        )

        osc_row = self.score(oscillatory)
        grow_row = self.score(growing)

        self.assertEqual(osc_row["metric_guiding_singlecase_status"], "ok")
        self.assertEqual(grow_row["metric_guiding_singlecase_status"], "ok")
        self.assertGreater(
            float(osc_row["metric_guiding_singlecase_score_v1"]),
            float(grow_row["metric_guiding_singlecase_score_v1"]),
        )
        self.assertGreater(
            float(osc_row["metric_guiding_singlecase_score_v1"]),
            90.0,
        )
        self.assertLess(
            float(grow_row["metric_guiding_waist_growth_component_v1"]),
            0.2,
        )

    def test_strong_a0_degradation_is_penalized(self) -> None:
        stable = _guiding_df(
            a0=[2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            waist=[20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        )
        degraded = _guiding_df(
            a0=[2.0, 2.0, 1.4, 1.0, 0.6, 0.3],
            waist=[20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        )

        stable_row = self.score(stable)
        degraded_row = self.score(degraded)

        self.assertEqual(degraded_row["metric_guiding_singlecase_status"], "ok")
        self.assertGreater(
            float(stable_row["metric_guiding_singlecase_score_v1"]),
            float(degraded_row["metric_guiding_singlecase_score_v1"]),
        )
        self.assertLess(
            float(degraded_row["metric_guiding_a0_retention_component_v1"]),
            0.25,
        )
        self.assertLess(
            float(degraded_row["metric_guiding_a0_stability_component_v1"]),
            0.5,
        )

    def test_not_enough_valid_plateau_rows_fails(self) -> None:
        df = _guiding_df(
            a0=[1.0, float("nan"), 1.0, float("nan"), float("nan"), float("nan")],
            waist=[20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
        )

        row = self.score(df)

        self.assertEqual(row["metric_guiding_singlecase_status"], "failed")
        self.assertEqual(
            row["metric_guiding_singlecase_failure_reason"],
            "not_enough_valid_plateau_rows",
        )

    def test_sidecar_is_written_from_existing_guiding_metrics_csv(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "001_f20_chan_n4e18cm3_L5mm_d250um_foc0mm_rz"
            case_dir.mkdir(parents=True)
            csv_path = case_dir / "guiding_metrics.csv"
            _guiding_df(
                a0=[1.0, 1.1, 1.2, 1.3, 1.4, 1.5],
                waist=[20.0, 20.0, 20.0, 20.0, 20.0, 20.0],
            ).to_csv(csv_path, index=False)

            score_path = ensure_singlecase_guiding_score_csv(csv_path)

            self.assertEqual(score_path.name, SINGLECASE_GUIDING_SCORE_FILENAME)
            self.assertTrue(score_path.is_file())
            written = pd.read_csv(score_path)
            self.assertEqual(len(written), 1)
            self.assertEqual(
                written.iloc[0]["metric_guiding_singlecase_status"],
                "ok",
            )
            self.assertAlmostEqual(
                float(written.iloc[0]["metric_guiding_singlecase_score_v1"]),
                100.0,
                places=12,
            )


if __name__ == "__main__":
    unittest.main()

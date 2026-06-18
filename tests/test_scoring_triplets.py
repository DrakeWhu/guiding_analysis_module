from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cap_guiding.scoring import score_triplet_csvs


class TripletScoringRobustnessTests(unittest.TestCase):
    def write_case(
        self,
        path: Path,
        *,
        iterations: list[int],
        a0: float,
        waist: float,
    ) -> None:
        # L5mm => plateau window [5, 10] mm.
        # Default exit window is [9, 12] mm.
        rows = []
        for iteration, z in zip(iterations, [5.0, 6.0, 9.5, 10.5, 11.5]):
            rows.append(
                {
                    "iteration": iteration,
                    "propagation_mm": z,
                    "a0_peak": a0,
                    "waist_um": waist,
                }
            )

        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_triplet_score_falls_back_when_exit_iterations_do_not_overlap(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)

            channel = (
                root
                / "001_f20_chan_n6e18cm3_L5mm_d500um_foc0mm_rz"
                / "guiding_metrics.csv"
            )
            uniform = (
                root
                / "262_f20_uni_n6e18cm3_L5mm_refd500um_foc0mm_rz"
                / "guiding_metrics.csv"
            )
            vacuum = root / "vac_f20_vac_L5mm_foc0mm_rz" / "guiding_metrics.csv"

            self.write_case(channel, iterations=[1, 2, 3, 4, 5], a0=2.0, waist=20.0)
            self.write_case(
                uniform, iterations=[11, 12, 13, 14, 15], a0=1.5, waist=25.0
            )
            self.write_case(vacuum, iterations=[21, 22, 23, 24, 25], a0=1.0, waist=30.0)

            row = score_triplet_csvs(
                channel_csv=channel,
                uniform_csv=uniform,
                vacuum_csv=vacuum,
                channel_case_id="chan",
                uniform_case_id="uni",
                vacuum_case_id="vac",
            )

            self.assertEqual(row["status"], "ok")
            self.assertEqual(row["reference_factor_mode"], "exit_median_fallback")
            self.assertIn(
                "no common iterations",
                row["rowwise_reference_failure_reason"],
            )
            self.assertEqual(row["n_reference_rows"], 0)
            self.assertGreater(row["final_score"], 0.0)


if __name__ == "__main__":
    unittest.main()

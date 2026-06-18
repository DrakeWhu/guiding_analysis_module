from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cap_guiding.joint_scores import (
    bucket_counts,
    compute_joint_correlations,
    load_and_join_guiding_beamlike,
    write_joint_outputs,
)


class JointGuidingBeamlikeTests(unittest.TestCase):
    def write_csv(self, path: Path, rows: list[dict[str, object]]) -> None:
        pd.DataFrame(rows).to_csv(path, index=False)

    def test_join_classifies_both_positive(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            guiding = root / "triplet_scores.csv"
            beam = root / "beamlike_pair_scores.csv"

            self.write_csv(
                guiding,
                [
                    {
                        "status": "ok",
                        "channel_case_id": "chan",
                        "uniform_case_id": "uni",
                        "vacuum_case_id": "vac",
                        "final_score": 20.0,
                        "reference_factor": 0.5,
                        "score_channel": 40.0,
                        "score_uniform": 10.0,
                        "score_vacuum": 5.0,
                    }
                ],
            )
            self.write_csv(
                beam,
                [
                    {
                        "status": "ok",
                        "comparison_bucket": "positive",
                        "channel_case_id": "chan",
                        "uniform_case_id": "uni",
                        "beamlike_gain_score": 9.0,
                        "beamlike_reference_factor": 0.2,
                        "beamlike_score_channel": 30.0,
                        "beamlike_score_uniform": 20.0,
                    }
                ],
            )

            joined = load_and_join_guiding_beamlike(
                triplet_scores_csv=guiding,
                beamlike_pair_scores_csv=beam,
            )

            self.assertEqual(len(joined), 1)
            row = joined.iloc[0]
            self.assertEqual(row["guiding_bucket"], "positive")
            self.assertEqual(row["beam_bucket"], "positive")
            self.assertEqual(row["joint_bucket"], "guiding_positive__beam_positive")
            self.assertGreater(row["joint_positive_score"], 0.0)

    def test_join_detects_guiding_positive_beam_negative(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            guiding = root / "triplet_scores.csv"
            beam = root / "beamlike_pair_scores.csv"

            self.write_csv(
                guiding,
                [
                    {
                        "status": "ok",
                        "channel_case_id": "chan",
                        "uniform_case_id": "uni",
                        "final_score": 20.0,
                        "reference_factor": 0.5,
                    }
                ],
            )
            self.write_csv(
                beam,
                [
                    {
                        "status": "ok",
                        "comparison_bucket": "negative",
                        "channel_case_id": "chan",
                        "uniform_case_id": "uni",
                        "beamlike_gain_score": -10.0,
                        "beamlike_reference_factor": -0.8,
                    }
                ],
            )

            joined = load_and_join_guiding_beamlike(
                triplet_scores_csv=guiding,
                beamlike_pair_scores_csv=beam,
            )

            row = joined.iloc[0]
            self.assertEqual(row["joint_bucket"], "guiding_positive__beam_negative")
            self.assertLess(row["guiding_beam_alignment_factor"], 0.0)
            self.assertEqual(row["joint_positive_score"], 0.0)

    def test_bucket_counts_and_outputs_are_written(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            guiding = root / "triplet_scores.csv"
            beam = root / "beamlike_pair_scores.csv"
            outdir = root / "out"

            self.write_csv(
                guiding,
                [
                    {
                        "status": "ok",
                        "channel_case_id": "chan",
                        "uniform_case_id": "uni",
                        "final_score": 20.0,
                        "reference_factor": 0.5,
                    }
                ],
            )
            self.write_csv(
                beam,
                [
                    {
                        "status": "ok",
                        "comparison_bucket": "positive",
                        "channel_case_id": "chan",
                        "uniform_case_id": "uni",
                        "beamlike_gain_score": 9.0,
                        "beamlike_reference_factor": 0.2,
                    }
                ],
            )

            joined = load_and_join_guiding_beamlike(
                triplet_scores_csv=guiding,
                beamlike_pair_scores_csv=beam,
            )
            counts = bucket_counts(joined)
            paths = write_joint_outputs(outdir=outdir, joined=joined, top=10)

            self.assertEqual(int(counts["count"].sum()), 1)
            for path in paths.values():
                self.assertTrue(path.is_file())

    def test_correlations_returns_dataframe(self) -> None:
        df = pd.DataFrame(
            [
                {
                    "guiding_final_score": 1.0,
                    "beam_beamlike_gain_score": 2.0,
                    "guiding_reference_factor": 0.1,
                    "beam_beamlike_reference_factor": 0.2,
                },
                {
                    "guiding_final_score": 2.0,
                    "beam_beamlike_gain_score": 4.0,
                    "guiding_reference_factor": 0.2,
                    "beam_beamlike_reference_factor": 0.4,
                },
                {
                    "guiding_final_score": 3.0,
                    "beam_beamlike_gain_score": 6.0,
                    "guiding_reference_factor": 0.3,
                    "beam_beamlike_reference_factor": 0.6,
                },
            ]
        )

        corr = compute_joint_correlations(df)

        self.assertGreaterEqual(len(corr), 1)
        self.assertIn("pearson", corr.columns)
        self.assertIn("spearman", corr.columns)


if __name__ == "__main__":
    unittest.main()

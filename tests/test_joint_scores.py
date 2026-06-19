from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from cap_guiding.joint_scores import (
    bucket_counts,
    compute_joint_correlations,
    load_and_join_guiding_beamlike,
    triple_bucket_counts,
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

    def test_join_classifies_triple_positive_when_transverse_is_positive(self) -> None:
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
                        "final_score": 27.0,
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
                        "transverse_comparison_status": "ok",
                        "transverse_comparison_bucket": "positive",
                        "channel_case_id": "chan",
                        "uniform_case_id": "uni",
                        "beamlike_gain_score": 8.0,
                        "beamlike_reference_factor": 0.2,
                        "transverse_gain_score": 64.0,
                        "transverse_reference_factor": 0.7,
                    }
                ],
            )

            joined = load_and_join_guiding_beamlike(
                triplet_scores_csv=guiding,
                beamlike_pair_scores_csv=beam,
            )

            row = joined.iloc[0]
            self.assertEqual(row["joint_bucket"], "guiding_positive__beam_positive")
            self.assertEqual(row["transverse_bucket"], "positive")
            self.assertEqual(
                row["triple_bucket"],
                "guiding_positive__beam_positive__transverse_positive",
            )
            self.assertEqual(row["triple_status"], "ok")
            self.assertGreater(row["guiding_beam_transverse_alignment_factor"], 0.0)
            self.assertAlmostEqual(row["triple_positive_score"], 24.0)

    def test_triple_bucket_detects_good_acceleration_but_bad_transverse_quality(
        self,
    ) -> None:
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
                        "comparison_bucket": "positive",
                        "transverse_comparison_status": "ok",
                        "transverse_comparison_bucket": "negative",
                        "channel_case_id": "chan",
                        "uniform_case_id": "uni",
                        "beamlike_gain_score": 9.0,
                        "beamlike_reference_factor": 0.3,
                        "transverse_gain_score": -5.0,
                        "transverse_reference_factor": -0.4,
                    }
                ],
            )

            joined = load_and_join_guiding_beamlike(
                triplet_scores_csv=guiding,
                beamlike_pair_scores_csv=beam,
            )

            row = joined.iloc[0]
            self.assertEqual(row["joint_bucket"], "guiding_positive__beam_positive")
            self.assertEqual(row["transverse_bucket"], "negative")
            self.assertEqual(
                row["triple_bucket"],
                "guiding_positive__beam_positive__transverse_negative",
            )
            self.assertEqual(row["triple_positive_score"], 0.0)

    def test_missing_transverse_comparison_preserves_legacy_joint_bucket(self) -> None:
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

            row = joined.iloc[0]
            self.assertEqual(row["joint_bucket"], "guiding_positive__beam_positive")
            self.assertEqual(row["transverse_bucket"], "failed")
            self.assertEqual(row["triple_bucket"], "failed")
            self.assertEqual(row["triple_status"], "failed")
            self.assertEqual(row["joint_status"], "ok")

    def test_triple_bucket_counts_and_outputs_are_written(self) -> None:
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
                        "final_score": 27.0,
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
                        "transverse_comparison_status": "ok",
                        "transverse_comparison_bucket": "positive",
                        "channel_case_id": "chan",
                        "uniform_case_id": "uni",
                        "beamlike_gain_score": 8.0,
                        "beamlike_reference_factor": 0.2,
                        "transverse_gain_score": 64.0,
                        "transverse_reference_factor": 0.7,
                    }
                ],
            )

            joined = load_and_join_guiding_beamlike(
                triplet_scores_csv=guiding,
                beamlike_pair_scores_csv=beam,
            )
            counts = triple_bucket_counts(joined)
            paths = write_joint_outputs(outdir=outdir, joined=joined, top=10)

            self.assertEqual(int(counts["count"].sum()), 1)
            self.assertIn("joined_triple", paths)
            self.assertIn("triple_bucket_counts", paths)
            self.assertIn("triple_positive", paths)
            for path in paths.values():
                self.assertTrue(path.is_file())

            triple_positive = pd.read_csv(paths["triple_positive"])
            self.assertEqual(len(triple_positive), 1)
            self.assertIn("triple_bucket", triple_positive.columns)


if __name__ == "__main__":
    unittest.main()

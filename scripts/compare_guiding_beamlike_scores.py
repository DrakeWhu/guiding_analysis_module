#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cap_guiding.joint_scores import (
    bucket_counts,
    compute_joint_correlations,
    load_and_join_guiding_beamlike,
    write_joint_outputs,
)


def _latest_analysis_file(
    *,
    campaign_root: Path,
    dirname_glob: str,
    filename: str,
) -> Path:
    analysis_root = campaign_root / "analysis_outputs"
    candidates = []

    if analysis_root.is_dir():
        for directory in analysis_root.glob(dirname_glob):
            path = directory / filename
            if path.is_file():
                candidates.append(path)

    if not candidates:
        raise FileNotFoundError(
            f"No {filename} found under {analysis_root}/{dirname_glob}"
        )

    return max(candidates, key=lambda path: path.stat().st_mtime)


def _print_rows(rows, cols: list[str]) -> None:
    for _, row in rows.iterrows():
        parts = []
        for col in cols:
            if col in row:
                parts.append(f"{col}={row[col]}")
        print("  " + "  ".join(parts))


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Join guiding triplet scores with beamlike channel-vs-uniform scores. "
            "This consumes reduced CSVs only."
        )
    )

    parser.add_argument(
        "--campaign-root",
        default=None,
        help=(
            "Used to auto-locate latest analysis_outputs/triplet_scores_* and "
            "analysis_outputs/beamlike_pairs_* when explicit CSV paths are omitted."
        ),
    )
    parser.add_argument("--triplet-scores-csv", default=None)
    parser.add_argument("--beamlike-pair-scores-csv", default=None)
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument(
        "--join-how",
        choices=["inner", "left", "right", "outer"],
        default="inner",
        help="Default inner compares only pairs present in both CSVs.",
    )

    args = parser.parse_args()

    if args.campaign_root is None and (
        args.triplet_scores_csv is None or args.beamlike_pair_scores_csv is None
    ):
        raise SystemExit(
            "Provide --campaign-root or both --triplet-scores-csv and "
            "--beamlike-pair-scores-csv."
        )

    campaign_root = Path(args.campaign_root) if args.campaign_root else None

    if args.triplet_scores_csv is not None:
        triplet_scores_csv = Path(args.triplet_scores_csv)
    else:
        assert campaign_root is not None
        triplet_scores_csv = _latest_analysis_file(
            campaign_root=campaign_root,
            dirname_glob="triplet_scores_*",
            filename="triplet_scores.csv",
        )

    if args.beamlike_pair_scores_csv is not None:
        beamlike_pair_scores_csv = Path(args.beamlike_pair_scores_csv)
    else:
        assert campaign_root is not None
        beamlike_pair_scores_csv = _latest_analysis_file(
            campaign_root=campaign_root,
            dirname_glob="beamlike_pairs_*",
            filename="beamlike_pair_scores.csv",
        )

    if args.outdir is not None:
        outdir = Path(args.outdir)
    elif campaign_root is not None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = campaign_root / "analysis_outputs" / f"guiding_beamlike_joint_{stamp}"
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = Path.cwd() / f"guiding_beamlike_joint_{stamp}"

    joined = load_and_join_guiding_beamlike(
        triplet_scores_csv=triplet_scores_csv,
        beamlike_pair_scores_csv=beamlike_pair_scores_csv,
        join_how=args.join_how,
    )

    paths = write_joint_outputs(outdir=outdir, joined=joined, top=args.top)
    counts = bucket_counts(joined)
    correlations = compute_joint_correlations(joined)

    print("=== Guiding + beamlike joint summary ===")
    print(f"triplet_scores_csv        = {triplet_scores_csv}")
    print(f"beamlike_pair_scores_csv  = {beamlike_pair_scores_csv}")
    print(f"join_how                  = {args.join_how}")
    print(f"outdir                    = {outdir}")
    print(f"joined pairs              = {len(joined)}")
    print()

    if len(counts):
        print("Joint buckets:")
        print(counts.to_string(index=False))
    else:
        print("No joint buckets.")

    print()
    print("Outputs:")
    for name, path in paths.items():
        print(f"{name:36s} = {path}")

    print()

    if len(correlations):
        print("Correlations:")
        visible = correlations[["description", "n", "pearson", "spearman"]].copy()
        print(visible.to_string(index=False))
    else:
        print("No correlations could be computed.")

    print()

    visible_cols = [
        "channel_case_id",
        "uniform_case_id",
        "guiding_final_score",
        "guiding_reference_factor",
        "beam_beamlike_gain_score",
        "beam_beamlike_reference_factor",
        "joint_positive_score",
        "guiding_a0_exit_channel",
        "beam_E95_hot_MeV_channel",
        "beam_charge_hot_pC_channel",
        "guiding_waist_growth_channel",
        "beam_z_span_hot_mm_channel",
        "plateau",
        "diameter",
        "focus",
    ]

    if "joint_bucket" in joined.columns:
        both_positive = joined[
            joined["joint_bucket"] == "guiding_positive__beam_positive"
        ].copy()
        both_positive = both_positive.sort_values(
            "joint_positive_score",
            ascending=False,
        ).head(args.top)

        if len(both_positive):
            print(f"Top {len(both_positive)} both-positive cases:")
            _print_rows(both_positive, visible_cols)
        else:
            print("No both-positive cases.")

        print()

        contradictions = joined[
            joined["joint_bucket"].isin(
                [
                    "guiding_positive__beam_negative",
                    "guiding_negative__beam_positive",
                ]
            )
        ].copy()

        if len(contradictions):
            contradictions = contradictions.sort_values(
                "guiding_beam_alignment_factor",
                ascending=True,
            ).head(args.top)
            print(f"Top {len(contradictions)} guiding/beam contradictions:")
            _print_rows(contradictions, visible_cols + ["joint_bucket"])
        else:
            print("No hard guiding/beam contradictions.")


if __name__ == "__main__":
    main()

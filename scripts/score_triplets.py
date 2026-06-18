#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cap_guiding.campaign import build_triplets, discover_cases
from cap_guiding.scoring import (
    ScoreConfig,
    TripletScoreConfig,
    score_triplet_csvs,
)


def _case_metrics_path(case_metrics_root: Path, case_id: str) -> Path:
    return case_metrics_root / case_id / "guiding_metrics.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score channel/uniform/vacuum triplets from reduced CSV metrics."
    )

    parser.add_argument("--campaign-root", required=True)
    parser.add_argument(
        "--case-metrics-root",
        default=None,
        help="Defaults to CAMPAIGN_ROOT",
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Defaults to CAMPAIGN_ROOT/analysis_outputs/triplet_scores_<timestamp>",
    )
    parser.add_argument("--top", type=int, default=50)

    parser.add_argument("--entry-window-mm", type=float, default=1.0)
    parser.add_argument("--exit-before-mm", type=float, default=1.0)
    parser.add_argument("--exit-after-mm", type=float, default=2.0)
    parser.add_argument("--a0-target", type=float, default=1.5)
    parser.add_argument("--a0-component-cap", type=float, default=1.25)
    parser.add_argument("--waist-growth-sigma", type=float, default=0.75)
    parser.add_argument("--waist-jitter-sigma", type=float, default=0.25)

    parser.add_argument("--reference-deadband", type=float, default=5.0)
    parser.add_argument("--reference-scale", type=float, default=15.0)

    args = parser.parse_args()

    campaign_root = Path(args.campaign_root)
    case_metrics_root = (
        Path(args.case_metrics_root)
        if args.case_metrics_root is not None
        else campaign_root
    )

    if args.outdir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = campaign_root / "analysis_outputs" / f"triplet_scores_{stamp}"
    else:
        outdir = Path(args.outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    case_config = ScoreConfig(
        entry_window_mm=args.entry_window_mm,
        exit_before_mm=args.exit_before_mm,
        exit_after_mm=args.exit_after_mm,
        a0_target=args.a0_target,
        a0_component_cap=args.a0_component_cap,
        waist_growth_sigma=args.waist_growth_sigma,
        waist_jitter_sigma=args.waist_jitter_sigma,
    )
    triplet_config = TripletScoreConfig(
        reference_deadband=args.reference_deadband,
        reference_scale=args.reference_scale,
    )

    cases = discover_cases(campaign_root)
    triplets = build_triplets(cases)

    rows = []
    skipped_incomplete = 0

    for triplet in triplets:
        if not triplet.complete:
            skipped_incomplete += 1
            continue

        assert triplet.channel is not None
        assert triplet.uniform is not None
        assert triplet.vacuum is not None

        channel_csv = _case_metrics_path(case_metrics_root, triplet.channel.case_id)
        uniform_csv = _case_metrics_path(case_metrics_root, triplet.uniform.case_id)
        vacuum_csv = _case_metrics_path(case_metrics_root, triplet.vacuum.case_id)

        try:
            row = score_triplet_csvs(
                channel_csv=channel_csv,
                uniform_csv=uniform_csv,
                vacuum_csv=vacuum_csv,
                channel_case_id=triplet.channel.case_id,
                uniform_case_id=triplet.uniform.case_id,
                vacuum_case_id=triplet.vacuum.case_id,
                case_config=case_config,
                triplet_config=triplet_config,
            )
        except Exception as exc:
            row = {
                "status": "failed",
                "failure_reason": f"score_triplet_exception: {exc}",
                "channel_case_id": triplet.channel.case_id,
                "uniform_case_id": triplet.uniform.case_id,
                "vacuum_case_id": triplet.vacuum.case_id,
                "score_channel": float("nan"),
                "score_uniform": float("nan"),
                "score_vacuum": float("nan"),
                "final_score": float("nan"),
            }

        row.update(
            {
                "triplet_label": triplet.label,
                "laser_case": triplet.channel.laser_case,
                "density": triplet.channel.density,
                "plateau": triplet.channel.plateau,
                "focus": triplet.channel.focus,
                "diameter": triplet.channel.diameter,
                "channel_csv": str(channel_csv),
                "uniform_csv": str(uniform_csv),
                "vacuum_csv": str(vacuum_csv),
            }
        )

        rows.append(row)

    df = pd.DataFrame(rows)
    scores_path = outdir / "triplet_scores.csv"
    df.to_csv(scores_path, index=False)

    ok = df[df["status"] == "ok"].copy()
    ok = ok.sort_values("final_score", ascending=False)
    top = ok.head(args.top).copy()
    top.insert(0, "rank", range(1, len(top) + 1))

    top_path = outdir / "top_triplets.csv"
    top.to_csv(top_path, index=False)

    print("=== Triplet score summary ===")
    print(f"campaign_root      = {campaign_root}")
    print(f"case_metrics_root  = {case_metrics_root}")
    print(f"outdir             = {outdir}")
    print(f"triplets total     = {len(triplets)}")
    print(f"triplets scored    = {len(df)}")
    print(f"triplets ok        = {len(ok)}")
    print(f"triplets failed    = {len(df) - len(ok)}")
    print(f"incomplete skipped = {skipped_incomplete}")
    print(f"triplet_scores     = {scores_path}")
    print(f"top_triplets       = {top_path}")
    print()

    if len(top) == 0:
        print("No valid triplet scores.")
        if len(df):
            print()
            print("Failure reasons:")
            print(df["failure_reason"].value_counts(dropna=False).to_string())
        return

    cols = [
        "rank",
        "final_score",
        "reference_factor",
        "score_channel",
        "score_uniform",
        "score_vacuum",
        "reference_kind",
        "score_delta_vs_reference",
        "channel_case_id",
        "uniform_case_id",
        "vacuum_case_id",
        "a0_exit_channel",
        "a0_exit_uniform",
        "a0_exit_vacuum",
        "waist_growth_channel",
        "waist_growth_uniform",
        "waist_growth_vacuum",
        "plateau",
        "diameter",
        "focus",
    ]
    visible_cols = [col for col in cols if col in top.columns]

    print(f"Top {len(top)} triplets:")
    print(top[visible_cols].to_string(index=False))


if __name__ == "__main__":
    main()

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

from cap_guiding.campaign import discover_cases
from cap_guiding.scoring import ScoreConfig, score_case_csv


def _case_metrics_path(case_metrics_root: Path, case_id: str) -> Path:
    return case_metrics_root / case_id / "guiding_metrics.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Score reduced capillary guiding cases from guiding_metrics.csv."
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
        help="Defaults to CAMPAIGN_ROOT/analysis_outputs/scored_<timestamp>",
    )
    parser.add_argument(
        "--case-type",
        default="channel",
        choices=["channel", "uniform", "vacuum", "all"],
        help="Cases to score. Default: channel.",
    )
    parser.add_argument("--top", type=int, default=30)

    parser.add_argument("--entry-window-mm", type=float, default=1.0)
    parser.add_argument("--exit-before-mm", type=float, default=1.0)
    parser.add_argument("--exit-after-mm", type=float, default=2.0)
    parser.add_argument("--a0-target", type=float, default=1.5)
    parser.add_argument("--a0-component-cap", type=float, default=1.25)
    parser.add_argument("--waist-growth-sigma", type=float, default=0.75)
    parser.add_argument("--waist-jitter-sigma", type=float, default=0.25)

    args = parser.parse_args()

    campaign_root = Path(args.campaign_root)
    case_metrics_root = (
        Path(args.case_metrics_root)
        if args.case_metrics_root is not None
        else campaign_root
    )

    if args.outdir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = campaign_root / "analysis_outputs" / f"scored_{stamp}"
    else:
        outdir = Path(args.outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    config = ScoreConfig(
        entry_window_mm=args.entry_window_mm,
        exit_before_mm=args.exit_before_mm,
        exit_after_mm=args.exit_after_mm,
        a0_target=args.a0_target,
        a0_component_cap=args.a0_component_cap,
        waist_growth_sigma=args.waist_growth_sigma,
        waist_jitter_sigma=args.waist_jitter_sigma,
    )

    cases = discover_cases(campaign_root)
    if args.case_type != "all":
        cases = [case for case in cases if case.case_type == args.case_type]

    rows = []
    for case in cases:
        csv_path = _case_metrics_path(case_metrics_root, case.case_id)
        row = score_case_csv(csv_path, case_id=case.case_id, config=config)
        row.update(
            {
                "case_type": case.case_type,
                "laser_case": case.laser_case,
                "density": case.density,
                "ref_density": case.ref_density,
                "plateau": case.plateau,
                "focus": case.focus,
                "diameter": case.diameter,
            }
        )
        rows.append(row)

    df = pd.DataFrame(rows)

    scores_path = outdir / "case_scores.csv"
    df.to_csv(scores_path, index=False)

    ok = df[df["status"] == "ok"].copy()
    ok = ok.sort_values("score", ascending=False)
    top = ok.head(args.top).copy()
    top.insert(0, "rank", range(1, len(top) + 1))

    top_path = outdir / "top_cases.csv"
    top.to_csv(top_path, index=False)

    print("=== Guiding score summary ===")
    print(f"campaign_root     = {campaign_root}")
    print(f"case_metrics_root = {case_metrics_root}")
    print(f"outdir            = {outdir}")
    print(f"case_type         = {args.case_type}")
    print(f"cases scored      = {len(df)}")
    print(f"ok scores         = {len(ok)}")
    print(f"failed scores     = {len(df) - len(ok)}")
    print(f"case_scores       = {scores_path}")
    print(f"top_cases         = {top_path}")
    print()

    if len(top) == 0:
        print("No valid scores.")
        return

    print(f"Top {len(top)} cases:")
    cols = [
        "rank",
        "score",
        "case_id",
        "a0_exit",
        "a0_exit_over_analysis_max",
        "waist_growth",
        "waist_jitter_log",
        "valid_fraction",
        "plateau",
        "diameter",
        "focus",
    ]
    visible_cols = [col for col in cols if col in top.columns]
    print(top[visible_cols].to_string(index=False))


if __name__ == "__main__":
    main()

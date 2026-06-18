#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cap_guiding.beamlike_pairs import (
    BeamlikePairConfig,
    compare_beamlike_pair_csvs,
    split_pair_rows,
    write_rows_csv,
)
from cap_guiding.campaign import build_triplets, discover_cases


def _particle_summary_path(
    case_metrics_root: Path,
    case_id: str,
    particle_outdir_name: str,
) -> Path:
    return case_metrics_root / case_id / particle_outdir_name / "particle_summary.csv"


def _score_float(row: dict[str, object], key: str) -> float:
    try:
        value = float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")
    return value if value == value else float("nan")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare channel and uniform WarpX particle summaries using "
            "beamlike metrics. This consumes reduced particle_summary.csv files; "
            "it does not read openPMD diagnostics."
        )
    )

    parser.add_argument("--campaign-root", required=True)
    parser.add_argument(
        "--case-metrics-root",
        default=None,
        help="Defaults to CAMPAIGN_ROOT.",
    )
    parser.add_argument(
        "--particle-outdir-name",
        default="particle_analysis",
        help="Directory inside each case containing particle_summary.csv.",
    )
    parser.add_argument(
        "--row-selection",
        choices=["single", "last", "max-beamlike"],
        default="single",
        help=(
            "How to select one row if particle_summary.csv has several rows. "
            "Default 'single' is strict and fails on multi-row summaries."
        ),
    )
    parser.add_argument(
        "--outdir",
        default=None,
        help="Defaults to CAMPAIGN_ROOT/analysis_outputs/beamlike_pairs_<timestamp>.",
    )
    parser.add_argument("--top", type=int, default=50)
    parser.add_argument(
        "--reference-deadband-log",
        type=float,
        default=None,
        help="Deadband in log-score advantage. Default log(1.05).",
    )
    parser.add_argument(
        "--reference-scale-log",
        type=float,
        default=None,
        help="Scale in log-score advantage. Default log(1.5).",
    )
    parser.add_argument(
        "--score-floor",
        type=float,
        default=1.0,
        help="Positive floor used in log comparison of beamlike_score.",
    )

    args = parser.parse_args()

    campaign_root = Path(args.campaign_root)
    case_metrics_root = (
        Path(args.case_metrics_root)
        if args.case_metrics_root is not None
        else campaign_root
    )

    if args.outdir is None:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        outdir = campaign_root / "analysis_outputs" / f"beamlike_pairs_{stamp}"
    else:
        outdir = Path(args.outdir)

    outdir.mkdir(parents=True, exist_ok=True)

    default_config = BeamlikePairConfig()
    config = BeamlikePairConfig(
        reference_deadband_log=(
            default_config.reference_deadband_log
            if args.reference_deadband_log is None
            else args.reference_deadband_log
        ),
        reference_scale_log=(
            default_config.reference_scale_log
            if args.reference_scale_log is None
            else args.reference_scale_log
        ),
        score_floor=args.score_floor,
    )

    cases = discover_cases(campaign_root)
    triplets = build_triplets(cases)

    rows: list[dict[str, object]] = []
    skipped_without_channel = 0
    skipped_without_uniform = 0

    for triplet in triplets:
        if triplet.channel is None:
            skipped_without_channel += 1
            continue
        if triplet.uniform is None:
            skipped_without_uniform += 1
            continue

        channel_csv = _particle_summary_path(
            case_metrics_root,
            triplet.channel.case_id,
            args.particle_outdir_name,
        )
        uniform_csv = _particle_summary_path(
            case_metrics_root,
            triplet.uniform.case_id,
            args.particle_outdir_name,
        )

        row = compare_beamlike_pair_csvs(
            channel_csv=channel_csv,
            uniform_csv=uniform_csv,
            channel_case_id=triplet.channel.case_id,
            uniform_case_id=triplet.uniform.case_id,
            row_selection=args.row_selection,
            config=config,
        )

        row.update(
            {
                "pair_label": triplet.label,
                "laser_case": triplet.channel.laser_case,
                "density": triplet.channel.density,
                "plateau": triplet.channel.plateau,
                "focus": triplet.channel.focus,
                "diameter": triplet.channel.diameter,
                "case_source": triplet.channel.source,
            }
        )

        rows.append(row)

    buckets = split_pair_rows(rows)
    positive_rows = sorted(
        buckets["positive"],
        key=lambda row: _score_float(row, "beamlike_gain_score"),
        reverse=True,
    )
    neutral_rows = sorted(
        buckets["neutral"],
        key=lambda row: (
            str(row.get("plateau", "")),
            str(row.get("diameter", "")),
            str(row.get("focus", "")),
            str(row.get("channel_case_id", "")),
        ),
    )
    negative_rows = sorted(
        buckets["negative"],
        key=lambda row: _score_float(row, "beamlike_gain_score"),
    )
    failed_rows = sorted(
        buckets["failed"],
        key=lambda row: (
            str(row.get("failure_reason", "")),
            str(row.get("channel_case_id", "")),
            str(row.get("uniform_case_id", "")),
        ),
    )

    all_rows = [
        *positive_rows,
        *neutral_rows,
        *negative_rows,
        *failed_rows,
    ]

    scores_path = outdir / "beamlike_pair_scores.csv"
    positive_path = outdir / "positive_beamlike_pairs.csv"
    neutral_path = outdir / "neutral_beamlike_pairs.csv"
    negative_path = outdir / "negative_beamlike_pairs.csv"
    failed_path = outdir / "failed_beamlike_pairs.csv"
    top_path = outdir / "top_beamlike_pairs.csv"
    worst_path = outdir / "worst_negative_beamlike_pairs.csv"

    write_rows_csv(scores_path, all_rows)
    write_rows_csv(positive_path, positive_rows)
    write_rows_csv(neutral_path, neutral_rows)
    write_rows_csv(negative_path, negative_rows)
    write_rows_csv(failed_path, failed_rows)

    top_rows: list[dict[str, object]] = []
    for rank, row in enumerate(positive_rows[: args.top], start=1):
        out = dict(row)
        out["rank"] = rank
        top_rows.append(out)

    worst_rows: list[dict[str, object]] = []
    for rank, row in enumerate(negative_rows[: args.top], start=1):
        out = dict(row)
        out["rank"] = rank
        worst_rows.append(out)

    write_rows_csv(top_path, top_rows)
    write_rows_csv(worst_path, worst_rows)

    ok_count = len(positive_rows) + len(neutral_rows) + len(negative_rows)

    print("=== Beamlike channel-vs-uniform summary ===")
    print(f"campaign_root           = {campaign_root}")
    print(f"case_metrics_root       = {case_metrics_root}")
    print(f"particle_outdir_name    = {args.particle_outdir_name}")
    print(f"row_selection           = {args.row_selection}")
    print(f"outdir                  = {outdir}")
    print(f"cases discovered        = {len(cases)}")
    print(f"pairs attempted         = {len(rows)}")
    print(f"pairs ok                = {ok_count}")
    print(f"pairs positive          = {len(positive_rows)}")
    print(f"pairs neutral           = {len(neutral_rows)}")
    print(f"pairs negative          = {len(negative_rows)}")
    print(f"pairs failed            = {len(failed_rows)}")
    print(f"skipped no channel      = {skipped_without_channel}")
    print(f"skipped no uniform      = {skipped_without_uniform}")
    print(f"pair_scores             = {scores_path}")
    print(f"positive_pairs          = {positive_path}")
    print(f"neutral_pairs           = {neutral_path}")
    print(f"negative_pairs          = {negative_path}")
    print(f"failed_pairs            = {failed_path}")
    print(f"top_positive_pairs      = {top_path}")
    print(f"worst_negative_pairs    = {worst_path}")
    print()

    visible_cols = [
        "rank",
        "beamlike_gain_score",
        "beamlike_reference_factor",
        "beamlike_score_channel",
        "beamlike_score_uniform",
        "beamlike_score_delta",
        "charge_hot_pC_channel",
        "charge_hot_pC_uniform",
        "E95_hot_MeV_channel",
        "E95_hot_MeV_uniform",
        "mono_proxy_E95_over_Emax_channel",
        "mono_proxy_E95_over_Emax_uniform",
        "channel_case_id",
        "uniform_case_id",
        "plateau",
        "diameter",
        "focus",
    ]

    if top_rows:
        print(f"Top {len(top_rows)} positive beamlike channel-vs-uniform pairs:")
        for row in top_rows:
            parts = []
            for col in visible_cols:
                if col in row:
                    parts.append(f"{col}={row[col]}")
            print("  " + "  ".join(parts))
    else:
        print("No positive beamlike channel-vs-uniform pairs.")

    print()

    if worst_rows:
        print(f"Worst {len(worst_rows)} negative beamlike channel-vs-uniform pairs:")
        for row in worst_rows:
            parts = []
            for col in visible_cols:
                if col in row:
                    parts.append(f"{col}={row[col]}")
            print("  " + "  ".join(parts))
    else:
        print("No negative beamlike channel-vs-uniform pairs.")

    if failed_rows:
        print()
        print("First failures:")
        for row in failed_rows[:10]:
            print(
                f"  {row.get('channel_case_id', '')} vs "
                f"{row.get('uniform_case_id', '')}: "
                f"{row.get('failure_reason', '')}"
            )


if __name__ == "__main__":
    main()

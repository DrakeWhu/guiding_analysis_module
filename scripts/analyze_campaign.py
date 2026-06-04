#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cap_guiding.campaign import (
    build_triplets,
    case_is_ready,
    discover_cases,
    newest_h5_age_min,
    triplet_is_ready,
    triplet_ready_min_h5,
    write_campaign_report,
)
from cap_guiding.plots import save_triplet_plots
from cap_guiding.triplet import build_triplet_tables, write_triplet_tables
from cap_guiding.workflows import ensure_case_metrics


def _case_metrics_path(case_metrics_root: Path, case_id: str) -> Path:
    return case_metrics_root / case_id / "guiding_metrics.csv"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dry-run or analyze a full CLPU capillary guiding campaign."
    )

    parser.add_argument(
        "--campaign-root",
        required=True,
        help="Root containing case folders or cases_full.tsv",
    )
    parser.add_argument(
        "--outdir",
        default="analysis_outputs/campaign",
        help="Campaign analysis output root",
    )
    parser.add_argument(
        "--case-metrics-root",
        default=None,
        help="Defaults to OUTDIR/case_metrics",
    )
    parser.add_argument(
        "--triplets-root",
        default=None,
        help="Defaults to OUTDIR/triplets",
    )

    parser.add_argument(
        "--run-cases",
        action="store_true",
        help="Generate missing per-case guiding_metrics.csv",
    )
    parser.add_argument(
        "--run-triplets",
        action="store_true",
        help="Generate triplet CSVs/plots for complete ready triplets",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip existing case/triplet outputs",
    )
    parser.add_argument(
        "--overwrite-cases",
        action="store_true",
        help="Regenerate existing case metrics",
    )
    parser.add_argument(
        "--overwrite-triplets",
        action="store_true",
        help="Regenerate existing triplet outputs",
    )

    parser.add_argument("--min-h5", type=int, default=2)
    parser.add_argument(
        "--min-last-h5-age-min",
        type=float,
        default=0.0,
        help=(
            "Require the newest HDF5 file of each case to be at least this many "
            "minutes old before analyzing it. Use this to avoid reading active "
            "WarpX/openPMD diagnostics while a campaign is still running. "
            "Default 0 disables the stability gate."
        ),
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--smooth-um", type=float, default=2.0)
    parser.add_argument("--wake-behind-um", type=float, default=120.0)
    parser.add_argument("--wake-gap-um", type=float, default=5.0)
    parser.add_argument("--lambda0-m", type=float, default=0.8e-6)
    parser.add_argument("--late-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--no-case-plots", action="store_true")
    parser.add_argument("--no-triplet-plots", action="store_true")

    args = parser.parse_args()

    campaign_root = Path(args.campaign_root)
    outdir = Path(args.outdir)

    case_metrics_root = (
        Path(args.case_metrics_root)
        if args.case_metrics_root is not None
        else outdir / "case_metrics"
    )
    triplets_root = (
        Path(args.triplets_root)
        if args.triplets_root is not None
        else outdir / "triplets"
    )

    cases = discover_cases(campaign_root)
    triplets = build_triplets(cases)

    complete = [triplet for triplet in triplets if triplet.complete]
    incomplete = [triplet for triplet in triplets if not triplet.complete]

    ready_min_h5 = [
        triplet
        for triplet in complete
        if triplet_ready_min_h5(triplet, min_h5=args.min_h5)
    ]

    ready_for_analysis = [
        triplet
        for triplet in complete
        if triplet_is_ready(
            triplet,
            min_h5=args.min_h5,
            min_last_h5_age_min=args.min_last_h5_age_min,
        )
    ]

    insufficient_cases = [case for case in cases if case.h5_count < args.min_h5]

    unstable_cases = [
        case
        for case in cases
        if case.h5_count >= args.min_h5
        and not case_is_ready(
            case,
            min_h5=args.min_h5,
            min_last_h5_age_min=args.min_last_h5_age_min,
        )
    ]

    report_paths = write_campaign_report(
        cases=cases,
        triplets=triplets,
        outdir=outdir,
        min_h5=args.min_h5,
        min_last_h5_age_min=args.min_last_h5_age_min,
    )

    print("=== Campaign dry-run summary ===")
    print(f"campaign_root        = {campaign_root}")
    print(f"outdir               = {outdir}")
    print(f"case_metrics_root    = {case_metrics_root}")
    print(f"triplets_root        = {triplets_root}")
    print(f"cases detected       = {len(cases)}")
    print(f"triplets structural complete = {len(complete)}")
    print(f"triplets ready min_h5        = {len(ready_min_h5)}")
    print(f"triplets ready for analysis  = {len(ready_for_analysis)}")
    print(f"triplets incomplete          = {len(incomplete)}")
    print(f"cases insufficient h5        = {len(insufficient_cases)}")
    print(f"cases unstable h5 age        = {len(unstable_cases)}")
    print(f"min_h5                       = {args.min_h5}")
    print(f"min_last_h5_age_min          = {args.min_last_h5_age_min}")
    print("reports:")
    for path in report_paths.values():
        print(f"  - {path}")
    print("===============================")

    if not args.run_cases and not args.run_triplets:
        return

    if args.run_cases:
        for case in cases:
            if not case_is_ready(
                case,
                min_h5=args.min_h5,
                min_last_h5_age_min=args.min_last_h5_age_min,
            ):
                age = newest_h5_age_min(case.diag_dir)
                age_text = "none" if age is None else f"{age:.2f} min"
                print(
                    f"[SKIP] case not ready "
                    f"(h5={case.h5_count}, min_h5={args.min_h5}, "
                    f"newest_h5_age={age_text}, "
                    f"min_last_h5_age_min={args.min_last_h5_age_min}): "
                    f"{case.case_id}"
                )
                continue

            csv_path = _case_metrics_path(case_metrics_root, case.case_id)

            if csv_path.exists() and args.skip_existing and not args.overwrite_cases:
                print(f"[SKIP] existing case metrics: {csv_path}")
                continue

            ensure_case_metrics(
                diag=case.diag_dir,
                case_metrics_root=case_metrics_root,
                case_id=case.case_id,
                stride=args.stride,
                smooth_um=args.smooth_um,
                wake_behind_um=args.wake_behind_um,
                wake_gap_um=args.wake_gap_um,
                lambda0_m=args.lambda0_m,
                overwrite=args.overwrite_cases,
                make_plots=not args.no_case_plots,
            )

    if args.run_triplets:
        for triplet in complete:
            assert triplet.channel is not None
            assert triplet.uniform is not None
            assert triplet.vacuum is not None

            members = [triplet.channel, triplet.uniform, triplet.vacuum]

            if not triplet_is_ready(
                triplet,
                min_h5=args.min_h5,
                min_last_h5_age_min=args.min_last_h5_age_min,
            ):
                print(
                    f"[SKIP] triplet not ready "
                    f"(min_h5={args.min_h5}, "
                    f"min_last_h5_age_min={args.min_last_h5_age_min}): "
                    f"{triplet.label}"
                )
                for case in members:
                    age = newest_h5_age_min(case.diag_dir)
                    age_text = "none" if age is None else f"{age:.2f} min"
                    print(
                        f"       - {case.case_id}: "
                        f"h5={case.h5_count}, newest_h5_age={age_text}"
                    )
                continue

            channel_csv = _case_metrics_path(case_metrics_root, triplet.channel.case_id)
            uniform_csv = _case_metrics_path(case_metrics_root, triplet.uniform.case_id)
            vacuum_csv = _case_metrics_path(case_metrics_root, triplet.vacuum.case_id)

            missing_csv = [
                path
                for path in [channel_csv, uniform_csv, vacuum_csv]
                if not path.is_file()
            ]

            if missing_csv:
                print(f"[SKIP] missing case metrics for triplet {triplet.label}:")
                for path in missing_csv:
                    print(f"       - {path}")
                continue

            triplet_outdir = triplets_root / triplet.label
            wide_path = triplet_outdir / "guiding_triplet_wide.csv"

            if (
                wide_path.exists()
                and args.skip_existing
                and not args.overwrite_triplets
            ):
                print(f"[SKIP] existing triplet: {wide_path}")
                continue

            if wide_path.exists() and not args.overwrite_triplets:
                print(f"[SKIP] existing triplet, use --overwrite-triplets: {wide_path}")
                continue

            try:
                tables = build_triplet_tables(
                    channel_csv=channel_csv,
                    uniform_csv=uniform_csv,
                    vacuum_csv=vacuum_csv,
                    label=triplet.label,
                    late_fraction=args.late_fraction,
                )
            except Exception as exc:
                print(f"[FAIL] triplet {triplet.label}: {exc}")
                continue

            write_triplet_tables(tables, triplet_outdir)

            if not args.no_triplet_plots:
                save_triplet_plots(tables["wide"], triplet_outdir)


if __name__ == "__main__":
    main()

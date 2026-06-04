#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cap_guiding.plots import save_triplet_plots
from cap_guiding.triplet import build_triplet_tables, write_triplet_tables


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare one channel/uniform/vacuum triplet from guiding_metrics.csv files."
    )
    parser.add_argument("--channel", required=True, help="channel guiding_metrics.csv")
    parser.add_argument("--uniform", required=True, help="uniform guiding_metrics.csv")
    parser.add_argument("--vacuum", required=True, help="vacuum guiding_metrics.csv")
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--label", default="triplet")
    parser.add_argument("--late-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if guiding_triplet_wide.csv already exists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing triplet CSVs and plots.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only write CSV tables.",
    )
    args = parser.parse_args()

    outdir = Path(args.outdir)
    wide_path = outdir / "guiding_triplet_wide.csv"

    if wide_path.exists() and args.skip_existing and not args.overwrite:
        print(f"[SKIP] existing {wide_path}")
        return

    if wide_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {wide_path}. Use --overwrite or --skip-existing."
        )

    print("=== Capillary guiding triplet comparison ===")
    print(f"channel       = {args.channel}")
    print(f"uniform       = {args.uniform}")
    print(f"vacuum        = {args.vacuum}")
    print(f"outdir        = {outdir}")
    print(f"label         = {args.label}")
    print(f"late_fraction = {args.late_fraction}")
    print("============================================")

    missing = [
        path
        for path in [Path(args.channel), Path(args.uniform), Path(args.vacuum)]
        if not path.is_file()
    ]

    if missing:
        print()
        print("[ERROR] Missing guiding_metrics.csv file(s):")
        for path in missing:
            print(f"  - {path}")
        print()
        print(
            "compare_triplet.py only compares existing CSVs. "
            "To generate missing case metrics from WarpX diagnostics, use "
            "scripts/compare_triplet_cases.py or scripts/analyze_campaign.py."
        )
        raise SystemExit(2)

    tables = build_triplet_tables(
        channel_csv=args.channel,
        uniform_csv=args.uniform,
        vacuum_csv=args.vacuum,
        label=args.label,
        late_fraction=args.late_fraction,
    )

    write_triplet_tables(tables, outdir)

    wide = tables["wide"]
    late_summary = tables["late_summary"]
    late_ratios = tables["late_ratios"]

    print()
    print("=== Common iterations ===")
    print(f"n_common = {len(wide)}")
    print(
        "first,last iteration = "
        f"{int(wide['iteration'].iloc[0])}, {int(wide['iteration'].iloc[-1])}"
    )
    print(
        "first,last propagation = "
        f"{wide['propagation_mm'].iloc[0]:.3f}, "
        f"{wide['propagation_mm'].iloc[-1]:.3f} mm"
    )

    print()
    print("=== Late summary ===")
    print(late_summary.to_string(index=False))

    print()
    print("=== Late ratios ===")
    print(late_ratios.to_string(index=False))

    if not args.no_plots:
        save_triplet_plots(wide, outdir)


if __name__ == "__main__":
    main()

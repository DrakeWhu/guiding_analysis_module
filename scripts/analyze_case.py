#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cap_guiding.metrics import compute_case_rows, write_case_csv
from cap_guiding.plots import save_case_plots


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze one WarpX RZ openPMD diagnostic for capillary guiding metrics."
    )
    parser.add_argument(
        "--diag",
        required=True,
        help="Path to openPMD diagnostic directory, e.g. CASE/diags/diag1",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory, e.g. analysis_outputs/case_metrics/CASE_ID",
    )
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--smooth-um", type=float, default=2.0)
    parser.add_argument("--wake-behind-um", type=float, default=120.0)
    parser.add_argument("--wake-gap-um", type=float, default=5.0)
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip if guiding_metrics.csv already exists.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing guiding_metrics.csv and plots.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Only write guiding_metrics.csv.",
    )
    args = parser.parse_args()

    diag = Path(args.diag)
    outdir = Path(args.outdir)
    csv_path = outdir / "guiding_metrics.csv"

    if csv_path.exists() and args.skip_existing and not args.overwrite:
        print(f"[SKIP] existing {csv_path}")
        return

    if csv_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {csv_path}. Use --overwrite or --skip-existing."
        )

    outdir.mkdir(parents=True, exist_ok=True)

    print("=== Capillary guiding case analysis ===")
    print(f"diag           = {diag}")
    print(f"outdir         = {outdir}")
    print(f"stride         = {args.stride}")
    print(f"smooth_um      = {args.smooth_um}")
    print(f"wake_behind_um = {args.wake_behind_um}")
    print(f"wake_gap_um    = {args.wake_gap_um}")
    print("=======================================")

    rows = compute_case_rows(
        diag=diag,
        stride=args.stride,
        smooth_um=args.smooth_um,
        wake_behind_um=args.wake_behind_um,
        wake_gap_um=args.wake_gap_um,
    )

    write_case_csv(rows, csv_path)
    print(f"[OK] wrote {csv_path}")

    if not args.no_plots:
        save_case_plots(rows, outdir)


if __name__ == "__main__":
    main()

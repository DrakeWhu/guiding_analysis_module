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
from cap_guiding.workflows import ensure_case_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare one channel/uniform/vacuum triplet from WarpX RZ openPMD "
            "diagnostics, generating missing guiding_metrics.csv files if needed."
        )
    )

    parser.add_argument(
        "--channel-diag", required=True, help="channel CASE/diags/diag1"
    )
    parser.add_argument(
        "--uniform-diag", required=True, help="uniform CASE/diags/diag1"
    )
    parser.add_argument("--vacuum-diag", required=True, help="vacuum CASE/diags/diag1")

    parser.add_argument("--outdir", required=True, help="Triplet output directory")
    parser.add_argument(
        "--case-metrics-root",
        default="analysis_outputs/case_metrics",
        help="Root directory where per-case guiding_metrics.csv files are stored",
    )

    parser.add_argument("--label", default="triplet")
    parser.add_argument("--late-fraction", type=float, default=1.0 / 3.0)

    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--smooth-um", type=float, default=2.0)
    parser.add_argument("--wake-behind-um", type=float, default=120.0)
    parser.add_argument("--wake-gap-um", type=float, default=5.0)
    parser.add_argument(
        "--lambda0-m",
        type=float,
        default=0.8e-6,
        help="Laser wavelength [m] used to convert peak transverse E field to a0.",
    )

    parser.add_argument(
        "--overwrite-cases",
        action="store_true",
        help="Regenerate per-case guiding_metrics.csv even if they already exist.",
    )
    parser.add_argument(
        "--overwrite-triplet",
        action="store_true",
        help="Overwrite existing triplet outputs.",
    )
    parser.add_argument(
        "--no-case-plots",
        action="store_true",
        help="Generate missing case CSVs but skip per-case plots.",
    )
    parser.add_argument(
        "--no-triplet-plots",
        action="store_true",
        help="Write triplet CSVs but skip triplet plots.",
    )

    args = parser.parse_args()

    outdir = Path(args.outdir)
    wide_path = outdir / "guiding_triplet_wide.csv"

    if wide_path.exists() and not args.overwrite_triplet:
        raise FileExistsError(
            f"Triplet output already exists: {wide_path}. "
            "Use --overwrite-triplet to overwrite it."
        )

    print("=== Capillary guiding triplet workflow ===")
    print(f"channel diag      = {args.channel_diag}")
    print(f"uniform diag      = {args.uniform_diag}")
    print(f"vacuum diag       = {args.vacuum_diag}")
    print(f"lambda0 [m]       = {args.lambda0_m}")
    print(f"case metrics root = {args.case_metrics_root}")
    print(f"triplet outdir    = {outdir}")
    print(f"label             = {args.label}")
    print("==========================================")

    channel_csv = ensure_case_metrics(
        diag=args.channel_diag,
        case_metrics_root=args.case_metrics_root,
        stride=args.stride,
        smooth_um=args.smooth_um,
        wake_behind_um=args.wake_behind_um,
        wake_gap_um=args.wake_gap_um,
        lambda0_m=args.lambda0_m,
        overwrite=args.overwrite_cases,
        make_plots=not args.no_case_plots,
    )

    uniform_csv = ensure_case_metrics(
        diag=args.uniform_diag,
        case_metrics_root=args.case_metrics_root,
        stride=args.stride,
        smooth_um=args.smooth_um,
        wake_behind_um=args.wake_behind_um,
        wake_gap_um=args.wake_gap_um,
        lambda0_m=args.lambda0_m,
        overwrite=args.overwrite_cases,
        make_plots=not args.no_case_plots,
    )

    vacuum_csv = ensure_case_metrics(
        diag=args.vacuum_diag,
        case_metrics_root=args.case_metrics_root,
        stride=args.stride,
        smooth_um=args.smooth_um,
        wake_behind_um=args.wake_behind_um,
        wake_gap_um=args.wake_gap_um,
        lambda0_m=args.lambda0_m,
        overwrite=args.overwrite_cases,
        make_plots=not args.no_case_plots,
    )

    tables = build_triplet_tables(
        channel_csv=channel_csv,
        uniform_csv=uniform_csv,
        vacuum_csv=vacuum_csv,
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

    if not args.no_triplet_plots:
        save_triplet_plots(wide, outdir)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from cap_guiding.openpmd_io import open_series, get_iterations, describe_series
from cap_guiding.particles import (
    last_iteration,
    read_particle_dump,
    save_energy_spectrum,
    save_longitudinal_phase_space,
    summarize_dump,
    write_summary_csv,
)


def resolve_iterations(diag: Path, which: str, stride: int) -> list[int]:
    if which == "last":
        return [last_iteration(diag)]

    ts = open_series(diag)
    return get_iterations(ts, stride=stride)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze WarpX electron particle openPMD diagnostics for LWFA signatures."
    )
    parser.add_argument(
        "--diag",
        required=True,
        help="Path to particle openPMD diagnostic directory, e.g. CASE/diags/electron_particles",
    )
    parser.add_argument(
        "--outdir",
        required=True,
        help="Output directory, usually CASE/particle_analysis",
    )
    parser.add_argument("--species", default="electrons")
    parser.add_argument("--which", choices=["last", "all"], default="last")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--hot-energy-mev", type=float, default=10.0)
    parser.add_argument(
        "--longitudinal",
        choices=["x", "y", "z"],
        default="z",
        help="Longitudinal coordinate/momentum component for phase space and forward cut.",
    )
    parser.add_argument(
        "--no-forward-cut",
        action="store_true",
        help="Do not require positive longitudinal momentum for hot-electron selection.",
    )
    parser.add_argument(
        "--exit-window-mm",
        type=float,
        default=None,
        help="Optional window behind max longitudinal coordinate in each dump.",
    )
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--emax-mev", type=float, default=None)
    parser.add_argument("--max-phase-points", type=int, default=200_000)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    diag = Path(args.diag)
    outdir = Path(args.outdir)
    summary_csv = outdir / "particle_summary.csv"

    if summary_csv.exists() and args.skip_existing and not args.overwrite:
        print(f"[SKIP] existing {summary_csv}")
        return

    if summary_csv.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output already exists: {summary_csv}. Use --overwrite or --skip-existing."
        )

    print("=== Particle case analysis ===")
    print(f"diag              = {diag}")
    print(f"outdir            = {outdir}")
    print(f"species           = {args.species}")
    print(f"which             = {args.which}")
    print(f"stride            = {args.stride}")
    print(f"hot_energy_mev    = {args.hot_energy_mev}")
    print(f"longitudinal      = {args.longitudinal}")
    print(f"forward_cut       = {not args.no_forward_cut}")
    print(f"exit_window_mm    = {args.exit_window_mm}")
    print("==============================")

    ts = open_series(diag)
    print("[SERIES]", describe_series(ts))

    iterations = resolve_iterations(diag, args.which, args.stride)
    if not iterations:
        raise RuntimeError(f"No particle iterations found in {diag}")

    rows = []
    plots_dir = outdir / "plots"

    for iteration in iterations:
        print(f"[READ] iteration {iteration}")

        dump = read_particle_dump(
            diag,
            species=args.species,
            iteration=iteration,
        )

        row = summarize_dump(
            dump,
            hot_energy_mev=args.hot_energy_mev,
            longitudinal=args.longitudinal,
            exit_window_mm=args.exit_window_mm,
            forward_only=not args.no_forward_cut,
        )
        rows.append(row)

        suffix = f"it{int(iteration):08d}"

        save_energy_spectrum(
            dump,
            path=plots_dir / f"energy_spectrum_{suffix}.png",
            hot_energy_mev=args.hot_energy_mev,
            bins=args.bins,
            emax_mev=args.emax_mev,
        )

        try:
            save_longitudinal_phase_space(
                dump,
                path=plots_dir / f"longitudinal_phase_space_hot_{suffix}.png",
                hot_energy_mev=args.hot_energy_mev,
                longitudinal=args.longitudinal,
                exit_window_mm=args.exit_window_mm,
                forward_only=not args.no_forward_cut,
                max_points=args.max_phase_points,
            )
        except ValueError as exc:
            print(f"[WARN] skipping phase-space plot for iteration {iteration}: {exc}")

    write_summary_csv(rows, summary_csv)
    print(f"[OK] wrote {summary_csv}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
from __future__ import annotations

import argparse
from email import parser
import subprocess
import sys
from pathlib import Path
from cap_guiding.diagnostics import resolve_particle_diag_dir

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run particle analysis over a campaign of WarpX particle diagnostics."
    )
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument(
        "--particle-diag-name",
        default="auto",
        help=(
            "Diagnostic directory name under CASE/diags, or 'auto'. "
            "Auto maps legacy species electrons to electron_particles[/openpmd] "
            "and new species to plasma_electrons/ionized_electrons."
        ),
    )
    parser.add_argument("--species", default="electrons")
    parser.add_argument("--which", choices=["last", "all", "exit"], default="last")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--hot-energy-mev", type=float, default=10.0)
    parser.add_argument("--longitudinal", choices=["x", "y", "z"], default="z")
    parser.add_argument("--exit-window-mm", type=float, default=None)
    parser.add_argument(
        "--exit-kind", choices=["plateau", "capillary"], default="plateau"
    )
    parser.add_argument("--target-propagation-mm", type=float, default=None)
    parser.add_argument("--no-forward-cut", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--case-glob",
        default="0*_from_*",
        help="Glob for case directories under campaign root.",
    )
    parser.add_argument("--downramp-mm", type=float, default=None)
    parser.add_argument("--bins", type=int, default=200)
    parser.add_argument("--emax-mev", type=float, default=None)
    parser.add_argument("--spectrum-emin-mev", type=float, default=0.0)
    parser.add_argument("--spectrum-log-y", action="store_true")
    parser.add_argument("--max-phase-points", type=int, default=200_000)
    parser.add_argument(
        "--outdir-name",
        default="particle_analysis",
        help="Output directory name inside each case directory.",
    )
    args = parser.parse_args()

    root = Path(args.campaign_root)
    script = PROJECT_ROOT / "scripts" / "analyze_particle_case.py"

    cases = sorted(p for p in root.glob(args.case_glob) if p.is_dir())
    if not cases:
        raise RuntimeError(f"No case directories found with glob {args.case_glob!r}")

    print("=== Particle campaign analysis ===")
    print(f"campaign_root     = {root}")
    print(f"cases discovered  = {len(cases)}")
    print(f"diag name         = {args.particle_diag_name}")
    print(f"species           = {args.species}")
    print(f"which             = {args.which}")
    print(f"exit_kind         = {args.exit_kind}")
    print(f"target_prop_mm    = {args.target_propagation_mm}")
    print(f"hot_energy_mev    = {args.hot_energy_mev}")
    print("==================================")

    ok = 0
    failed = 0

    for case_dir in cases:
        try:
            diag = resolve_particle_diag_dir(
                case_dir,
                args.species,
                particle_diag_name=args.particle_diag_name,
            )
        except FileNotFoundError as exc:
            print(f"[MISSING] {exc}")
            failed += 1
            continue

        outdir = case_dir / args.outdir_name

        cmd = [
            sys.executable,
            str(script),
            "--diag",
            str(diag),
            "--outdir",
            str(outdir),
            "--species",
            args.species,
            "--which",
            args.which,
            "--stride",
            str(args.stride),
            "--hot-energy-mev",
            str(args.hot_energy_mev),
            "--longitudinal",
            args.longitudinal,
            "--exit-kind",
            args.exit_kind,
            "--bins",
            str(args.bins),
            "--spectrum-emin-mev",
            str(args.spectrum_emin_mev),
            "--max-phase-points",
            str(args.max_phase_points),
        ]

        if args.target_propagation_mm is not None:
            cmd += ["--target-propagation-mm", str(args.target_propagation_mm)]
        if args.exit_window_mm is not None:
            cmd += ["--exit-window-mm", str(args.exit_window_mm)]
        if args.no_forward_cut:
            cmd += ["--no-forward-cut"]
        if args.skip_existing:
            cmd += ["--skip-existing"]
        if args.overwrite:
            cmd += ["--overwrite"]
        if args.downramp_mm is not None:
            cmd += ["--downramp-mm", str(args.downramp_mm)]
        if args.emax_mev is not None:
            cmd += ["--emax-mev", str(args.emax_mev)]
        if args.spectrum_log_y:
            cmd += ["--spectrum-log-y"]

        print(f"[CASE] {case_dir.name}")
        result = subprocess.run(cmd, check=False)

        if result.returncode == 0:
            ok += 1
        else:
            failed += 1
            print(
                f"[ERROR] case failed with return code {result.returncode}: {case_dir}"
            )

    print("=== Particle campaign summary ===")
    print(f"ok     = {ok}")
    print(f"failed = {failed}")

    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

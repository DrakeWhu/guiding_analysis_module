#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

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
        default="electron_particles",
        help="Diagnostic directory name under CASE/diags.",
    )
    parser.add_argument("--species", default="electrons")
    parser.add_argument("--which", choices=["last", "all"], default="last")
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--hot-energy-mev", type=float, default=10.0)
    parser.add_argument("--longitudinal", choices=["x", "y", "z"], default="z")
    parser.add_argument("--exit-window-mm", type=float, default=None)
    parser.add_argument("--no-forward-cut", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--case-glob",
        default="0*_from_*",
        help="Glob for case directories under campaign root.",
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
    print(f"hot_energy_mev    = {args.hot_energy_mev}")
    print("==================================")

    ok = 0
    failed = 0

    for case_dir in cases:
        diag = case_dir / "diags" / args.particle_diag_name
        outdir = case_dir / "particle_analysis"

        if not diag.exists():
            print(f"[MISSING] {diag}")
            failed += 1
            continue

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
        ]

        if args.exit_window_mm is not None:
            cmd += ["--exit-window-mm", str(args.exit_window_mm)]
        if args.no_forward_cut:
            cmd += ["--no-forward-cut"]
        if args.skip_existing:
            cmd += ["--skip-existing"]
        if args.overwrite:
            cmd += ["--overwrite"]

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
